import os, io, time, requests, pytz, json, asyncio, websockets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
import pandas as pd

BENIN = pytz.timezone('Africa/Porto-Novo')
CONFIG = json.load(open("config.json"))
SMA_FAST, SMA_SLOW = 3, 25
JOURNAL_FILE = "journal.json"

if not os.path.exists(JOURNAL_FILE):
    json.dump({"trades": [], "solde": CONFIG["backtest"]["solde_depart"]}, open(JOURNAL_FILE, "w"))
JOURNAL = json.load(open(JOURNAL_FILE))

FOREX_SYMBOLS = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"}
DERIV_SYMBOLS = {"R_75": "R_75"}

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
def sauvegarder_journal(): json.dump(JOURNAL, open(JOURNAL_FILE, "w"))

def post_ntfy(url, msg, img=None, title=None):
    if not url: return
    headers = {"Title": title or "V75 S/R"}
    if img:
        headers["Message"] = msg; headers["Filename"] = "chart.png"
        requests.post(url, data=img, headers=headers, timeout=30)
    else:
        requests.post(url, data=msg.encode("utf-8"), headers=headers, timeout=15)

def fetch_yahoo(symbol, interval):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range=5d"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    if r.status_code != 200: return None
    j = r.json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({"t": pd.to_datetime(j["timestamp"], unit="s", utc=True), "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"]}).dropna()
    return df.iloc[:-1].reset_index(drop=True) if len(df) > 30 else None

async def _fetch_deriv(symbol, granularity, count=100):
    uri = 'wss://api.derivws.com/trading/v1/options/ws/public'
    async with websockets.connect(uri, open_timeout=30) as ws:
        await ws.send(json.dumps({'ticks_history': symbol, 'style': 'candles', 'granularity': granularity, 'count': count, 'end': 'latest'}))
        while True:
            data = json.loads(await ws.recv())
            if 'candles' in data: return pd.DataFrame(data['candles'])

def fetch_deriv(symbol, tf):
    gran = {"15m": 900, "30m": 1800, "60m": 3600}.get(tf, 900)
    try:
        df = asyncio.run(_fetch_deriv(symbol, gran))
        if len(df) <= 1: return None
        df = df.rename(columns={"high": "h", "low": "l", "open": "o", "close": "c"})
        return df.iloc[:-1].reset_index(drop=True)
    except Exception as e:
        log(f"Erreur Deriv {symbol}: {e}")
        return None

def tendance(df):
    df["sma_f"] = df["c"].rolling(SMA_FAST).mean()
    df["sma_s"] = df["c"].rolling(SMA_SLOW).mean()
    last = df.iloc[-1]
    if last["sma_f"] > last["sma_s"]: return "HAUSSE"
    if last["sma_f"] < last["sma_s"]: return "BAISSE"
    return "NEUTRE"

# --- DÉTECTION DE L'ORDER BLOCK ---
def detecter_order_block(df):
    # On cherche le dernier Order Block : la dernière bougie haussière avant une forte baisse (ou l'inverse)
    for i in range(len(df) - 2, 0, -1):
        bougie = df.iloc[i]
        suivante = df.iloc[i+1] if i+1 < len(df) else None
        
        if suivante is None: continue
        
        # Order Block Baissier (bougie verte suivie d'une grosse bougie rouge)
        if bougie["c"] > bougie["o"] and suivante["c"] < suivante["o"] and (suivante["o"] - suivante["c"]) > (bougie["c"] - bougie["o"]):
            return "BAISSIER", bougie["o"], bougie["c"]
            
        # Order Block Haussier (bougie rouge suivie d'une grosse bougie verte)
        if bougie["c"] < bougie["o"] and suivante["c"] > suivante["o"] and (suivante["c"] - suivante["o"]) > (bougie["o"] - bougie["c"]):
            return "HAUSSIER", bougie["c"], bougie["o"]
            
    return None, None, None

def check_cassure_ob(df, ob_type, ob_bas, ob_haut, sens):
    if sens == "HAUSSE" and ob_type == "HAUSSIER":
        if df["c"].iloc[-2] <= ob_haut and df["c"].iloc[-1] > ob_haut: return "ACHAT"
    if sens == "BAISSE" and ob_type == "BAISSIER":
        if df["c"].iloc[-2] >= ob_bas and df["c"].iloc[-1] < ob_bas: return "VENTE"
    return None

def verifier_trades(df_15):
    trades = JOURNAL["trades"]
    solde = JOURNAL["solde"]
    updated = False
    for t in trades:
        if t["statut"] == "FERME": continue
        for i in range(t.get("idx_ouverture", 0) + 1, len(df_15)):
            r = df_15.iloc[i]
            if t["sens"] == "ACHAT":
                if r["l"] <= t["sl"]: t["statut"] = "FERME"; t["resultat"] = "PERDU"; solde -= t["risque"]; break
                if r["h"] >= t["tp1"] and not t["tp1_touche"]: t["tp1_touche"] = True; solde += t["gain_tp1"]; t["sl"] = t["entree"]; updated = True
                if r["h"] >= t["tp2"]: t["statut"] = "FERME"; t["resultat"] = "GAGNE"; solde += t["gain_tp2"]; break
            else:
                if r["h"] >= t["sl"]: t["statut"] = "FERME"; t["resultat"] = "PERDU"; solde -= t["risque"]; break
                if r["l"] <= t["tp1"] and not t["tp1_touche"]: t["tp1_touche"] = True; solde += t["gain_tp1"]; t["sl"] = t["entree"]; updated = True
                if r["l"] <= t["tp2"]: t["statut"] = "FERME"; t["resultat"] = "GAGNE"; solde += t["gain_tp2"]; break
    JOURNAL["solde"] = solde
    if updated: sauvegarder_journal()

def ouvrir_trade(sens, prix, atr, idx, now):
    cfg = CONFIG["backtest"]
    sl_dist = atr * cfg["ratio_sl"]
    tp1_dist = sl_dist * cfg["ratio_tp1"]
    tp2_dist = sl_dist * cfg["ratio_tp2"]
    if sens == "ACHAT": sl = prix - sl_dist; tp1 = prix + tp1_dist; tp2 = prix + tp2_dist
    else: sl = prix + sl_dist; tp1 = prix - tp1_dist; tp2 = prix - tp2_dist
    trade = {"date": str(now), "sens": sens, "entree": prix, "sl": sl, "tp1": tp1, "tp2": tp2,
             "tp1_touche": False, "risque": abs(sl_dist * cfg["valeur_pip"] * 100),
             "gain_tp1": abs(tp1_dist * cfg["valeur_pip"] * 100), "gain_tp2": abs(tp2_dist * cfg["valeur_pip"] * 100),
             "statut": "EN COURS", "resultat": None, "idx_ouverture": idx}
    JOURNAL["trades"].append(trade)
    sauvegarder_journal()
    log(f"BACKTEST : {sens} ouvert à {prix:.2f}")

def generer_rapport_professionnel():
    t = JOURNAL["trades"]
    solde = JOURNAL["solde"]
    depart = CONFIG["backtest"]["solde_depart"]
    fermes = [x for x in t if x["statut"] == "FERME"]
    gagnants = [x for x in fermes if x.get("resultat") == "GAGNE"]
    perdants = [x for x in fermes if x.get("resultat") == "PERDU"]
    
    total = len(fermes)
    taux_reussite = (len(gagnants) / total * 100) if total > 0 else 0
    pnl = solde - depart
    gain_moyen = sum(x["gain_tp2"] for x in gagnants) / len(gagnants) if gagnants else 0
    perte_moyenne = sum(x["risque"] for x in perdants) / len(perdants) if perdants else 0
    total_gains = sum(x["gain_tp2"] for x in gagnants)
    total_pertes = sum(x["risque"] for x in perdants)
    facteur_profit = total_gains / total_pertes if total_pertes > 0 else 0
    perte_max = 0
    current = 0
    for x in fermes:
        if x.get("resultat") == "PERDU": current += x["risque"]; perte_max = max(perte_max, current)
        else: current = 0
    serie = "".join(["G" if x.get("resultat") == "GAGNE" else "P" for x in fermes[-10:]])
    
    txt = f"\n\n📊 **STATISTIQUES DE TRADING**\n"
    txt += f"━━━━━━━━━━━━━━━━━━━━━\n"
    txt += f"💰 **COMPTE VIRTUEL**\n"
    txt += f"• Solde actuel : {solde:.2f}$\n"
    txt += f"• Solde de départ : {depart:.2f}$\n"
    txt += f"• Gain/Perte : {pnl:+.2f}$ ({(pnl/depart*100):+.1f}%)\n"
    txt += f"━━━━━━━━━━━━━━━━━━━━━\n"
    txt += f"📈 **PERFORMANCE**\n"
    txt += f"• Trades clôturés : {total}\n"
    txt += f"• Gagnants : {len(gagnants)} | Perdants : {len(perdants)}\n"
    txt += f"• Taux de réussite : {taux_reussite:.0f}%\n"
    txt += f"• Facteur de profit : {facteur_profit:.2f}\n"
    txt += f"• Gain moyen : {gain_moyen:.2f}$\n"
    txt += f"• Perte moyenne : {perte_moyenne:.2f}$\n"
    txt += f"• Perte maximale : {perte_max:.2f}$\n"
    txt += f"• Série récente (10 derniers) : {serie if serie else 'Aucun'}\n"
    txt += f"━━━━━━━━━━━━━━━━━━━━━\n"
    
    en_cours = [x for x in t if x["statut"] == "EN COURS"]
    if en_cours:
        d = en_cours[-1]
        txt += f"🔹 **TRADE EN COURS**\n"
        txt += f"• Sens : {d['sens']}\n"
        txt += f"• Entrée : {d['entree']:.2f}\n"
        txt += f"• SL : {d['sl']:.2f}\n"
        txt += f"• TP1 : {d['tp1']:.2f} {'✅ Touché' if d['tp1_touche'] else '⏳ En attente'}\n"
        txt += f"• TP2 : {d['tp2']:.2f} ⏳ En attente\n"
        if d['tp1_touche']: txt += f"• Break-Even : ✅ SL remonté à {d['sl']:.2f}\n"
    else:
        txt += f"🔹 Aucun trade en cours. En attente d'un signal.\n"
    return txt

def figure(df, tend, sig, tf_label, pair_name, ob_type, ob_bas, ob_haut):
    df = df.tail(50).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('#1e1e1e'); ax.set_facecolor('#1e1e1e')
    for i, r in df.iterrows():
        col = '#26a69a' if r['c'] >= r['o'] else '#ef5350'
        ax.plot([i, i], [r['l'], r['h']], color=col, lw=1.2)
        ax.add_patch(Rectangle((i-0.3, min(r['o'], r['c'])), 0.6, abs(r['o']-r['c']), facecolor=col, edgecolor=col))
    
    # Tracé de l'Order Block (rectangle)
    if ob_type == "BAISSIER":
        ax.axhspan(ob_bas, ob_haut, color='red', alpha=0.2, label=f"Order Block Baissier")
        ax.axhline(ob_bas, color="red", ls="--", lw=1.5)
        ax.axhline(ob_haut, color="red", ls="--", lw=1.5)
    elif ob_type == "HAUSSIER":
        ax.axhspan(ob_bas, ob_haut, color='lime', alpha=0.2, label=f"Order Block Haussier")
        ax.axhline(ob_bas, color="lime", ls="--", lw=1.5)
        ax.axhline(ob_haut, color="lime", ls="--", lw=1.5)
        
    if tf_label == "M15":
        en_cours = [x for x in JOURNAL["trades"] if x["statut"] == "EN COURS"]
        if en_cours:
            t = en_cours[-1]
            ax.axhline(t["entree"], color="white", ls="-", lw=1.5, label=f"Entrée {t['entree']:.2f}")
            ax.axhline(t["sl"], color="red", ls="--", lw=1.5, label=f"SL {t['sl']:.2f}")
            ax.axhline(t["tp1"], color="lime", ls=":", lw=1.5, label=f"TP1 {t['tp1']:.2f}")
            ax.axhline(t["tp2"], color="green", ls=":", lw=1.5, label=f"TP2 {t['tp2']:.2f}")
            
    ax.set_title(f"{pair_name} - {tf_label} | {tend} | {sig or ''}", color="white")
    ax.tick_params(colors='white'); ax.grid(color='#333333', ls='--', alpha=0.5)
    ax.legend(loc="upper left", facecolor='#1e1e1e', labelcolor='white')
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return buf.getvalue()

def analyser(pair_key, source, now):
    try:
        if source == "yahoo":
            df_15 = fetch_yahoo(FOREX_SYMBOLS[pair_key], "15m")
            df_30 = fetch_yahoo(FOREX_SYMBOLS[pair_key], "30m")
            df_60 = fetch_yahoo(FOREX_SYMBOLS[pair_key], "60m")
        else:
            df_15 = fetch_deriv(DERIV_SYMBOLS[pair_key], "15m")
            df_30 = fetch_deriv(DERIV_SYMBOLS[pair_key], "30m")
            df_60 = fetch_deriv(DERIV_SYMBOLS[pair_key], "60m")
            
        if df_15 is None or df_30 is None or df_60 is None: return
        
        verifier_trades(df_15)
        
        tend_h1 = tendance(df_60)
        tend_m30 = tendance(df_30)
        aligne = (tend_h1 == tend_m30 and tend_h1 in ("HAUSSE", "BAISSE"))
        
        ob_type, ob_bas, ob_haut = detecter_order_block(df_60)
        
        sig = None
        if aligne and ob_type:
            sig = check_cassure_ob(df_15, ob_type, ob_bas, ob_haut, tend_h1)
        
        log(f"{pair_key} analysé. H1:{tend_h1} M30:{tend_m30} OB:{ob_type} Signal: {sig}")
        
        if sig and not any(x["statut"] == "EN COURS" for x in JOURNAL["trades"]):
            atr = (df_15["h"] - df_15["l"]).rolling(14).mean().iloc[-1]
            ouvrir_trade(sig, df_15["c"].iloc[-1], atr, len(df_15)-1, now)
        
        texte = f"📊 **RAPPORT ORDER BLOCK {pair_key}** - {now.strftime('%a %d/%m %H:%M')}\n\n"
        texte += f"🔹 **TENDANCE**\n• H1 : {tend_h1}\n• M30 : {tend_m30}\n"
        texte += f"• Alignement : {'✅ OUI' if aligne else '❌ NON'}\n\n"
        if ob_type:
            texte += f"📏 **ORDER BLOCK DÉTECTÉ**\n• Type : {ob_type}\n• Zone : {ob_bas:.2f} - {ob_haut:.2f}\n\n"
        texte += f"🎯 **SIGNAL**\n"
        if sig: texte += f"• {sig} CONFIRMÉ ! ✅ (Cassure de l'OB)\n"
        else:
            if not aligne: texte += f"• Aucun signal (H1/M30 non alignés).\n"
            else: texte += f"• Aucun signal (pas de cassure de l'OB).\n"
        
        texte += generer_rapport_professionnel()
        
        topic = CONFIG["topics"]["principal"]
        for tf, df, tend in [("H1", df_60, tend_h1), ("M30", df_30, tend_m30), ("M15", df_15, tend_h1)]:
            img = figure(df, tend, sig if tf == "M15" else None, tf, pair_key, ob_type, ob_bas, ob_haut)
            post_ntfy(topic, f"{pair_key} - {tf}", img, f"{pair_key} {tf}")
            time.sleep(1.5)
        post_ntfy(topic, texte, None, f"Rapport OB {pair_key}")
    except Exception as e: log(f"Erreur {pair_key}: {e}")

def main():
    now = datetime.now(BENIN)
    log("DÉMARRAGE ORDER BLOCK")
    if CONFIG["marches"]["forex"]["actif"]:
        for k in CONFIG["marches"]["forex"]["paires"]:
            if CONFIG["marches"]["forex"]["paires"][k]: analyser(k, "yahoo", now)
    if CONFIG["marches"]["deriv"]["actif"]:
        for k in CONFIG["marches"]["deriv"]["indices"]:
            if CONFIG["marches"]["deriv"]["indices"][k]: analyser(k, "deriv", now)

if __name__ == "__main__": main()
