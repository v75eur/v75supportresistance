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

FOREX_SYMBOLS = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"}
DERIV_SYMBOLS = {"R_75": "R_75"}

# --- GESTION DU BACKTEST ---
JOURNAL_FILE = "journal.json"
if not os.path.exists(JOURNAL_FILE):
    json.dump({"trades": [], "solde": CONFIG["backtest"]["solde_depart"]}, open(JOURNAL_FILE, "w"))
JOURNAL = json.load(open(JOURNAL_FILE))

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
def sauvegarder_journal(): json.dump(JOURNAL, open(JOURNAL_FILE, "w"))

def post_ntfy(url, msg, img=None, title=None):
    if not url: return
    headers = {"Title": title or "V75 SMA"}
    if img:
        headers["Message"] = msg; headers["Filename"] = "chart.png"
        requests.post(url, data=img, headers=headers, timeout=30)
    else:
        requests.post(url, data=msg.encode("utf-8"), headers=headers, timeout=15)

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

def check_sma_croisement(df, sens):
    df["sma_f"] = df["c"].rolling(SMA_FAST).mean()
    df["sma_s"] = df["c"].rolling(SMA_SLOW).mean()
    a, b = df.iloc[-2], df.iloc[-1]
    if sens == "HAUSSE" and a["sma_f"] <= a["sma_s"] and b["sma_f"] > b["sma_s"]: return "ACHAT"
    if sens == "BAISSE" and a["sma_f"] >= a["sma_s"] and b["sma_f"] < b["sma_s"]: return "VENTE"
    return None

# --- BACKTEST ---
def verifier_trades(df_15):
    trades = JOURNAL["trades"]
    solde = JOURNAL["solde"]
    updated = False
    for t in trades:
        if t["statut"] == "FERME": continue
        for i in range(t.get("idx_ouverture", 0) + 1, len(df_15)):
            r = df_15.iloc[i]
            if t["sens"] == "ACHAT":
                if r["l"] <= t["sl"]: t["statut"] = "FERME"; t["resultat"] = "LOSS"; solde -= t["risque"]; break
                if r["h"] >= t["tp1"] and not t["tp1_touche"]: t["tp1_touche"] = True; solde += t["gain_tp1"]; t["sl"] = t["entree"]; updated = True
                if r["h"] >= t["tp2"]: t["statut"] = "FERME"; t["resultat"] = "WIN"; solde += t["gain_tp2"]; break
            else:
                if r["h"] >= t["sl"]: t["statut"] = "FERME"; t["resultat"] = "LOSS"; solde -= t["risque"]; break
                if r["l"] <= t["tp1"] and not t["tp1_touche"]: t["tp1_touche"] = True; solde += t["gain_tp1"]; t["sl"] = t["entree"]; updated = True
                if r["l"] <= t["tp2"]: t["statut"] = "FERME"; t["resultat"] = "WIN"; solde += t["gain_tp2"]; break
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

def rapport_backtest(now):
    t = JOURNAL["trades"]; solde = JOURNAL["solde"]
    g = sum(1 for x in t if x.get("resultat") == "WIN"); p = sum(1 for x in t if x.get("resultat") == "LOSS")
    total = g + p; wr = (g / total * 100) if total > 0 else 0
    txt = f"\n\n📊 **BACKTEST SMA**\n💰 Compte : {solde:.2f}$ ({(solde - CONFIG['backtest']['solde_depart']):+.2f}$)\n📈 Trades : {total} ({g} WIN / {p} LOSS)\n🎯 Winrate : {wr:.0f}%"
    en_cours = [x for x in t if x["statut"] == "EN COURS"]
    if en_cours:
        d = en_cours[-1]
        txt += f"\n🔹 Trade en cours : {d['sens']} à {d['entree']:.2f}\n• SL : {d['sl']:.2f}\n• TP1 : {d['tp1']:.2f} {'✅' if d['tp1_touche'] else '⏳'}\n• TP2 : {d['tp2']:.2f} ⏳"
        if d['tp1_touche']: txt += f"\n• Break-Even : ✅ SL à {d['sl']:.2f}"
    else: txt += "\n🔹 Aucun trade en cours."
    return txt

def figure(df, tend, sig, tf_label, pair_name):
    df = df.tail(50).reset_index(drop=True)
    df["sma_f"] = df["c"].rolling(SMA_FAST).mean()
    df["sma_s"] = df["c"].rolling(SMA_SLOW).mean()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('#1e1e1e'); ax.set_facecolor('#1e1e1e')
    for i, r in df.iterrows():
        col = '#26a69a' if r['c'] >= r['o'] else '#ef5350'
        ax.plot([i, i], [r['l'], r['h']], color=col, lw=1.2)
        ax.add_patch(Rectangle((i-0.3, min(r['o'], r['c'])), 0.6, abs(r['o']-r['c']), facecolor=col, edgecolor=col))
    ax.plot(df.index, df["sma_f"], color="cyan", lw=1.2, label="SMA 3")
    ax.plot(df.index, df["sma_s"], color="yellow", lw=1.2, label="SMA 25")
    ax.set_title(f"{pair_name} - {tf_label} | {tend} | {sig or ''}", color="white")
    ax.tick_params(colors='white'); ax.grid(color='#333333', ls='--', alpha=0.5)
    ax.legend(loc="upper left", facecolor='#1e1e1e', labelcolor='white')
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return buf.getvalue()

def analyser(pair_key, source, now):
    try:
        if source == "yahoo":
            df_15 = None  # Non implémenté pour le Forex dans ce bot
        else:
            df_15 = fetch_deriv(DERIV_SYMBOLS[pair_key], "15m")
            df_30 = fetch_deriv(DERIV_SYMBOLS[pair_key], "30m")
            df_60 = fetch_deriv(DERIV_SYMBOLS[pair_key], "60m")
            
        if df_15 is None or df_30 is None or df_60 is None: return
        
        verifier_trades(df_15)
        
        tend_h1 = tendance(df_60)
        tend_m30 = tendance(df_30)
        aligne = (tend_h1 == tend_m30 and tend_h1 in ("HAUSSE", "BAISSE"))
        
        sig = None
        if aligne:
            sig = check_sma_croisement(df_15, tend_h1)
        
        log(f"{pair_key} ({source}) analysé. H1:{tend_h1} M30:{tend_m30} Signal: {sig}")
        
        if sig and not any(x["statut"] == "EN COURS" for x in JOURNAL["trades"]):
            atr = (df_15["h"] - df_15["l"]).rolling(14).mean().iloc[-1]
            ouvrir_trade(sig, df_15["c"].iloc[-1], atr, len(df_15)-1, now)
        
        prix = df_15["c"].iloc[-1]
        sma3 = df_15["c"].rolling(SMA_FAST).mean().iloc[-1]
        sma25 = df_15["c"].rolling(SMA_SLOW).mean().iloc[-1]
        
        texte = f"📊 **RAPPORT SMA {pair_key}** - {now.strftime('%a %d/%m %H:%M')}\n\n"
        texte += f"🔹 **TENDANCE**\n• H1 : {tend_h1}\n• M30 : {tend_m30}\n"
        texte += f"• Alignement : {'✅ OUI' if aligne else '❌ NON'}\n\n"
        texte += f"🔸 **M15**\n• Prix : {prix:.2f}\n• SMA3 : {sma3:.2f}\n• SMA25 : {sma25:.2f}\n\n"
        texte += f"🎯 **SIGNAL**\n"
        if sig: texte += f"• {sig} CONFIRMÉ ! ✅\n• Conseil : Entrée manuelle possible.\n"
        else:
            if not aligne: texte += f"• Aucun signal. ⏳\n• Raison : H1 et M30 non alignés.\n"
            else: texte += f"• Aucun signal. ⏳\n• Raison : Tendance alignée, pas de croisement M15.\n"
        
        texte += rapport_backtest(now)
        
        topic = CONFIG["topics"]["principal"]
        if CONFIG["mode_affichage"] == "RAPPORT" or sig:
            for tf, df, tend in [("H1", df_60, tend_h1), ("M30", df_30, tend_m30), ("M15", df_15, tend_h1)]:
                img = figure(df, tend, sig if tf == "M15" else None, tf, pair_key)
                post_ntfy(topic, f"{pair_key} - {tf}", img, f"{pair_key} {tf}")
                time.sleep(1.5)
            post_ntfy(topic, texte, None, f"Rapport SMA {pair_key}")
            log(f"Envoyé à Admin pour {pair_key}")
    except Exception as e: log(f"Erreur {pair_key}: {e}")

def main():
    now = datetime.now(BENIN)
    log("DÉMARRAGE SMA")
    
    if CONFIG["marches"]["forex"]["actif"]:
        for k in CONFIG["marches"]["forex"]["paires"]:
            if CONFIG["marches"]["forex"]["paires"][k]: analyser(k, "yahoo", now)
            
    if CONFIG["marches"]["deriv"]["actif"]:
        for k in CONFIG["marches"]["deriv"]["indices"]:
            if CONFIG["marches"]["deriv"]["indices"][k]: analyser(k, "deriv", now)

if __name__ == "__main__": main()
