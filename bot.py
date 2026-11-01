import os, io, time, requests, pytz, json, asyncio, websockets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
import pandas as pd
import numpy as np

BENIN = pytz.timezone('Africa/Porto-Novo')
CONFIG = json.load(open("config.json"))
SMA_FAST, SMA_SLOW = 3, 25

FOREX_SYMBOLS = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"}
DERIV_SYMBOLS = {"R_75": "R_75"}

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def post_ntfy(url, msg, img=None, title=None):
    if not url: return
    headers = {"Title": title or "V75 S/R"}
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
    # Tendance basée sur la SMA 3 vs SMA 25 ET la pente de la SMA 25
    df["sma_f"] = df["c"].rolling(SMA_FAST).mean()
    df["sma_s"] = df["c"].rolling(SMA_SLOW).mean()
    last = df.iloc[-1]
    pente = df["sma_s"].iloc[-1] - df["sma_s"].iloc[-5]  # Pente sur 5 bougies
    
    if last["sma_f"] > last["sma_s"] and pente > 0: return "HAUSSE"
    if last["sma_f"] < last["sma_s"] and pente < 0: return "BAISSE"
    return "NEUTRE"

def get_zones(df_h1):
    df_h1 = df_h1.tail(20).reset_index(drop=True)
    res = df_h1["h"].max()
    sup = df_h1["l"].min()
    return res, sup

def check_cassure(df, res, sup, sens):
    if sens == "HAUSSE":
        if df["c"].iloc[-2] <= res and df["c"].iloc[-1] > res: return "ACHAT"
    if sens == "BAISSE":
        if df["c"].iloc[-2] >= sup and df["c"].iloc[-1] < sup: return "VENTE"
    return None

def figure(df, tend, sig, tf_label, pair_name, res, sup, canal_haut, canal_bas):
    df = df.tail(50).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('#1e1e1e'); ax.set_facecolor('#1e1e1e')
    
    for i, r in df.iterrows():
        col = '#26a69a' if r['c'] >= r['o'] else '#ef5350'
        ax.plot([i, i], [r['l'], r['h']], color=col, lw=1.2)
        ax.add_patch(Rectangle((i-0.3, min(r['o'], r['c'])), 0.6, abs(r['o']-r['c']), facecolor=col, edgecolor=col))
    
    if res: ax.axhline(res, color="red", ls="--", lw=1.5, label=f"Résistance {res:.1f}")
    if sup: ax.axhline(sup, color="lime", ls="--", lw=1.5, label=f"Support {sup:.1f}")
    
    # Tracé du canal (si les données sont fournies)
    if canal_haut is not None and canal_bas is not None:
        x = np.arange(len(df))
        # On s'assure que le canal a la bonne taille
        if len(canal_haut) >= len(df):
            canal_haut = canal_haut[-len(df):]
            canal_bas = canal_bas[-len(df):]
        else:
            canal_haut = np.pad(canal_haut, (len(df) - len(canal_haut), 0), mode='edge')
            canal_bas = np.pad(canal_bas, (len(df) - len(canal_bas), 0), mode='edge')
        ax.plot(x, canal_haut, color="red", alpha=0.7, lw=1.5, ls="-.", label="Canal Haut")
        ax.plot(x, canal_bas, color="lime", alpha=0.7, lw=1.5, ls="-.", label="Canal Bas")
        
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
        
        tend_h1 = tendance(df_60)
        tend_m30 = tendance(df_30)
        aligne = (tend_h1 == tend_m30 and tend_h1 in ("HAUSSE", "BAISSE"))
        
        res, sup = get_zones(df_60)
        
        # Calcul du canal pour H1 (50 bougies)
        df_h1_50 = df_60.tail(50).reset_index(drop=True)
        x = np.arange(len(df_h1_50))
        z = np.polyfit(x, df_h1_50["c"], 1)
        p = np.poly1d(z)
        canal_haut = p(x) + (df_h1_50["h"] - df_h1_50["c"]).mean()
        canal_bas = p(x) - (df_h1_50["c"] - df_h1_50["l"]).mean()
        
        # Calcul du canal pour M30 (50 bougies)
        df_m30_50 = df_30.tail(50).reset_index(drop=True)
        x_m30 = np.arange(len(df_m30_50))
        z_m30 = np.polyfit(x_m30, df_m30_50["c"], 1)
        p_m30 = np.poly1d(z_m30)
        canal_haut_m30 = p_m30(x_m30) + (df_m30_50["h"] - df_m30_50["c"]).mean()
        canal_bas_m30 = p_m30(x_m30) - (df_m30_50["c"] - df_m30_50["l"]).mean()
        
        sig = None
        if aligne:
            sig = check_cassure(df_15, res, sup, tend_h1)
        
        log(f"{pair_key} ({source}) analysé. H1:{tend_h1} M30:{tend_m30} Signal: {sig}")
        
        prix = df_15["c"].iloc[-1]
        
        texte = f"📊 **RAPPORT S/R {pair_key}** - {now.strftime('%a %d/%m %H:%M')}\n\n"
        texte += f"🔹 **TENDANCE DE FOND**\n• H1 : {tend_h1}\n• M30 : {tend_m30}\n"
        texte += f"• Alignement : {'✅ OUI' if aligne else '❌ NON'}\n\n"
        
        texte += f"📏 **ZONES (Sur 20 bougies H1)**\n• Résistance : {res:.2f}\n• Support : {sup:.2f}\n• Prix actuel : {prix:.2f}\n\n"
        
        texte += f"🎯 **SIGNAL**\n"
        if sig:
            texte += f"• {sig} CONFIRMÉ ! 🚀\n• La bougie M15 a clôturé {'au-dessus de la résistance' if sig == 'ACHAT' else 'en-dessous du support'}.\n• Conseil : Cassure validée. Place ton Stop Loss de l'autre côté de la ligne.\n"
        else:
            if not aligne:
                texte += f"• Aucun signal. ⏳\n• Raison : H1 et M30 ne sont pas alignés.\n"
            else:
                texte += f"• Aucun signal. ⏳\n• Raison : Tendance alignée ({tend_h1}), mais pas de cassure confirmée.\n"
                if tend_h1 == "HAUSSE": texte += f"• Conseil : Attends que le prix clôture au-dessus de {res:.2f}.\n"
                else: texte += f"• Conseil : Attends que le prix clôture en-dessous de {sup:.2f}.\n"

        topic = CONFIG["topics"]["principal"]
        if CONFIG["mode_affichage"] == "RAPPORT" or sig:
            for tf, df, tend, ch, cb in [
                ("H1", df_60, tend_h1, canal_haut, canal_bas),
                ("M30", df_30, tend_m30, canal_haut_m30, canal_bas_m30),
                ("M15", df_15, tend_h1, None, None)
            ]:
                img = figure(df, tend, sig if tf == "M15" else None, tf, pair_key, res, sup, ch, cb)
                post_ntfy(topic, f"{pair_key} - {tf}", img, f"{pair_key} {tf}")
                time.sleep(1.5)
            
            post_ntfy(topic, texte, None, f"Rapport S/R {pair_key}")
            log(f"Envoyé à Admin pour {pair_key}")
    except Exception as e: log(f"Erreur {pair_key}: {e}")

def main():
    now = datetime.now(BENIN)
    log("DÉMARRAGE S/R")
    
    if CONFIG["marches"]["forex"]["actif"]:
        for k in CONFIG["marches"]["forex"]["paires"]:
            if CONFIG["marches"]["forex"]["paires"][k]: analyser(k, "yahoo", now)
            
    if CONFIG["marches"]["deriv"]["actif"]:
        for k in CONFIG["marches"]["deriv"]["indices"]:
            if CONFIG["marches"]["deriv"]["indices"][k]: analyser(k, "deriv", now)

if __name__ == "__main__": main()
