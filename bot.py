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
    df["sma_f"] = df["c"].rolling(SMA_FAST).mean()
    df["sma_s"] = df["c"].rolling(SMA_SLOW).mean()
    last = df.iloc[-1]
    if last["sma_f"] > last["sma_s"]: return "HAUSSE"
    if last["sma_f"] < last["sma_s"]: return "BAISSE"
    return "NEUTRE"

def check_cassure_sr(df, sup, res, sens):
    derniere = df.iloc[-1]
    precedente = df.iloc[-2]
    if sens == "HAUSSE" and precedente["c"] <= res and derniere["c"] > res: return "ACHAT"
    if sens == "BAISSE" and precedente["c"] >= sup and derniere["c"] < sup: return "VENTE"
    return None

def figure(df, tend, sig, tf_label, sup, res):
    df = df.tail(50).reset_index(drop=True)
    df["sma_f"] = df["c"].rolling(SMA_FAST).mean()
    df["sma_s"] = df["c"].rolling(SMA_SLOW).mean()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('#1e1e1e'); ax.set_facecolor('#1e1e1e')
    for i, r in df.iterrows():
        col = '#26a69a' if r['c'] >= r['o'] else '#ef5350'
        ax.plot([i, i], [r['l'], r['h']], color=col, lw=1.2)
        ax.add_patch(Rectangle((i-0.3, min(r['o'], r['c'])), 0.6, abs(r['o']-r['c']), facecolor=col, edgecolor=col))
    if sup: ax.axhline(sup, color="lime", ls="--", lw=1.5, label="Support")
    if res: ax.axhline(res, color="red", ls="--", lw=1.5, label="Résistance")
    ax.set_title(f"R_75 - {tf_label} | {tend} | {sig or ''}", color="white")
    ax.tick_params(colors='white'); ax.grid(color='#333333', ls='--', alpha=0.5)
    ax.legend(loc="upper left", facecolor='#1e1e1e', labelcolor='white')
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return buf.getvalue()

def main():
    now = datetime.now(BENIN)
    log("DÉMARRAGE S/R")
    
    df_15 = fetch_deriv("R_75", "15m")
    df_30 = fetch_deriv("R_75", "30m")
    df_60 = fetch_deriv("R_75", "60m")
    
    if df_15 is None or df_30 is None or df_60 is None: return
    
    tend_h1 = tendance(df_60)
    tend_m30 = tendance(df_30)
    aligne = (tend_h1 == tend_m30 and tend_h1 in ("HAUSSE", "BAISSE"))
    
    res = df_60["h"].tail(50).max()
    sup = df_60["l"].tail(50).min()
    
    sig = None
    if aligne:
        sig = check_cassure_sr(df_15, sup, res, tend_h1)
    
    log(f"R_75 analysé. H1:{tend_h1} M30:{tend_m30} Signal: {sig}")
    
    prix = df_15["c"].iloc[-1]
    
    # --- RAPPORT PÉDAGOGIQUE S/R ---
    texte = f"📊 **RAPPORT S/R R_75** - {now.strftime('%a %d/%m %H:%M')}\n\n"
    texte += f"🔹 **TENDANCE DE FOND**\n"
    texte += f"• H1 : {tend_h1}\n"
    texte += f"• M30 : {tend_m30}\n"
    texte += f"• Alignement : {'✅ OUI' if aligne else '❌ NON (Pas de trade contre la tendance)'}\n\n"
    
    texte += f"📏 **ZONES CLÉS (H1)**\n"
    texte += f"• Résistance : {res:.2f}\n"
    texte += f"• Support : {sup:.2f}\n"
    texte += f"• Prix actuel : {prix:.2f}\n\n"
    
    texte += f"🎯 **SIGNAL DE CASSURE**\n"
    if sig:
        texte += f"• {sig} CONFIRMÉ ! 🚀\n"
        texte += f"• La bougie M15 a clôturé {'au-dessus de la résistance' if sig == 'ACHAT' else 'en-dessous du support'}.\n"
        texte += f"• Conseil : C'est un moment de cassure. Tu peux envisager une entrée.\n"
    else:
        if not aligne:
            texte += f"• Aucun signal. ⏳\n"
            texte += f"• Raison : H1 et M30 ne sont pas alignés. Le bot attend une tendance claire.\n"
            texte += f"• Conseil : Patiente, ne trade pas dans le bruit.\n"
        else:
            texte += f"• Aucun signal. ⏳\n"
            texte += f"• Raison : Tendance alignée ({tend_h1}), mais aucune cassure confirmée sur M15.\n"
            texte += f"• Conseil : Attends que le prix clôture {'au-dessus de ' + str(round(res, 2)) if tend_h1 == 'HAUSSE' else 'en-dessous de ' + str(round(sup, 2))} pour un signal.\n"
            
    texte += f"\n💡 **CONSEIL DU BOT**\n"
    if sig:
        texte += f"• Place ton Stop Loss {'sous le support' if sig == 'ACHAT' else 'au-dessus de la résistance'}.\n"
        texte += f"• Vise un Take Profit de 2x ton risque.\n"
    else:
        texte += f"• Reste attentif, une cassure peut arriver à tout moment.\n"

    # --- ENVOI ---
    topic = CONFIG["topics"]["principal"]
    if CONFIG["mode_affichage"] == "RAPPORT" or sig:
        for tf, df, tend in [("H1", df_60, tend_h1), ("M30", df_30, tend_m30), ("M15", df_15, tend_h1)]:
            img = figure(df, tend, sig if tf == "M15" else None, tf, sup, res)
            post_ntfy(topic, f"R_75 - {tf}", img, f"R_75 {tf}")
            time.sleep(1.5)
        
        post_ntfy(topic, texte, None, "Rapport S/R R_75")

if __name__ == "__main__": main()
