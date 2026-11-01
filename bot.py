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

# --- DÉTECTION DE ZONES (SUR 20 BOUGIES H1) ---
def trouver_pivots(df, window=3):
    pivots_hauts = []
    pivots_bas = []
    for i in range(window, len(df) - window):
        if df["h"].iloc[i] == df["h"].iloc[i-window:i+window+1].max():
            pivots_hauts.append(df["h"].iloc[i])
        if df["l"].iloc[i] == df["l"].iloc[i-window:i+window+1].min():
            pivots_bas.append(df["l"].iloc[i])
    return pivots_hauts, pivots_bas

def get_zones(df_h1):
    # On analyse UNIQUEMENT les 20 dernières bougies pour trouver les zones
    df_h1 = df_h1.tail(20).reset_index(drop=True)
    
    hauts, bas = trouver_pivots(df_h1, window=2)
    if not hauts or not bas:
        return None, None
    
    def regrouper(pivots, tolerance=0.005):
        if not pivots: return []
        pivots = sorted(pivots)
        zones = []
        current_zone = [pivots[0]]
        for p in pivots[1:]:
            if (p - current_zone[-1]) / current_zone[-1] < tolerance:
                current_zone.append(p)
            else:
                zones.append((min(current_zone), max(current_zone)))
                current_zone = [p]
        zones.append((min(current_zone), max(current_zone)))
        return zones

    zones_res = regrouper(hauts)
    zones_sup = regrouper(bas)
    
    prix = df_h1["c"].iloc[-1]
    
    res_proches = [z for z in zones_res if z[0] > prix]
    res = min(res_proches, key=lambda z: z[0]) if res_proches else None
    
    sup_proches = [z for z in zones_sup if z[1] < prix]
    sup = max(sup_proches, key=lambda z: z[1]) if sup_proches else None
    
    return res, sup

def check_cassure_zone(df, zone_sup, zone_res, sens):
    if sens == "HAUSSE" and zone_res:
        if df["c"].iloc[-2] <= zone_res[1] and df["c"].iloc[-1] > zone_res[1]:
            return "ACHAT"
    if sens == "BAISSE" and zone_sup:
        if df["c"].iloc[-2] >= zone_sup[0] and df["c"].iloc[-1] < zone_sup[0]:
            return "VENTE"
    return None

def figure(df, tend, sig, tf_label, zone_sup, zone_res):
    df = df.tail(50).reset_index(drop=True) # On AFFICHE 50 bougies
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('#1e1e1e'); ax.set_facecolor('#1e1e1e')
    for i, r in df.iterrows():
        col = '#26a69a' if r['c'] >= r['o'] else '#ef5350'
        ax.plot([i, i], [r['l'], r['h']], color=col, lw=1.2)
        ax.add_patch(Rectangle((i-0.3, min(r['o'], r['c'])), 0.6, abs(r['o']-r['c']), facecolor=col, edgecolor=col))
    
    if zone_res:
        ax.axhspan(zone_res[0], zone_res[1], color='red', alpha=0.2, label=f"Résistance ({zone_res[0]:.1f}-{zone_res[1]:.1f})")
    if zone_sup:
        ax.axhspan(zone_sup[0], zone_sup[1], color='lime', alpha=0.2, label=f"Support ({zone_sup[0]:.1f}-{zone_sup[1]:.1f})")
        
    ax.set_title(f"R_75 - {tf_label} | {tend} | {sig or ''}", color="white")
    ax.tick_params(colors='white'); ax.grid(color='#333333', ls='--', alpha=0.5)
    ax.legend(loc="upper left", facecolor='#1e1e1e', labelcolor='white')
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return buf.getvalue()

def main():
    now = datetime.now(BENIN)
    log("DÉMARRAGE S/R (ZONES SUR 20 BOUGIES)")
    
    df_15 = fetch_deriv("R_75", "15m")
    df_30 = fetch_deriv("R_75", "30m")
    df_60 = fetch_deriv("R_75", "60m")
    
    if df_15 is None or df_30 is None or df_60 is None: return
    
    tend_h1 = tendance(df_60)
    tend_m30 = tendance(df_30)
    aligne = (tend_h1 == tend_m30 and tend_h1 in ("HAUSSE", "BAISSE"))
    
    zone_res, zone_sup = get_zones(df_60)
    
    sig = None
    if aligne:
        sig = check_cassure_zone(df_15, zone_sup, zone_res, tend_h1)
    
    log(f"R_75 analysé. H1:{tend_h1} M30:{tend_m30} Signal: {sig}")
    if zone_res: log(f"Zone Résistance: {zone_res[0]:.2f} - {zone_res[1]:.2f}")
    if zone_sup: log(f"Zone Support: {zone_sup[0]:.2f} - {zone_sup[1]:.2f}")
    
    prix = df_15["c"].iloc[-1]
    
    texte = f"📊 **RAPPORT ZONES S/R R_75** - {now.strftime('%a %d/%m %H:%M')}\n\n"
    texte += f"🔹 **TENDANCE DE FOND**\n"
    texte += f"• H1 : {tend_h1}\n• M30 : {tend_m30}\n"
    texte += f"• Alignement : {'✅ OUI' if aligne else '❌ NON'}\n\n"
    
    texte += f"📏 **ZONES DÉTECTÉES (Sur 20 bougies H1)**\n"
    texte += f"• Résistance : {f'{zone_res[0]:.2f} - {zone_res[1]:.2f}' if zone_res else 'Aucune'}\n"
    texte += f"• Support : {f'{zone_sup[0]:.2f} - {zone_sup[1]:.2f}' if zone_sup else 'Aucun'}\n"
    texte += f"• Prix actuel : {prix:.2f}\n\n"
    
    texte += f"🎯 **SIGNAL DE CASSURE**\n"
    if sig:
        texte += f"• {sig} CONFIRMÉ ! 🚀\n"
        texte += f"• La bougie M15 a clôturé {'au-dessus de la zone de résistance' if sig == 'ACHAT' else 'en-dessous de la zone de support'}.\n"
        texte += f"• Conseil : Cassure validée. Envisage une entrée avec Stop Loss de l'autre côté de la zone.\n"
    else:
        if not aligne:
            texte += f"• Aucun signal. ⏳\n• Raison : H1 et M30 ne sont pas alignés.\n"
        else:
            texte += f"• Aucun signal. ⏳\n• Raison : Tendance alignée ({tend_h1}), mais aucune cassure de zone confirmée sur M15.\n"
            if tend_h1 == "HAUSSE" and zone_res:
                texte += f"• Conseil : Attends que le prix clôture au-dessus de {zone_res[1]:.2f}.\n"
            elif tend_h1 == "BAISSE" and zone_sup:
                texte += f"• Conseil : Attends que le prix clôture en-dessous de {zone_sup[0]:.2f}.\n"

    topic = CONFIG["topics"]["principal"]
    if CONFIG["mode_affichage"] == "RAPPORT" or sig:
        for tf, df, tend in [("H1", df_60, tend_h1), ("M30", df_30, tend_m30), ("M15", df_15, tend_h1)]:
            img = figure(df, tend, sig if tf == "M15" else None, tf, zone_sup, zone_res)
            post_ntfy(topic, f"R_75 - {tf}", img, f"R_75 {tf}")
            time.sleep(1.5)
        
        post_ntfy(topic, texte, None, "Rapport Zones S/R R_75")

if __name__ == "__main__": main()
