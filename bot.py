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
SMA_FAST, SMA_SLOW = 3, 25  # conservés mais plus utilisés dans tendance()

FOREX_SYMBOLS = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"}
DERIV_SYMBOLS = {"R_75": "R_75"}

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def post_ntfy(url, msg, img=None, title=None):
    if not url: return
    headers = {"Title": title or "V75 S/R"}
    if img:
        headers["Filename"] = "chart.png"
        headers["Content-Type"] = "image/png"
        headers["Message"] = msg
        requests.post(url, data=img, headers=headers, timeout=30)
    else:
        requests.post(url, data=msg.encode("utf-8"), headers=headers, timeout=15)

def fetch_yahoo(symbol, interval):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range=5d"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    if r.status_code != 200: return None
    j = r.json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({"t": pd.to_datetime(j["timestamp"], unit="s", utc=True),
                       "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"]}).dropna()
    return df.iloc[:-1].reset_index(drop=True) if len(df) > 30 else None

async def _fetch_deriv(symbol, granularity, count=100):
    uri = 'wss://api.derivws.com/trading/v1/options/ws/public'
    async with websockets.connect(uri, open_timeout=30) as ws:
        await ws.send(json.dumps({'ticks_history': symbol, 'style': 'candles',
                                  'granularity': granularity, 'count': count, 'end': 'latest'}))
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

# ---------- ÉTAPE 1 : S/R + TENDANCE VIA S/R ----------
def atr(df, period=14):
    h, l, c = df["h"], df["l"], df["c"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return v if not np.isnan(v) else 0.0

def detecter_sr(df, lookback=3, cluster_atr=0.3, min_touches=2):
    if len(df) < lookback * 2 + 5:
        return []
    a = atr(df)
    if a == 0:
        return []
    tol = cluster_atr * a
    pivots = []
    for i in range(lookback, len(df) - lookback):
        h, l = df["h"], df["l"]
        if h.iloc[i] == h.iloc[i-lookback:i+lookback+1].max():
            pivots.append(("RES", h.iloc[i]))
        if l.iloc[i] == l.iloc[i-lookback:i+lookback+1].min():
            pivots.append(("SUP", l.iloc[i]))
    if not pivots:
        return []
    pivots.sort(key=lambda x: x[1])
    clusters = []
    for typ, prix in pivots:
        if clusters and abs(prix - clusters[-1]["prix"]) <= tol:
            c = clusters[-1]
            c["prix"] = (c["prix"] * c["n"] + prix) / (c["n"] + 1)
            c["n"] += 1
        else:
            clusters.append({"type": typ, "prix": prix, "n": 1})
    return [{"type": c["type"], "prix": c["prix"],
             "bas": c["prix"] - tol, "haut": c["prix"] + tol, "n": c["n"]}
            for c in clusters if c["n"] >= min_touches]

def tendance(df):
    zones = detecter_sr(df)
    if not zones:
        return "NEUTRE"
    prix = df["c"].iloc[-1]
    supports = [z for z in zones if z["type"] == "SUP" and z["prix"] < prix]
    resistances = [z for z in zones if z["type"] == "RES" and z["prix"] > prix]
    if supports and not resistances:
        return "HAUSSE"
    if resistances and not supports:
        return "BAISSE"
    if supports and resistances:
        d_sup = prix - max(s["prix"] for s in supports)
        d_res = min(r["prix"] for r in resistances) - prix
        return "HAUSSE" if d_sup < d_res else "BAISSE"
    return "NEUTRE"

# ---------- ORDER BLOCK (inchangé) ----------
def detecter_order_block(df):
    for i in range(len(df) - 2, 0, -1):
        bougie = df.iloc[i]
        suivante = df.iloc[i+1] if i+1 < len(df) else None
        if suivante is None: continue
        if bougie["c"] > bougie["o"] and suivante["c"] < suivante["o"] and (suivante["o"] - suivante["c"]) > (bougie["c"] - bougie["o"]):
            return "BAISSIER", bougie["o"], bougie["c"]
        if bougie["c"] < bougie["o"] and suivante["c"] > suivante["o"] and (suivante["c"] - suivante["o"]) > (bougie["o"] - bougie["c"]):
            return "HAUSSIER", bougie["c"], bougie["o"]
    return None, None, None

def check_cassure_ob(df, ob_type, ob_bas, ob_haut, sens):
    if sens == "HAUSSE" and ob_type == "HAUSSIER":
        if df["c"].iloc[-2] <= ob_haut and df["c"].iloc[-1] > ob_haut: return "ACHAT"
    if sens == "BAISSE" and ob_type == "BAISSIER":
        if df["c"].iloc[-2] >= ob_bas and df["c"].iloc[-1] < ob_bas: return "VENTE"
    return None

def figure(df, tend, sig, tf_label, pair_name, ob_type, ob_bas, ob_haut):
    df = df.tail(50).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('#1e1e1e'); ax.set_facecolor('#1e1e1e')

    for i, r in df.iterrows():
        col = '#26a69a' if r['c'] >= r['o'] else '#ef5350'
        ax.plot([i, i], [r['l'], r['h']], color=col, lw=1.2)
        ax.add_patch(Rectangle((i-0.3, min(r['o'], r['c'])), 0.6, abs(r['o']-r['c']), facecolor=col, edgecolor=col))

    x = np.arange(len(df))
    z = np.polyfit(x, df["c"], 1)
    p = np.poly1d(z)
    tendance_line = p(x)
    ecart_haut = (df["h"] - tendance_line).max()
    ecart_bas = (df["l"] - tendance_line).min()
    ax.plot(x, tendance_line + ecart_haut, color="red", alpha=0.6, lw=1.5, ls="-.", label="Canal Haut")
    ax.plot(x, tendance_line + ecart_bas, color="lime", alpha=0.6, lw=1.5, ls="-.", label="Canal Bas")
    ax.plot(x, tendance_line, color="gray", alpha=0.5, lw=1, ls=":", label="Ligne de Tendance")

    if ob_type == "BAISSIER":
        ax.axhspan(ob_bas, ob_haut, color='red', alpha=0.2, label="Order Block Baissier")
        ax.axhline(ob_bas, color="red", ls="--", lw=1.5)
        ax.axhline(ob_haut, color="red", ls="--", lw=1.5)
    elif ob_type == "HAUSSIER":
        ax.axhspan(ob_bas, ob_haut, color='lime', alpha=0.2, label="Order Block Haussier")
        ax.axhline(ob_bas, color="lime", ls="--", lw=1.5)
        ax.axhline(ob_haut, color="lime", ls="--", lw=1.5)

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

        ob_type, ob_bas, ob_haut = detecter_order_block(df_60)

        sig = None
        if aligne and ob_type:
            sig = check_cassure_ob(df_15, ob_type, ob_bas, ob_haut, tend_h1)

        log(f"{pair_key} analysé. H1:{tend_h1} M30:{tend_m30} OB:{ob_type} Signal: {sig}")

        mode_notif = CONFIG.get("mode_notification", "EVENEMENT")
        mode_affichage = CONFIG.get("mode_affichage", "RAPPORT")

        doit_envoyer = False
        if mode_affichage == "RAPPORT":
            doit_envoyer = True
        elif mode_affichage == "SIGNAL" and sig:
            doit_envoyer = True
        if mode_notif == "EVENEMENT" and not sig:
            doit_envoyer = False

        if doit_envoyer:
            texte = f"📊 **RAPPORT S/R {pair_key}** - {now.strftime('%a %d/%m %H:%M')}\n\n"
            texte += f"🔹 **TENDANCE**\n• H1 : {tend_h1}\n• M30 : {tend_m30}\n• Alignement : {'✅ OUI' if aligne else '❌ NON'}\n\n"
            if ob_type:
                texte += f"📏 **ORDER BLOCK**\n• Type : {ob_type}\n• Zone : {ob_bas:.5f} - {ob_haut:.5f}\n\n"
            texte += f"🎯 **SIGNAL**\n"
            if sig: texte += f"• {sig} CONFIRMÉ ! ✅ (Cassure de l'OB)\n"
            else:
                if not aligne: texte += f"• Aucun signal (H1/M30 non alignés).\n"
                else: texte += f"• Aucun signal (pas de cassure).\n"

            topic = CONFIG["topics"]["principal"]
            for tf, df, tend in [("H1", df_60, tend_h1), ("M30", df_30, tend_m30), ("M15", df_15, tend_h1)]:
                img = figure(df, tend, sig if tf == "M15" else None, tf, pair_key, ob_type, ob_bas, ob_haut)
                post_ntfy(topic, f"{pair_key} - {tf}", img, f"{pair_key} {tf}")
                time.sleep(1.5)
            post_ntfy(topic, texte, None, f"Rapport S/R {pair_key}")
            log(f"Envoyé à Admin pour {pair_key}")
    except Exception as e: log(f"Erreur {pair_key}: {e}")

def main():
    now = datetime.now(BENIN)
    log("DÉMARRAGE — Étape 1 : tendance via S/R")
    if CONFIG["marches"]["forex"]["actif"]:
        for k in CONFIG["marches"]["forex"]["paires"]:
            if CONFIG["marches"]["forex"]["paires"][k]: analyser(k, "yahoo", now)
    if CONFIG["marches"]["deriv"]["actif"]:
        for k in CONFIG["marches"]["deriv"]["indices"]:
            if CONFIG["marches"]["deriv"]["indices"][k]: analyser(k, "deriv", now)

if __name__ == "__main__": main()
