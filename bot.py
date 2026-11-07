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

# Paramètres S/R et OB depuis config (avec défauts)
SR_CFG = CONFIG.get("support_resistance", {})
OB_CFG = CONFIG.get("order_blocks", {})
PIVOT_LOOKBACK = SR_CFG.get("pivot_lookback", 3)
CLUSTER_ATR = SR_CFG.get("cluster_atr_mult", 0.3)
MIN_TOUCHES = SR_CFG.get("min_touches", 2)
OB_LOOKBACK = OB_CFG.get("swing_lookback", 3)
OB_ATR_MULT = OB_CFG.get("impulse_atr_mult", 1.5)

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

def tendance(df):
    df["sma_f"] = df["c"].rolling(SMA_FAST).mean()
    df["sma_s"] = df["c"].rolling(SMA_SLOW).mean()
    last = df.iloc[-1]
    if last["sma_f"] > last["sma_s"]: return "HAUSSE"
    if last["sma_f"] < last["sma_s"]: return "BAISSE"
    return "NEUTRE"

def atr(df, period=14):
    h, l, c = df["h"], df["l"], df["c"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

# ---------- ORDER BLOCK (close-based, réparé) ----------
def detecter_order_block(df):
    """Cherche l'OB le plus récent : dernière bougie opposée avant impulsion > 1.5×ATR."""
    if len(df) < 20: return None, None, None, None
    a = atr(df)
    if a is None or np.isnan(a): return None, None, None, None
    # On part de la fin, on remonte
    for i in range(len(df) - 2, OB_LOOKBACK, -1):
        bougie = df.iloc[i]
        suivante = df.iloc[i + 1]
        corps_b = abs(bougie["c"] - bougie["o"])
        corps_s = abs(suivante["c"] - suivante["o"])
        # Impulsion = bougie suivante > 1.5 ATR et > corps bougie
        if corps_s < OB_ATR_MULT * a or corps_s <= corps_b:
            continue
        # OB baissier : bougie verte puis bougie rouge impulsive
        if bougie["c"] > bougie["o"] and suivante["c"] < suivante["o"]:
            return "BAISSIER", bougie["o"], bougie["c"], i
        # OB haussier : bougie rouge puis bougie verte impulsive
        if bougie["c"] < bougie["o"] and suivante["c"] > suivante["o"]:
            return "HAUSSIER", bougie["c"], bougie["o"], i
    return None, None, None, None

# ---------- SUPPORT / RÉSISTANCE (M15) ----------
def detecter_sr(df, lookback=None):
    """Détecte pivots puis clusterise en zones. Retourne liste de dicts {type, bas, haut, prix}."""
    lb = lookback or PIVOT_LOOKBACK
    if len(df) < lb * 2 + 5: return []
    a = atr(df)
    if a is None or np.isnan(a): return []
    tol = CLUSTER_ATR * a
    pivots = []
    for i in range(lb, len(df) - lb):
        h, l = df["h"], df["l"]
        if h.iloc[i] == h.iloc[i-lb:i+lb+1].max():
            pivots.append(("RES", h.iloc[i]))
        if l.iloc[i] == l.iloc[i-lb:i+lb+1].min():
            pivots.append(("SUP", l.iloc[i]))
    if not pivots: return []
    # Clustering par proximité
    pivots.sort(key=lambda x: x[1])
    clusters = []
    for typ, prix in pivots:
        if clusters and abs(prix - clusters[-1]["prix"]) <= tol:
            clusters[-1]["prix"] = (clusters[-1]["prix"] * clusters[-1]["n"] + prix) / (clusters[-1]["n"] + 1)
            clusters[-1]["n"] += 1
        else:
            clusters.append({"type": typ, "prix": prix, "n": 1})
    zones = []
    for c in clusters:
        if c["n"] >= MIN_TOUCHES:
            zones.append({"type": c["type"], "prix": c["prix"],
                          "bas": c["prix"] - tol, "haut": c["prix"] + tol, "n": c["n"]})
    return zones

def check_cassure_sr_confirmee(df_m15, zones, tend_h1, tend_m30):
    """Cassure clôturée + confirmation bougie suivante + tendances alignées."""
    if len(df_m15) < 3: return None
    if tend_h1 != tend_m30 or tend_h1 not in ("HAUSSE", "BAISSE"): return None
    b1 = df_m15.iloc[-2]   # bougie de cassure
    b2 = df_m15.iloc[-1]   # bougie de confirmation
    for z in zones:
        # Cassure haussière clôturée
        if tend_h1 == "HAUSSE" and z["type"] == "RES":
            if b1["c"] > z["haut"] and b2["c"] > z["haut"]:
                return {"sens": "ACHAT", "zone": z, "entree": b2["c"]}
        # Cassure baissière clôturée
        if tend_h1 == "BAISSE" and z["type"] == "SUP":
            if b1["c"] < z["bas"] and b2["c"] < z["bas"]:
                return {"sens": "VENTE", "zone": z, "entree": b2["c"]}
    return None

def plan_ordre(sig, zones, df_m15):
    """Calcule Entrée / SL / TP à partir du S/R cassé."""
    if not sig: return None
    a = atr(df_m15)
    z = sig["zone"]
    entree = sig["entree"]
    if sig["sens"] == "ACHAT":
        sl = z["bas"] - 0.5 * a
        # TP = prochaine zone RES au-dessus
        tps = sorted([zz["prix"] for zz in zones if zz["type"] == "RES" and zz["prix"] > entree])
        tp = tps[0] if tps else entree + 2 * (entree - sl)
    else:
        sl = z["haut"] + 0.5 * a
        tps = sorted([zz["prix"] for zz in zones if zz["type"] == "SUP" and zz["prix"] < entree], reverse=True)
        tp = tps[0] if tps else entree - 2 * (sl - entree)
    rr = abs(tp - entree) / abs(entree - sl) if entree != sl else 0
    return {"entree": entree, "sl": sl, "tp": tp, "rr": rr}

# ---------- GRAPHIQUE ----------
def figure(df, tend, sig, tf_label, pair_name, ob_type, ob_bas, ob_haut, zones=None):
    df_plot = df.tail(50).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('#1e1e1e'); ax.set_facecolor('#1e1e1e')

    for i, r in df_plot.iterrows():
        col = '#26a69a' if r['c'] >= r['o'] else '#ef5350'
        ax.plot([i, i], [r['l'], r['h']], color=col, lw=1.2)
        ax.add_patch(Rectangle((i-0.3, min(r['o'], r['c'])), 0.6, abs(r['o']-r['c']),
                               facecolor=col, edgecolor=col))

    # Canal de tendance
    x = np.arange(len(df_plot))
    z = np.polyfit(x, df_plot["c"], 1); p = np.poly1d(z)
    tl = p(x)
    eh = (df_plot["h"] - tl).max(); eb = (df_plot["l"] - tl).min()
    ax.plot(x, tl + eh, color="red", alpha=0.6, lw=1.5, ls="-.", label="Canal Haut")
    ax.plot(x, tl + eb, color="lime", alpha=0.6, lw=1.5, ls="-.", label="Canal Bas")
    ax.plot(x, tl, color="gray", alpha=0.5, lw=1, ls=":", label="Tendance")

    # Zones S/R
    if zones:
        for zz in zones:
            col = '#ff9800' if zz["type"] == "RES" else '#2196f3'
            ax.axhspan(zz["bas"], zz["haut"], color=col, alpha=0.12)
            ax.axhline(zz["prix"], color=col, ls=":", lw=1, alpha=0.7)

    # Order Block (close-based)
    if ob_type == "BAISSIER":
        ax.axhspan(ob_bas, ob_haut, color='red', alpha=0.2, label="OB Baissier")
        ax.axhline(ob_bas, color="red", ls="--", lw=1.5)
        ax.axhline(ob_haut, color="red", ls="--", lw=1.5)
    elif ob_type == "HAUSSIER":
        ax.axhspan(ob_bas, ob_haut, color='lime', alpha=0.2, label="OB Haussier")
        ax.axhline(ob_bas, color="lime", ls="--", lw=1.5)
        ax.axhline(ob_haut, color="lime", ls="--", lw=1.5)

    titre = f"{pair_name} - {tf_label} | {tend}"
    if sig: titre += f" | {sig['sens']}"
    ax.set_title(titre, color="white")
    ax.tick_params(colors='white'); ax.grid(color='#333333', ls='--', alpha=0.5)
    ax.legend(loc="upper left", facecolor='#1e1e1e', labelcolor='white', fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return buf.getvalue()

# ---------- ANALYSE ----------
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

        ob_type, ob_bas, ob_haut, _ = detecter_order_block(df_60)
        zones = detecter_sr(df_15)
        sig = check_cassure_sr_confirmee(df_15, zones, tend_h1, tend_m30) if aligne else None
        plan = plan_ordre(sig, zones, df_15) if sig else None

        log(f"{pair_key} | H1:{tend_h1} M30:{tend_m30} | OB:{ob_type} | S/R:{len(zones)} | Sig:{sig['sens'] if sig else None}")

        mode_notif = CONFIG.get("mode_notification", "RAPPORT")
        mode_affichage = CONFIG.get("mode_affichage", "RAPPORT")

        doit_envoyer = (mode_affichage == "RAPPORT") or (mode_affichage == "SIGNAL" and sig)
        if mode_notif == "EVENEMENT" and not sig:
            doit_envoyer = False

        if doit_envoyer:
            texte = f"📊 **{pair_key}** - {now.strftime('%a %d/%m %H:%M')}\n\n"
            texte += f"🔹 TENDANCE\n• H1 : {tend_h1}\n• M30 : {tend_m30}\n• Aligné : {'✅' if aligne else '❌'}\n\n"
            if ob_type:
                texte += f"📏 ORDER BLOCK (H1)\n• {ob_type}\n• Zone : {ob_bas:.5f} - {ob_haut:.5f}\n\n"
            if zones:
                texte += f"📐 S/R M15 : {len(zones)} zone(s)\n\n"
            if sig and plan:
                texte += f"🎯 PLAN D'ORDRE (MT manuel)\n"
                texte += f"• Sens   : {sig['sens']} LIMITE\n"
                texte += f"• Entrée : {plan['entree']:.5f}\n"
                texte += f"• SL     : {plan['sl']:.5f}\n"
                texte += f"• TP     : {plan['tp']:.5f}\n"
                texte += f"• R/R    : {plan['rr']:.2f}\n"
                texte += f"• Confirmation : cassure M15 clôturée + bougie suivante\n"
            else:
                texte += f"🎯 Aucun signal (cassure confirmée + tendances alignées requises).\n"

            topic = CONFIG["topics"]["principal"]
            for tf, df, tend in [("H1", df_60, tend_h1), ("M30", df_30, tend_m30), ("M15", df_15, tend_h1)]:
                img = figure(df, tend, sig if tf == "M15" else None, tf, pair_key,
                             ob_type, ob_bas, ob_haut, zones)
                post_ntfy(topic, f"{pair_key} - {tf}", img, f"{pair_key} {tf}")
                time.sleep(1.5)
            post_ntfy(topic, texte, None, f"Rapport {pair_key}")
            log(f"Envoyé {pair_key}")
    except Exception as e:
        log(f"Erreur {pair_key}: {e}")

def main():
    now = datetime.now(BENIN)
    log("DÉMARRAGE S/R + OB + CASSURE CONFIRMÉE")
    if CONFIG["marches"]["forex"]["actif"]:
        for k, v in CONFIG["marches"]["forex"]["paires"].items():
            if v: analyser(k, "yahoo", now)
    if CONFIG["marches"]["deriv"]["actif"]:
        for k, v in CONFIG["marches"]["deriv"]["indices"].items():
            if v: analyser(k, "deriv", now)

if __name__ == "__main__": main()
