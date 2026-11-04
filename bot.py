#!/usr/bin/env python3
"""
Bot Order Block (H1) + tendance SMA (H1/M30) + entrée M15 sur retest.
Pensé pour le trading MANUEL : lot fixe, risque = distance du SL en $.

Usage :
  python bot.py backtest   -> rejoue l'historique, envoie rapport + images
  python bot.py live       -> signaux détaillés + suivi (TP1 / clôture)
"""
import io, os, sys, json, time, asyncio
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import websockets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = json.load(open("config.json"))
S, A, H, LV = CFG["strategie"], CFG["compte"], CFG["historique"], CFG["live"]
TZ = timezone(timedelta(hours=CFG.get("fuseau_heures", 1)))
STATE_FILE = "state.json"
UA = {"User-Agent": "Mozilla/5.0"}
JOURS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
TEND = {1: "HAUSSE", -1: "BAISSE", 0: "NEUTRE"}


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def local(ts):
    t = ts.tz_convert(TZ)
    return f"{JOURS[t.weekday()]} {t.strftime('%d/%m %H:%M')}"


def fp(pair, x):
    if pair.endswith("JPY"):
        d = 3
    elif pair.startswith(("XAU", "XAG", "R_", "CRASH", "BOOM")):
        d = 2
    else:
        d = 5
    return f"{x:.{d}f}"


def contrat(pair):
    return A["taille_contrat"].get(pair, 1)


# ---------------------------------------------------------------- état
def charger_etat():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {}


def sauver_etat(etat):
    json.dump(etat, open(STATE_FILE, "w"), indent=2)


# ---------------------------------------------------------------- ntfy
def post_ntfy(titre, msg, img=None):
    url = CFG.get("ntfy")
    if not url:
        return
    headers = {"Title": titre}  # ASCII uniquement dans les headers
    try:
        if img:
            headers["Filename"] = "chart.png"
            headers["Message"] = titre
            requests.post(url, data=img, headers=headers, timeout=30)
        else:
            requests.post(url, data=msg.encode("utf-8"), headers=headers, timeout=15)
    except Exception as e:
        log(f"Erreur ntfy: {e}")


# ---------------------------------------------------------------- données
def yahoo_symbol(pair):
    return {"XAUUSD": "GC=F", "XAGUSD": "SI=F"}.get(pair, pair + "=X")


def fetch_yahoo(pair):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(pair)}"
    r = requests.get(url, params={"interval": "15m", "range": H["yahoo_range"]},
                     headers=UA, timeout=25)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({
        "time": pd.to_datetime(j["timestamp"], unit="s", utc=True),
        "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"],
    }).dropna()
    return df.iloc[:-1].reset_index(drop=True)  # retire la bougie en cours


async def _deriv(symbol, count):
    async with websockets.connect(CFG["deriv_ws"], open_timeout=30) as ws:
        await ws.send(json.dumps({
            "ticks_history": symbol, "style": "candles",
            "granularity": 900, "count": count, "end": "latest"}))
        while True:
            d = json.loads(await ws.recv())
            if "error" in d:
                raise RuntimeError(d["error"].get("message", "erreur Deriv"))
            if "candles" in d:
                return pd.DataFrame(d["candles"])


def fetch_deriv(symbol):
    df = asyncio.run(_deriv(symbol, H["deriv_bougies"]))
    df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"})
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    for k in ("o", "h", "l", "c"):
        df[k] = df[k].astype(float)
    return df[["time", "o", "h", "l", "c"]].iloc[:-1].reset_index(drop=True)


# ---------------------------------------------------------------- indicateurs
def preparer(df):
    """Ajoute tendance H1/M30 (sans lookahead), ATR et close_time au M15."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True).astype("datetime64[ns, UTC]")
    df["close_time"] = df["time"] + pd.Timedelta(minutes=15)

    pc = df["c"].shift(1)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - pc).abs(), (df["l"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(S["atr_periode"]).mean()

    d = df.set_index("time")
    h1 = None
    for nom, rule in (("h1", "1h"), ("m30", "30min")):
        r = d.resample(rule, label="left", closed="left").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
        f = r["c"].rolling(S["sma_rapide"]).mean()
        s = r["c"].rolling(S["sma_lente"]).mean()
        r["trend"] = np.select([f > s, f < s], [1, -1], 0)
        r["dispo"] = r.index + pd.Timedelta(rule)  # connue seulement à la clôture
        r = r.reset_index()
        r["time"] = r["time"].astype("datetime64[ns, UTC]")
        r["dispo"] = r["dispo"].astype("datetime64[ns, UTC]")
        df = pd.merge_asof(
            df, r[["dispo", "trend"]].rename(columns={"trend": "t_" + nom}),
            left_on="close_time", right_on="dispo", direction="backward"
        ).drop(columns="dispo")
        df["t_" + nom] = df["t_" + nom].fillna(0).astype(int)
        if nom == "h1":
            h1 = r
    return df, h1


def zones_ob(h1):
    """Order Blocks H1 : dernière bougie opposée avant une impulsion qui casse son extrême."""
    o, h, l, c = (h1[k].to_numpy() for k in ("o", "h", "l", "c"))
    dispo = h1["dispo"].tolist()
    corps = np.abs(c - o)
    moy = pd.Series(corps).rolling(20).mean().shift(1).to_numpy()
    zones = []
    for j in range(21, len(h1)):
        if corps[j] < S["impulsion_mult"] * moy[j]:
            continue
        haussier = c[j] > o[j]
        for k in range(j - 1, j - 1 - S["ob_lookback"], -1):
            oppose = (c[k] < o[k]) if haussier else (c[k] > o[k])
            if not oppose:
                continue
            if (haussier and c[j] > h[k]) or ((not haussier) and c[j] < l[k]):
                zones.append({
                    "type": "HAUSSIER" if haussier else "BAISSIER",
                    "bas": float(l[k]), "haut": float(h[k]),
                    "dispo": dispo[j],
                    "expire": dispo[j] + pd.Timedelta(hours=S["ob_age_max_h1"]),
                })
            break
    return sorted(zones, key=lambda z: z["dispo"])


# ---------------------------------------------------------------- simulation
def gerer(t, h, l):
    """Met à jour le trade sur une bougie. SL prioritaire si ambigu (conservateur)."""
    long_ = t["sens"] == "ACHAT"
    sgn = 1 if long_ else -1
    for _ in range(2):
        sl_touche = (l <= t["sl"]) if long_ else (h >= t["sl"])
        if sl_touche:
            t["r"] += t["reste"] * sgn * (t["sl"] - t["entree"]) / t["dist"]
            t["reste"] = 0.0
            return True
        tp1_touche = (h >= t["tp1"]) if long_ else (l <= t["tp1"])
        if t["etape"] == 0 and tp1_touche:
            t["r"] += S["part_tp1"] * S["rr_tp1"]
            t["reste"] = 1 - S["part_tp1"]
            t["sl"] = t["entree"]  # break-even
            t["etape"] = 1
            continue
        break
    tp2_touche = (h >= t["tp2"]) if long_ else (l <= t["tp2"])
    if t["etape"] == 1 and tp2_touche:
        t["r"] += t["reste"] * S["rr_tp2"]
        t["reste"] = 0.0
        return True
    return False


def simuler(df, zones, pair):
    solde = A["solde_depart"]
    trades, ouvert, actives, zi = [], None, [], 0
    H_, L_, C_ = df["h"].to_numpy(), df["l"].to_numpy(), df["c"].to_numpy()
    atr = df["atr"].to_numpy()
    th1, tm30 = df["t_h1"].to_numpy(), df["t_m30"].to_numpy()
    ct = df["close_time"].tolist()
    part, rr1, rr2 = S["part_tp1"], S["rr_tp1"], S["rr_tp2"]

    for i in range(1, len(df)):
        # 1. gestion du trade ouvert
        if ouvert is not None and gerer(ouvert, H_[i], L_[i]):
            pnl = ouvert["r"] * ouvert["risque"]
            solde += pnl
            ouvert.update(statut="FERME", pnl=pnl, solde=solde, sortie=ct[i])
            trades.append(ouvert)
            ouvert = None

        # 2. zones disponibles / valides
        while zi < len(zones) and zones[zi]["dispo"] <= ct[i]:
            actives.append(zones[zi])
            zi += 1
        actives = [z for z in actives
                   if ct[i] <= z["expire"]
                   and not (z["type"] == "HAUSSIER" and C_[i] < z["bas"])
                   and not (z["type"] == "BAISSIER" and C_[i] > z["haut"])]

        # 3. recherche d'entrée
        if ouvert is not None or np.isnan(atr[i]):
            continue
        if th1[i] == 0 or th1[i] != tm30[i]:
            continue
        for z in actives:
            sens = None
            if th1[i] == 1 and z["type"] == "HAUSSIER" and L_[i] <= z["haut"] and C_[i] > z["haut"]:
                sens, entree = "ACHAT", C_[i] + A["spread"]
                sl = min(z["bas"], L_[i]) - S["buffer_atr"] * atr[i]
            elif th1[i] == -1 and z["type"] == "BAISSIER" and H_[i] >= z["bas"] and C_[i] < z["bas"]:
                sens, entree = "VENTE", C_[i] - A["spread"]
                sl = max(z["haut"], H_[i]) + S["buffer_atr"] * atr[i]
            if not sens:
                continue
            dist = abs(entree - sl)
            if dist <= 0:
                continue
            risque = dist * contrat(pair) * A["lot"]  # perte en $ si SL touché
            if A["risque_max_usd"] > 0 and risque > A["risque_max_usd"]:
                continue
            sgn = 1 if sens == "ACHAT" else -1
            ouvert = {
                "date": ct[i], "idx": i, "sens": sens, "entree": entree,
                "sl_init": sl, "sl": sl, "dist": dist,
                "tp1": entree + sgn * rr1 * dist,
                "tp2": entree + sgn * rr2 * dist,
                "risque": risque,
                "gain_tp1": part * rr1 * risque,
                "gain_max": (part * rr1 + (1 - part) * rr2) * risque,
                "atr": float(atr[i]), "tend": TEND[int(th1[i])],
                "ob_type": z["type"], "ob_bas": z["bas"], "ob_haut": z["haut"],
                "ob_date": z["dispo"],
                "r": 0.0, "reste": 1.0, "etape": 0, "statut": "EN COURS",
            }
            actives.remove(z)
            break
    return trades, ouvert, solde


# ---------------------------------------------------------------- statistiques
def statistiques(trades, solde_final):
    depart = A["solde_depart"]
    n = len(trades)
    gains = [t["pnl"] for t in trades if t["pnl"] > 1e-9]
    pertes = [t["pnl"] for t in trades if t["pnl"] < -1e-9]
    be = n - len(gains) - len(pertes)
    equity = [depart] + [t["solde"] for t in trades]
    pic, dd_usd, dd_pct = equity[0], 0.0, 0.0
    for e in equity:
        pic = max(pic, e)
        if pic - e > dd_usd:
            dd_usd = pic - e
            dd_pct = dd_usd / pic * 100 if pic > 0 else 0
    # pire série de pertes consécutives
    cur_n = cur_usd = pire_n = 0
    pire_usd = 0.0
    for t in trades:
        if t["pnl"] < -1e-9:
            cur_n += 1
            cur_usd += t["pnl"]
            if cur_n > pire_n:
                pire_n, pire_usd = cur_n, cur_usd
        else:
            cur_n, cur_usd = 0, 0.0
    sg, sp = sum(gains), -sum(pertes)
    return {
        "n": n, "gagnants": len(gains), "perdants": len(pertes), "be": be,
        "winrate": len(gains) / n * 100 if n else 0,
        "pf": sg / sp if sp > 0 else (float("inf") if sg > 0 else 0),
        "gain_moyen": sg / len(gains) if gains else 0,
        "perte_moyenne": sum(pertes) / len(pertes) if pertes else 0,
        "esperance_usd": sum(t["pnl"] for t in trades) / n if n else 0,
        "esperance_r": sum(t["r"] for t in trades) / n if n else 0,
        "dd_usd": dd_usd, "dd_pct": dd_pct,
        "pire_n": pire_n, "pire_usd": pire_usd,
        "risque_moy": sum(t["risque"] for t in trades) / n if n else 0,
        "risque_max": max((t["risque"] for t in trades), default=0),
        "solde": solde_final, "pnl": solde_final - depart,
        "pnl_pct": (solde_final - depart) / depart * 100,
        "serie": "".join("G" if t["pnl"] > 1e-9 else ("P" if t["pnl"] < -1e-9 else "B")
                         for t in trades[-10:]) or "-",
    }


def detail_trade(pair, t, prix=None):
    p = lambda v: fp(pair, v)
    part = S["part_tp1"]
    if part > 0:
        tp1_txt = f"+{t['gain_tp1']:.2f}$ (clôture {part*100:.0f}%) puis SL à l'entrée"
    else:
        tp1_txt = "SL à l'entrée (break-even)"
    x = f"📌 {t['sens']} {pair} — lot {A['lot']}\n"
    x += f"🕐 Bougie M15 : {local(t['date'])} (UTC{CFG.get('fuseau_heures', 1):+d})\n"
    x += "━━━━━━━━━━━━━━━━━━━━━\n"
    x += "📋 À PLACER\n"
    x += f"• Entrée : {p(t['entree'])}\n"
    x += f"• Stop Loss : {p(t['sl_init'])} (distance {p(t['dist'])})\n"
    x += f"• TP1 : {p(t['tp1'])} → {tp1_txt}\n"
    x += f"• TP2 : {p(t['tp2'])}\n"
    x += "━━━━━━━━━━━━━━━━━━━━━\n"
    x += f"💵 RISQUE / GAIN (lot {A['lot']})\n"
    x += f"• Perte si SL : -{t['risque']:.2f}$\n"
    x += f"• Gain si TP2 : +{t['gain_max']:.2f}$\n"
    x += f"• Ratio R:R : 1:{t['gain_max'] / t['risque']:.1f}\n"
    x += "━━━━━━━━━━━━━━━━━━━━━\n"
    x += "🧭 CONTEXTE\n"
    x += f"• Tendance H1 + M30 : {t['tend']}\n"
    x += (f"• Order Block H1 {t['ob_type']} : {p(t['ob_bas'])} - {p(t['ob_haut'])} "
          f"(formé {local(t['ob_date'])})\n")
    x += f"• ATR M15 : {p(t['atr'])}\n"
    if prix is not None:
        x += f"• Prix actuel : {p(prix)}\n"
    x += "━━━━━━━━━━━━━━━━━━━━━\n"
    x += "🛠 GESTION\n"
    if part > 0:
        x += f"• TP1 touché : clôture {part*100:.0f}% du lot + SL à l'entrée\n"
    else:
        x += "• TP1 touché : remonte le SL à l'entrée, garde tout jusqu'au TP2\n"
    x += "• TP2 touché : clôture le reste\n• SL touché : perte limitée au SL\n"
    return x


def ligne_trade(pair, t):
    ico = "✅" if t["pnl"] > 1e-9 else ("❌" if t["pnl"] < -1e-9 else "⚪")
    return (f"{ico} {local(t['date'])} {t['sens']} {fp(pair, t['entree'])} "
            f"SL {fp(pair, t['sl_init'])} → {t['pnl']:+.2f}$")


def rapport(pair, st, trades, ouvert, nb, debut, fin):
    t = f"📊 BACKTEST ORDER BLOCK {pair}\n"
    t += f"Période : {debut:%d/%m/%Y} → {fin:%d/%m/%Y} ({nb} bougies M15)\n"
    t += f"⚙️ Lot fixe {A['lot']} — risque = distance du SL\n"
    t += "━━━━━━━━━━━━━━━━━━━━━\n"
    t += "💰 COMPTE VIRTUEL\n"
    t += f"• Départ : {A['solde_depart']:.2f}$ → Actuel : {st['solde']:.2f}$\n"
    t += f"• Gain/Perte : {st['pnl']:+.2f}$ ({st['pnl_pct']:+.1f}%)\n"
    t += f"• Drawdown max : {st['dd_usd']:.2f}$ ({st['dd_pct']:.1f}%)\n"
    t += "━━━━━━━━━━━━━━━━━━━━━\n"
    t += "📈 PERFORMANCE\n"
    t += (f"• Trades : {st['n']} (G {st['gagnants']} / P {st['perdants']} / BE {st['be']})\n")
    t += f"• Taux de réussite : {st['winrate']:.0f}%\n"
    pf = "∞" if st["pf"] == float("inf") else f"{st['pf']:.2f}"
    t += f"• Facteur de profit : {pf}\n"
    t += f"• Gain moyen : {st['gain_moyen']:+.2f}$ | Perte moyenne : {st['perte_moyenne']:+.2f}$\n"
    t += f"• Espérance : {st['esperance_usd']:+.2f}$ ({st['esperance_r']:+.2f}R) par trade\n"
    t += f"• Risque moyen/trade : {st['risque_moy']:.2f}$ (max {st['risque_max']:.2f}$)\n"
    t += f"• Pire série de pertes : {st['pire_n']} trades ({st['pire_usd']:+.2f}$)\n"
    t += f"• 10 derniers : {st['serie']}\n"
    if st["n"] < 30:
        t += "⚠️ Moins de 30 trades : échantillon trop petit pour conclure.\n"
    if trades:
        t += "━━━━━━━━━━━━━━━━━━━━━\n🗂 DERNIERS TRADES\n"
        for tr in trades[-10:]:
            t += ligne_trade(pair, tr) + "\n"
    if ouvert:
        t += "━━━━━━━━━━━━━━━━━━━━━\n🔹 TRADE EN COURS\n" + detail_trade(pair, ouvert)
    else:
        t += "━━━━━━━━━━━━━━━━━━━━━\n🔹 Aucun trade en cours.\n"
    return t


# ---------------------------------------------------------------- graphiques
def ohlc(df, rule):
    r = df.set_index("time").resample(rule, label="left", closed="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return r.reset_index()


def courbe(trades, pair):
    equity = [A["solde_depart"]] + [t["solde"] for t in trades]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.plot(equity, color="#26a69a", lw=1.5)
    ax.axhline(A["solde_depart"], color="white", ls="--", lw=0.8)
    ax.set_title(f"{pair} - Courbe de capital ($)", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", ls="--", alpha=0.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def figure(pair, d, titre, zone=None, trade=None, n=50):
    from matplotlib.patches import Rectangle
    p = lambda v: fp(pair, v)
    d = d.copy()
    d["sf"] = d["c"].rolling(S["sma_rapide"]).mean()
    d["ss"] = d["c"].rolling(S["sma_lente"]).mean()
    d = d.tail(n).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    for i, r in d.iterrows():
        col = "#26a69a" if r["c"] >= r["o"] else "#ef5350"
        ax.plot([i, i], [r["l"], r["h"]], color=col, lw=1.2)
        ax.add_patch(Rectangle((i - 0.3, min(r["o"], r["c"])), 0.6,
                               abs(r["o"] - r["c"]), facecolor=col, edgecolor=col))
    ax.plot(d.index, d["sf"], color="#ffa726", lw=1, label=f"SMA {S['sma_rapide']}")
    ax.plot(d.index, d["ss"], color="#42a5f5", lw=1, label=f"SMA {S['sma_lente']}")
    if zone:
        col = "lime" if zone["type"] == "HAUSSIER" else "red"
        ax.axhspan(zone["bas"], zone["haut"], color=col, alpha=0.2,
                   label=f"OB {zone['type'].lower()} {p(zone['bas'])}-{p(zone['haut'])}")
        ax.axhline(zone["bas"], color=col, ls="--", lw=1.2)
        ax.axhline(zone["haut"], color=col, ls="--", lw=1.2)
    if trade:
        ax.axhspan(trade["entree"], trade["tp2"], color="green", alpha=0.12)
        ax.axhspan(trade["sl_init"], trade["entree"], color="red", alpha=0.15)
        ax.axhline(trade["entree"], color="white", lw=1.5, label=f"Entrée {p(trade['entree'])}")
        ax.axhline(trade["sl_init"], color="red", ls="--", lw=1.5,
                   label=f"SL {p(trade['sl_init'])} (-{trade['risque']:.2f}$)")
        ax.axhline(trade["tp1"], color="lime", ls=":", lw=1.5, label=f"TP1 {p(trade['tp1'])}")
        ax.axhline(trade["tp2"], color="green", ls=":", lw=1.5,
                   label=f"TP2 {p(trade['tp2'])} (+{trade['gain_max']:.2f}$)")
    ax.set_title(titre, color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", ls="--", alpha=0.5)
    ax.legend(loc="upper left", facecolor="#1e1e1e", labelcolor="white", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def envoyer_graphiques(pair, df, zones, ouvert):
    fin = df["close_time"].iloc[-1]
    valides = [z for z in zones if z["dispo"] <= fin]
    zone = valides[-1] if valides else None
    th1, tm30 = TEND[int(df["t_h1"].iloc[-1])], TEND[int(df["t_m30"].iloc[-1])]
    vues = [
        ("H1", ohlc(df, "1h"), f"{pair} H1 | {th1}", None, 50),
        ("M30", ohlc(df, "30min"), f"{pair} M30 | {tm30}", None, 50),
        ("M15 plan", df, f"{pair} M15 - plan de trade", ouvert, 50),
        ("M15 contexte", df, f"{pair} M15 - contexte (150 bougies)", ouvert, 150),
    ]
    for nom, d, titre, trade, n in vues:
        img = figure(pair, d, titre, zone, trade, n)
        post_ntfy(f"{pair} {nom}", "", img)
        time.sleep(1.5)


# ---------------------------------------------------------------- suivi live
def maj_suivi(pair, trades, ouvert, suivi):
    """Notifie TP1 atteint / clôture pour les trades déjà signalés."""
    tous = {str(t["date"]): t for t in trades + ([ouvert] if ouvert else [])}
    for cle, s in suivi.items():
        if s["statut"] == "FERME":
            continue
        t = tous.get(cle)
        if not t:
            continue
        p = lambda v: fp(pair, v)
        if t["etape"] >= 1 and s["etape"] == 0:
            s["etape"] = 1
            msg = (f"🔔 TP1 ATTEINT — {pair} {t['sens']}\n"
                   f"→ Remonte le SL à l'entrée : {p(t['entree'])}\n")
            if S["part_tp1"] > 0:
                msg += f"→ Clôture {S['part_tp1']*100:.0f}% du lot (+{t['gain_tp1']:.2f}$)\n"
            msg += f"Objectif restant : TP2 {p(t['tp2'])} (+{t['gain_max']:.2f}$)"
            post_ntfy(f"TP1 {pair}", msg)
        if t["statut"] == "FERME":
            s["statut"] = "FERME"
            ico = "✅" if t["pnl"] > 1e-9 else ("❌" if t["pnl"] < -1e-9 else "⚪")
            if t["pnl"] > 1e-9:
                raison = "TP2 atteint" if t["etape"] >= 1 else "Gain"
            elif t["pnl"] < -1e-9:
                raison = "Stop Loss touché"
            else:
                raison = "Break-even (SL à l'entrée touché)"
            msg = (f"{ico} TRADE CLÔTURÉ — {pair} {t['sens']}\n{raison}\n"
                   f"Résultat : {t['pnl']:+.2f}$ ({t['r']:+.2f}R)\n"
                   f"Entrée {p(t['entree'])} | SL initial {p(t['sl_init'])}\n"
                   f"Sortie : {local(t['sortie'])}")
            post_ntfy(f"Cloture {pair}", msg)


# ---------------------------------------------------------------- orchestration
def analyser(pair, source, mode, etat):
    try:
        df = fetch_yahoo(pair) if source == "yahoo" else fetch_deriv(pair)
        if df is None or len(df) < 200:
            log(f"{pair}: pas assez de données")
            return
        df, h1 = preparer(df)
        zones = zones_ob(h1)
        trades, ouvert, solde = simuler(df, zones, pair)
        log(f"{pair}: {len(df)} bougies, {len(zones)} OB, {len(trades)} trades clôturés")
        st = statistiques(trades, solde)
        texte = rapport(pair, st, trades, ouvert, len(df), df["time"].iloc[0], df["time"].iloc[-1])

        if mode == "backtest":
            print(texte)
            envoyer_graphiques(pair, df, zones, ouvert)
            if trades:
                pd.DataFrame(trades).to_csv(f"backtest_{pair}.csv", index=False)
                post_ntfy(f"{pair} equity", "", courbe(trades, pair))
            post_ntfy(f"Backtest OB {pair}", texte)
            return

        # ---- live
        if not isinstance(etat.get(pair), dict):
            etat[pair] = {}
        suivi = etat[pair]
        maj_suivi(pair, trades, ouvert, suivi)

        recent = len(df) - 1 - LV["signal_age_max_bougies"]
        if ouvert and ouvert["idx"] >= recent and str(ouvert["date"]) not in suivi:
            suivi[str(ouvert["date"])] = {"etape": ouvert["etape"], "statut": "EN COURS"}
            prix = float(df["c"].iloc[-1])
            envoyer_graphiques(pair, df, zones, ouvert)
            post_ntfy(f"SIGNAL {pair} {ouvert['sens']}",
                      f"🎯 SIGNAL {ouvert['sens']} {pair}\n\n" + detail_trade(pair, ouvert, prix))
            post_ntfy(f"Rapport OB {pair}", texte)
            log(f"{pair}: signal envoyé")
        else:
            log(f"{pair}: pas de nouveau signal")
    except Exception as e:
        log(f"Erreur {pair}: {e}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if mode not in ("backtest", "live"):
        print("Usage: python bot.py [backtest|live]")
        return
    log(f"DÉMARRAGE ({mode})")
    etat = charger_etat()
    fx = CFG["marches"]["forex"]
    if fx["actif"]:
        for p, on in fx["paires"].items():
            if on:
                analyser(p, "yahoo", mode, etat)
    dv = CFG["marches"]["deriv"]
    if dv["actif"]:
        for p, on in dv["indices"].items():
            if on:
                analyser(p, "deriv", mode, etat)
    sauver_etat(etat)


if __name__ == "__main__":
    main()
