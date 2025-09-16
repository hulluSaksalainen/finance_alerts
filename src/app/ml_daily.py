""" maschinelles Lernen (ML) für tägliche Kursdaten."""
# src/app/ml_daily.py
from __future__ import annotations

import json
#import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import Dict, List #, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ------------------------------
# Konfiguration / Konstanten
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
IMG_ROOT = BASE_DIR / "src" / "img"
STATE_FILE = BASE_DIR / "json" / "alert_state.json"
logger = logging.getLogger("stock-alerts")
# Fallback-Streamlit-URL (kannst du in main.py überschreiben)
STREAMLIT_BASE_URL = "https://your-streamlit-host/info_zu_ticker"

# ------------------------------
# Dataclasses
# ------------------------------
@dataclass
class ModelResult:
    """Ergebnis eines ML-Durchlaufs für einen Ticker an einem Tag."""
    ticker: str
    date: str
    model_name: str
    accuracy_rf: float
    accuracy_1: float
    report_rf: str
    report_1: str
    link_streamlit: str
    img_dir: Path
    img_pic_rf1: Path
    img_pic_11: Path
    img_pic_rf2: Path
    img_pic_12: Path
    prediction_rf: int = None  # 1=steigt, 0=fällt, None=keine Prognose

# ------------------------------
# Feature Engineering
# ------------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Berechne den Relative Strength Index (RSI) für eine gegebene Periode."""
    delta = close.diff().to_numpy().ravel()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(period).mean()
    roll_down = pd.Series(down, index=close.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeuge technische Indikatoren als Features."""
    df_out = df.copy()
    df_out["return"] = df_out["Close"].pct_change()
    df_out["MA5"] = df_out["Close"].rolling(5).mean()
    df_out["MA20"] = df_out["Close"].rolling(20).mean()
    df_out["STD20"] = df_out["Close"].rolling(20).std()
    df_out["RSI14"] = compute_rsi(df_out["Close"], 14)
    df_out["BB_up"] = df_out["MA20"] + 2 * df_out["STD20"]
    df_out["BB_lo"] = df_out["MA20"] - 2 * df_out["STD20"]

    # Zielvariable: 1, wenn nächster Handelstag höher schließt, sonst 0
    df_out["target"] = (df_out["Close"].shift(-1) > df_out["Close"]).astype(int)

    # Drop NaNs von rollenden Fenstern
    df_out = df_out.dropna().copy()
    return df_out


# ------------------------------
# Daten laden
# ------------------------------
def load_hist_prices(ticker: str, lookback_years: int = 5) -> pd.DataFrame:
    """Lade historische Kursdaten von yfinance."""
    df = yf.download(ticker, period=f"{lookback_years}y", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Keine Kursdaten für {ticker} gefunden.")
    return df


# ------------------------------
# Training + Evaluation
# ------------------------------
def train_and_evaluate(df_feat: pd.DataFrame, model_kind: str = "rf") -> Dict:
    """
    model_kind: 'rf' (RandomForest) oder 'logreg'
    Zeitserien-CV für robustere Schätzung, letzte Fold-Performance reported.
    """
    feature_cols = ["return", "MA5", "MA20", "STD20", "RSI14", "BB_up", "BB_lo"]
    X = df_feat[feature_cols].values
    y = df_feat["target"].values

    if model_kind == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    elif model_kind == "logreg":
        model = LogisticRegression(max_iter=500, n_jobs=None if hasattr(LogisticRegression,
            "n_jobs") else None)
    elif model_kind == "gb":
        model = GradientBoostingClassifier(random_state=42)
    elif model_kind == "svc":
        model = SVC(probability=True, random_state=42)
    else:
        raise ValueError("model_kind muss 'rf','gb', 'svc' oder 'logreg' sein")

    models=[RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )]
    if model_kind not in models:
        models.append(model)
    elif model_kind == "rf":
        models.append(LogisticRegression(max_iter=500, n_jobs=None if hasattr(LogisticRegression,
            "n_jobs") else None))
    tscv = TimeSeriesSplit(n_splits=5)
    last_fold = None
    for train_idx, test_idx in tscv.split(X):
        last_fold = (train_idx, test_idx)

    train_idx, test_idx = last_fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models[0].fit(X_train, y_train)
    y_pred_rf = models[0].predict(X_test)
    models[1].fit(X_train, y_train)
    y_pred_1 = models[1].predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf, digits=3)
    acc_1 = accuracy_score(y_test, y_pred_1)
    cm_1 = confusion_matrix(y_test, y_pred_1)
    report_1 = classification_report(y_test, y_pred_1, digits=3)

    return {
        "models": models,
        "feature_cols": feature_cols,
        "acc_rf": acc_rf,
        "cm_rf": cm_rf,
        "report_rf": report_rf,
		"acc_1": acc_1,
		"cm_1": cm_1,
		"report_1": report_1,
        "test_index": df_feat.iloc[test_idx].index,
        "y_test": y_test,
        "y_pred_rf": y_pred_rf,
		"y_pred_1": y_pred_1,
    }


# ------------------------------
# Plotting
# ------------------------------
def plot_price_with_signals(df_feat: pd.DataFrame, test_index, y_pred, img_path: Path, ticker: str):
    """pic1: Kurs + Vorhersagepfeile auf dem Testsplit"""
    segment = df_feat.loc[test_index]
    close = segment["Close"]

    up_idx = segment.index[y_pred == 1]
    down_idx = segment.index[y_pred == 0]

    plt.figure(figsize=(12, 6))
    plt.plot(segment.index, close, label="Close")
    plt.scatter(up_idx, close.loc[up_idx], marker="^",s=80, label="Vorhersage: Steigt",alpha=0.8)
    plt.scatter(down_idx, close.loc[down_idx], marker="v", s=80, label="Vorhersage: Fällt",
        alpha=0.8)
    plt.title(f"{ticker} – Kurs & ML-Signale (Testsample)")
    plt.xlabel("Datum")
    plt.ylabel("Preis")
    plt.legend()
    plt.tight_layout()
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path, dpi=150)
    plt.close()


def plot_metrics(cm: np.ndarray, acc: float, report: str, img_path: Path,
    model_name: str, ticker: str):
    """pic2: Confusion Matrix + Accuracy + Kurzreport als Text"""
    plt.figure(figsize=(10, 6))

    # Confusion Matrix als Heatmap
    ax = plt.gca()
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fällt", "Steigt"])
    ax.set_yticklabels(["Fällt", "Steigt"])
    ax.set_xlabel("Vorhersagt")
    ax.set_ylabel("Ist")

    # Werte in Zellen
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black")

    plt.title(f"{ticker} – {model_name} | Accuracy={acc:.3f}")
    plt.tight_layout(rect=[0, 0.25, 1, 1])

    # Report unten als Textfeld
    plt.gcf().text(0.02, 0.02, report, fontsize=9, va="bottom", ha="left", family="monospace")

    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path, dpi=150)
    plt.close()


# ------------------------------
# State lesen/schreiben
# ------------------------------
def load_state(path: Path = STATE_FILE) -> Dict:
    """ Lade den State aus einer JSON-Datei. """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: Dict, path: Path = STATE_FILE) -> None:
    """ Speichere den State in einer JSON-Datei. """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_news_today_structure(state: Dict, ticker: str, today: str) -> None:
    """
    Erwartete Struktur:
      state[ticker]["news"] = [[url1, date1], [url2, date2], ...]
      Zusätzlich speichern wir 'last_ml' Informationen für den Tag.
    """
    if ticker not in state:
        state[ticker] = {}
    state[ticker].setdefault("news", [])
    state[ticker].setdefault("ml", {})
    state[ticker]["ml"].setdefault(today, {})


# ------------------------------
# Hauptablauf pro Ticker
# ------------------------------
def run_daily_for_ticker(
    ticker: str,
    model_kind: str = "rf",
    img_root: Path = IMG_ROOT,
    streamlit_base_url: str = STREAMLIT_BASE_URL,
    state_path: Path = STATE_FILE,
) -> ModelResult:
    """Führe den ML-Durchlauf für einen einzelnen Ticker durch."""
    today = datetime.now(timezone.utc).astimezone().date().isoformat()
    # prüfen, ob es für den Tag schon Ergebnisse gibt
    img_dir = img_root / ticker / today
    if img_dir.exists() and any(img_dir.glob("*.png")):
        logger.info("Ergebnisse für %s am %s existieren schon, überspringe.", ticker, today)
        state = load_state(state_path)
        ml_block = state.get(ticker, {}).get("ml", {}).get(today)
        if ml_block:
            return ModelResult(
                ticker=ticker,
                date=today,
                model_name=ml_block.get("model", "Unknown"),
                accuracy_rf=ml_block.get("accuracy_rf", 0.0),
                report_rf=ml_block.get("report_rf", ""),
                accuracy_1=ml_block.get("accuracy_1", 0.0),
				report_1=ml_block.get("report_1", ""),
                prediction_rf=ml_block.get("pred_next", None),
                link_streamlit=f"{streamlit_base_url}?ticker={ticker}&date={today}",
                img_dir=img_dir,
                img_pic_rf1=img_dir / "pic_rf1.png",
                img_pic_11=img_dir / "pic_11.png",
                img_pic_rf2=img_dir / "pic_rf2.png",
                img_pic_12=img_dir / "pic_12.png",
            )

	#####

    # 1) Daten laden + Features
    raw = load_hist_prices(ticker)
    feat = engineer_features(raw)

    # 2) Train/Eval
    res = train_and_evaluate(feat, model_kind=model_kind)
    models = res["models"]
    acc_rf = res["acc_rf"]
    cm_rf = res["cm_rf"]
    report_rf = res["report_rf"]
    test_index = res["test_index"]
    y_pred_rf = res["y_pred_rf"]
    acc_1 = res["acc_1"]
    cm_1 = res["cm_1"]
    report_1 = res["report_1"]
    y_pred_1 = res["y_pred_1"]

    # 3) Bilder speichern
    img_dir = img_root / ticker / today
    pic_rf1 = img_dir / "pic_rf1.png"  # Kurs + Signale
    pic_rf2 = img_dir / "pic_rf2.png"  # Metriken
    pic_11 = img_dir / "pic_11.png"  # Kurs + Signale
    pic_12 = img_dir / "pic_12.png"  # Metriken
    plot_price_with_signals(feat, test_index, y_pred_rf, pic_rf1, ticker)
    plot_metrics(cm_rf, acc_rf, report_rf, pic_rf2,
        model_name="RandomForest", ticker=ticker)
    plot_price_with_signals(feat, test_index, y_pred_1, pic_11, ticker)
    plot_metrics(cm_1, acc_1, report_1, pic_12,
        model_name=("LogisticRegression" if model_kind == "rf" else model_kind), ticker=ticker)

    # 4) State aktualisieren (News bleiben wie vorgesehen in state[ticker]['news'])
    state = load_state(state_path)
    ensure_news_today_structure(state, ticker, today)

    # ML Kurzinfo des Tages im State (z. B. für die Streamlit-Seite)
    # Optional: "direction" = Modell-Prognose für "morgen" auf Basis des letzten verfügbaren Samples
    last_row = feat.iloc[[-1]]
    feature_cols = res["feature_cols"]
    try:
        pred_next = int(models[0].predict(last_row[feature_cols].values)[0])
    except (ValueError, IndexError):
        pred_next = None

    state[ticker]["ml"][today] = {
        "model_rf": "RandomForest",
        "accuracy_rf": acc_rf,
        "report_rf": report_rf,
        "model_1": "LogisticRegression" if model_kind == 'rf' else model_kind,
        "accuracy_1": acc_rf,
        "report_1": report_rf,
        "pred_next": pred_next,  # 1=steigt, 0=fällt, None=keine Prognose
        "img_dir": str(img_dir.relative_to(BASE_DIR).as_posix()),
        "imgs": ["pic_rf1.png", "pic_rf2.png","pic_11.png", "pic_12.png"],
    }
    save_state(state, state_path)

    # 5) Streamlit-Link für Alerts
    link = f"{streamlit_base_url}?ticker={ticker}&date={today}"

    return ModelResult(
        ticker=ticker,
        date=today,
        model_name=("LogisticRegression" if model_kind == "rf" else model_kind),
        accuracy_rf=acc_rf,
        accuracy_1=acc_1,
        report_rf=report_rf,
        report_1=report_1,
        prediction_rf=pred_next,
        link_streamlit=link,
        img_dir=img_dir,
        img_pic_rf1=pic_rf1,
        img_pic_11=pic_11,
        img_pic_rf2=pic_rf2,
        img_pic_12=pic_12,
    )


def run_daily_for_tickers(
    ts: List[str],
    model_kind: str = "rf",
    img_root: Path = IMG_ROOT,
    streamlit_base_url: str = STREAMLIT_BASE_URL,
    state_path: Path = STATE_FILE,
) -> List[ModelResult]:
    """Führe den ML-Durchlauf für eine Liste von Tickers durch."""
    results: List[ModelResult] = []
    for t in ts:
        try:
            res = run_daily_for_ticker(
                t,
                model_kind=model_kind,
                img_root=img_root,
                streamlit_base_url=streamlit_base_url,
                state_path=state_path,
            )
            results.append(res)
        except (RuntimeError, ValueError) as e:
            logger.warning("In ml_daily zeile407 -> %s: %s", t, e)
    return results


# ------------------------------
# Direktstart (manuelles Testen)
# ------------------------------
if __name__ == "__main__":
    tickers = ["AAPL", "RHM.DE", "O"]
    out = run_daily_for_tickers(
        tickers,
        model_kind="rf",
        streamlit_base_url=STREAMLIT_BASE_URL,
    )
    for r in out:
        print(f"{r.ticker} ({r.date}) acc={r.accuracy:.3f} → {r.link_streamlit}")
