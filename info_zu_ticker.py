""" streamlit-Seite: Ticker-Informationen (Bilder + News) """
# info_zu_ticker.py
import json
from pathlib import Path
#from urllib.parse import parse_qs, urlparse

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR  # falls diese Datei im Repo-Root liegt
STATE_FILE = REPO_ROOT / "json" / "alert_state.json"
IMG_ROOT = REPO_ROOT / "src" / "img"


def load_state():
    """ Lade den State aus der state_file (oder leeres Dict) """
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def get_query_params():
    """ Query-Parameter aus der URL holen (Ticker, Datum) """
    # Streamlit 1.30+: st.query_params, abw. fallback:
    try:
        qp = st.query_params
        return {k: v[0] if isinstance(v, list) else v for k, v in qp.items()}
    except (AttributeError, RuntimeError):
        # Fallback über st.experimental_get_query_params (ältere Versionen)
        qp = st.experimental_get_query_params()
        return {k: v[0] if isinstance(v, list) else v for k, v in qp.items()}


def main():
    """ Hauptfunktion der Streamlit-Seite """
    st.set_page_config(page_title="Ticker Info", page_icon="📈", layout="wide")
    st.title("📈 Ticker-Informationen (Bilder + News)")

    params = get_query_params()
    ticker = params.get("ticker", "")
    date = params.get("date", "")

    # Falls ohne Query geöffnet, einfache Auswahl anbieten
    state = load_state()
    all_tickers = sorted(list(state.keys()))
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.selectbox("Ticker", all_tickers,
            index=(all_tickers.index(ticker) if ticker in all_tickers else 0)
            if all_tickers else st.text_input("Ticker"))
    with col2:
        available_dates = sorted(list(state.get(ticker, {}).get("ml",
            {}).keys())) if ticker in state else []
        date = st.selectbox("Datum (YYYY-MM-DD)", available_dates,
            index=(available_dates.index(date) 
                if date in available_dates else (len(available_dates)-1
            	if available_dates else 0))
            if available_dates else st.text_input("Datum (YYYY-MM-DD)"))

    if not ticker or not date:
        st.info("Bitte Ticker und Datum wählen.")
        return

    st.subheader(f"{ticker} – {date}")

    # ML-Block anzeigen
    ml_block = state.get(ticker, {}).get("ml", {}).get(date, {})
    if not ml_block:
        st.warning("Keine ML-Daten für diesen Tag gefunden.")
    else:
        colA, colB, colC = st.columns(3)
        colA.metric("Modell", ml_block.get("model", "-"))
        colB.metric("Accuracy", f"{ml_block.get('accuracy', 0.0):.3f}")
        pred_next = ml_block.get("pred_next", None)
        pred_txt = "Steigt" if pred_next == 1 else ("Fällt" if pred_next == 0 else "—")
        colC.metric("Prognose für nächsten Handelstag", pred_txt)

    # Bilder laden
    img_dir_rel = ml_block.get("img_dir")
    img_files = ml_block.get("imgs", [])
    if img_dir_rel:
        img_dir = REPO_ROOT / img_dir_rel
    else:
        img_dir = IMG_ROOT / ticker / date  # Fallback

    st.subheader("🖼️ Bilder")
    if img_dir.exists():
        cols = st.columns(2)
        for i, name in enumerate(img_files or []):
            p = img_dir / name
            if p.exists():
                with cols[i % 2]:
                    st.image(str(p), caption=name, use_column_width=True)
            else:
                st.info(f"Bild fehlt: {name}")
    else:
        st.info("Kein Bildordner vorhanden.")

    # News des Tages
    st.subheader("📰 News (heutiger Tag)")
    news_list = state.get(ticker, {}).get("news", [])
    # news: [[url, date], ...]
    todays_news = [n for n in news_list if (len(n) >= 2 and str(n[1]).startswith(date))]
    if todays_news:
        for url, nd in todays_news:
            st.markdown(f"- [{nd}] {url}")
    else:
        st.info("Keine News für diesen Tag hinterlegt.")


if __name__ == "__main__":
    main()
