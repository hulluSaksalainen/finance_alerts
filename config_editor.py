""" A Streamlit app to edit the config.json file for the stock notifier."""
import json
from pathlib import Path
import streamlit as st

CONFIG_FILE = Path("json/config.json")


def load_config():
    """ Load the config from the config.json file or return default values. """
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "tickers": ["AAPL", "MSFT"],
        "ntfy": {
            "server": "https://ntfy.sh",
            "topic": "stock-alerts"
        },
        "logging": {
            "level": "INFO",
            "file": "notifier.log"
        },
        "market_hours": {
            "start": "09:30",
            "end": "16:00",
            "timezone": "America/New_York"
        }
    }


def save_config(config):
    """ Save the config dictionary to the config.json file. """
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def main():
    """ Main function to run the Streamlit app. """
    st.title("⚙️ Stock Notifier Config Editor")

    config = load_config()

    # --- Ticker ---
    with st.expander("📈 Ticker", expanded=True):
        tickers = st.text_area("Zu überwachende Ticker (durch Komma getrennt)",
            value=",".join(config.get("tickers", [])))
        config["tickers"] = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        config["threshold_pct"] = st.number_input(
            "Threshold Percentage",
            min_value=0.1, max_value=10.0, step=0.1,
            value=float(config.get("threshold_pct", 3.0))
        )
        config["state_file"] = st.text_input("State File Path",
            value=config.get("state_file", "json/alert_state.json"))


    # --- NTFY ---
    with st.expander("📨 ntfy-Einstellungen", expanded=True):
        config["ntfy"]["server"] = st.text_input("Server-URL", value=config["ntfy"].get("server",
            "https://ntfy.sh"))
        config["ntfy"]["topic"] = st.text_input("Topic",
            value=config["ntfy"].get("topic", "stock-alerts"))
        config["ntfy"]["title"] = st.text_input("Title",
            value=config["ntfy"].get("title", "stock-alerts"))
        config["ntfy"]["message"] = st.text_input("Standart message",
            value=config["ntfy"].get("message", "My personal stock-alerts"))
        config["ntfy"]["markdown"] = st.radio(
            "Use Markdown",
            options=[True, False],
            index=0 if config["log"].get("to_file", True) else 1,
            format_func=lambda x: "Yes" if x else "No"
        )

    # --- Logging ---
    with st.expander("📝 Logging", expanded=True):
        config["log"]["level"] = st.selectbox(
            "Log-Level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config["log"].get("level", "INFO"))
        )
        config["log"]["to_file"] = st.radio(
            "in Log-Datei schreiben",
            options=[True, False],
            index=0 if config["log"].get("to_file", True) else 1,
            format_func=lambda x: "Ja" if x else "Nein"
        )
        if config["log"]["to_file"]:
            config["log"]["file"] = st.text_input("Log-Datei",
                value=config["log"].get("file", "alerts.log"))
            config["log"]["file_max_bytes"] = st.number_input("Maximale Log-Dateigröße (Bytes)",
                min_value=100000, max_value=10000000, step=100000,
                value=config["log"].get("file_max_bytes", 1000000))
            config["log"]["file_backup_count"] = st.number_input("Anzahl der Log-Backups",
                min_value=1, max_value=10, step=1,
                value=config["log"].get("file_backup_count", 3))

    # --- News ---
    with st.expander("📰 News-Einstellungen", expanded=True):
        config["news"]["enabled"] = st.radio(
            "Activate News-Alerts",
            options=[True, False],
            index=0 if config["news"].get("enabled", True) else 1,
            format_func=lambda x: "Yes" if x else "No"
        )
        if config["news"]["enabled"]:
            config["news"]["lookback_hours"] = st.number_input("Look back (hours)",
                min_value=1, max_value=48, step=1,
                value=config["news"].get("lookback_hours", 12))
            config["news"]["lang"] = st.selectbox("Language",
                ["de", "fr", "en", "fi", "it", "nl", "es", "sv"],
                index=["de", "fr", "en", "fi", "it", "nl", "es",
                    "sv"].index(config["news"].get("lang", "de"))
                )
            config["news"]["country"] = st.selectbox("Country",
                ["DE", "FR", "UK", "FI", "IT", "NL", "ES", "SV", "US"],
                index=["DE", "FR", "UK", "FI", "IT", "NL", "ES", "SV",
                    "US"].index(config["news"].get("country", "DE"))
                )
            config["news"]["limit"] = st.number_input("Maximale Number of Articles per Ticker",
                min_value=1, max_value=20, step=1,
                value=config["news"].get("limit", 5))

    # --- Market Times ---
    with st.expander("⏰ Marktzeiten", expanded=True):
        col1, col2 = st.columns(2)
        config["market_hours"]["enabled"] = st.radio(
            "Only during market hours",
            options=[True, False],
            index=0 if config["market_hours"].get("enabled", True) else 1,
            format_func=lambda x: "Yes" if x else "No"
        )
        if config["market_hours"]["enabled"]:
            config["market_hours"]["start_hour"] = col1.text_input("Startzeit (HH:MM)",
                value=config["market_hours"].get("start_hour", "09:30"))
            config["market_hours"]["end_hour"] = col2.text_input("Endzeit (HH:MM)",
                value=config["market_hours"].get("end_hour", "16:00"))
            config["market_hours"]["tz"] = st.text_input("Zeitzone",
                value=config["market_hours"].get("tz", "America/New_York"))
            config["market_hours"]["days_mon_to_fri_only"] = st.radio(
            "Weekdays only (Mon-Fri)",
            options=[True, False],
            index=0 if config["market_hours"].get("days_mon_to_fri_only", True) else 1,
            format_func=lambda x: "Yes" if x else "No"
        )

    # --- Speichern ---
    if st.button("💾 Speichern"):
        save_config(config)
        st.success("Config gespeichert!")

    # --- Vorschau ---
    st.subheader("📜 Vorschau")
    st.json(config)


if __name__ == "__main__":
    main()
