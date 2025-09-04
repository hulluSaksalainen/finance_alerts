"""Configuration management for the Stock Notifier application."""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Default configuration values used if no config.json or .env overrides are provided.
DEFAULTS: Dict[str, Any] = {
    "log": {
        "level": "INFO",               # Default logging level
        "to_file": False,              # Write logs to file? (default: only console)
        "file_path": "alerts.log",     # Log file location
        "file_max_bytes": 1_000_000,   # Max size of log file before rotation
        "file_backup_count": 3         # Number of rotated log files to keep
    },
    "ntfy": {
        "server": "https://ntfy.sh",   # Default ntfy server
        "topic": "Finnisch",
        "title": "Stock Notifier Alert",
        "message":"Test notification from Stock Notifier",
        "markdown": True,

    },
    "tickers": ["AAPL"],               # Default ticker(s) to monitor
    "threshold_pct": 3.0,              # Default % threshold for alerts
    "state_file": "alert_state.json",  # File to persist alert state (anti-spam)
    "market_hours": {                  # Market hours configuration
        "enabled": True,
        "tz": "Europe/Berlin",         # Default timezone
        "start_hour": 8,
        "end_hour": 22,
        "days_mon_to_fri_only": True   # Only Monday–Friday
    },
    "test": {                          # Test mode settings
        "enabled": False,
        "bypass_market_hours": True,
        "force_delta_pct": None,       # Simulate price changes
        "dry_run": False               # Dry-run: do not send actual notifications
    },
}

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v

    return out

def load_config(path: str = "json/config.json") -> Dict[str, Any]:
    """
    Load the con
    figuration for the application.

    Priority:
    1. Default values from DEFAULTS
    2. Overrides from config.json (if present)
    3. Overrides from environment variables (.env or OS-level)
    """
    load_dotenv()  # Load environment variables from .env file if present
    user: Dict[str, Any] = {}
    p = Path(path)
    
    # Pfad zum Verzeichnis
    verzeichnis = Path('./json')

    # Dateien und Ordner auflisten

    for datei in verzeichnis.rglob('*'):
        if datei.is_file() and datei.name[0] != '.':
            print(datei.resolve())


    if p.exists():
        try:
            user = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"config.json could not be read: {e}") from e
    else:
        raise FileNotFoundError(f"Configuration file {path} not found")
    if "news" not in user:
        print("news not in user")
    else:
        print(user["news"],"=user[news]")
    cfg = deep_merge(DEFAULTS,user)
    if "news" not in cfg:
        print("news not in cfg")
    else:
        print(cfg["news"],"=cfg[news]")
    cfg["log"]["level"] = os.getenv("LOG_LEVEL", cfg["log"]["level"]).upper()
    cfg["ntfy"]["server"] = os.getenv("NTFY_SERVER", cfg["ntfy"]["server"])
    cfg["ntfy"]["topic"] = os.getenv("NTFY_TOPIC", cfg["ntfy"]["topic"])
    if not cfg["ntfy"]["topic"]:
        raise ValueError("""ntfy.topic must be set in config.json or via
            environment variable NTFY_TOPIC""")
    return cfg
