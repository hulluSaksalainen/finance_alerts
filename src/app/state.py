"""Module for loading and saving the alert state."""
import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger("stock-alerts")


def load_state(path: Path) -> Dict[str, str]:
    """
    Load the last alert "state" from a JSON file.

    The state keeps track of which direction (up/down/none) a stock
    has already triggered an alert for. This prevents sending duplicate
    notifications every run.
    """
    # TO DO: Prüfen, ob die Datei existiert und deren Inhalt als JSON laden
    if path.exists():
        # TO DO: Bei Erfolg den geladenen Zustand zurückgeben und einen Debug-Log schreiben
        try:
            content = path.read_text(encoding="utf-8")
            state = json.loads(content)
            logger.debug("Loaded state from %s: %s", path, state)
            return state
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.warning("Failed to load state from %s: %s", path, e)
            return {}
    # TO DO: Bei Fehlern eine Warnung loggen und ein leeres Dict zurückgeben
    else:
        logger.debug("State file %s does not exist; starting with empty state", path)
        return {}

def save_state(path: Path, state: Dict[str, str]) -> None:
    """
    Save the current alert state to disk.
    """
    # TO DO: Den Zustand als JSON (UTF-8) in die Datei schreiben
    try:
        content = json.dumps(state, indent=2)
        path.write_text(content, encoding="utf-8")
        logger.debug("Saved state to %s: %s", path, state)
    # TO DO: Einen Debug-Log mit dem gespeicherten Zustand ausgeben
    except (OSError, IOError) as e:
        logger.warning("Failed to save state to %s: %s", path, e)
