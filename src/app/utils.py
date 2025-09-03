"""Utility functions for the Stock Notifier application."""
def mask_secret(s: str, keep: int = 1) -> str:
    """Maskiert sensible Strings für Logging-Ausgaben."""
    if not s:
        return "(unset)"
    elif len(s) > keep * 2:
        return s[keep] + "…" + s[-keep]
    else:
        return s[0] + "…" + s[-1]
