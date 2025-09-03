""" notify einbauen"""
import logging
import requests
from src.app.utils import mask_secret

logger = logging.getLogger("stock-alerts")


def notify_ntfy(
    server: str,
    topic: str,
    title: str,
    message: str,
    *,
    dry_run: bool = False,
    markdown: bool = False,
    click_url: str | None = None,
) -> None:
    """
    Send a push notification via ntfy.sh.
    """
    # TO DO: If dry_run is True, log the message and return without sending
    if dry_run:
        logger.info("Dry run: Notification to %s at %s | Title: %s | Message: %s",
            mask_secret(topic), server, title, message)
        return

    url = f"{server.rstrip('/')}/{topic}"
    headers = {
        "Title": title,
        "Priority": "high",
    }

    if markdown:
        headers["Markdown"] = "yes"

    if click_url:
        headers["Click"] = click_url


    try:
        safe_headers = {k: str(v).encode("latin-1", errors="ignore").decode("latin-1") for k,
            v in headers.items()}
        r = requests.post(url, data=message.encode("utf-8"), headers=safe_headers, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning(e)

    # Temporary print for demonstration
    logger.info("Notification sent to %s at %s", mask_secret(topic), server)
