"""Entry point of the Stock Notifier application."""
from pathlib import Path
import logging
from src.app.config import load_config
from src.app.logging_setup import setup_logging
from src.app.core import run_once
from src.app.utils import mask_secret
from src.app.ntfy import notify_ntfy

def main():
    """
    Entry point of the Stock Notifier application.
    """
    cfg = load_config()
    setup_logging(cfg["log"])

    logger = logging.getLogger()

    logger.info(
        "Configuration loaded: ntfy.server=%s | ntfy.topic(masked)=%s | log.level=%s",
        cfg["ntfy"]["server"],
        mask_secret(cfg["ntfy"]["topic"]),
        cfg["log"]["level"],
    )

    # TO DO: Run one monitoring cycle via run_once using settings from cfg
    run_once(
        tickers=cfg["tickers"],
        threshold_pct=float(cfg["threshold_pct"]),
        ntfy_server=cfg["ntfy"]["server"],
        ntfy_topic=cfg["ntfy"]["topic"],
        state_file=Path(cfg["state_file"]),
        market_hours_cfg=cfg["market_hours"],
        test_cfg=cfg["test"],
        news_cfg=cfg["news"],
    )
    notify_ntfy(
        server=cfg["ntfy"]["server"],
        topic=cfg["ntfy"]["topic"],
        title=cfg["ntfy"]["title"],
        message=cfg["ntfy"]["message"],
        dry_run=False,
        markdown=True,
        click_url="https://example.com",
     )

 # Remove once implemented



if __name__ == "__main__":
    main()
