"""
utils.py — Seeding, logging setup, and path helpers.
"""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

from src.config import RANDOM_SEED, REPORTS_DIR, TRAINING_LOG_PATH


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Create (or retrieve) a named logger.

    Outputs to stdout and optionally to reports/training_log.txt.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — return existing
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_to_file:
        ensure_dir(REPORTS_DIR)
        fh = logging.FileHandler(TRAINING_LOG_PATH, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Replace spaces and special chars with underscores for safe filenames."""
    import re
    return re.sub(r"[^\w\-.]", "_", name)
