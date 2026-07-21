"""
Structured Logger Setup for Fire & Smoke Detection
"""

import logging
import os
import sys
import config

def setup_logger(name="fire_smoke_detector"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        # File handler
        log_file = os.path.join(config.LOG_DIR, "detector.log")
        f_handler = logging.FileHandler(log_file)
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger

logger = setup_logger()
