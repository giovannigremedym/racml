#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Utilities
=================
Global utility functions for the racml project.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-10
:Version: 3.0.0
"""

import sys
import logging

def get_standard_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standard library logger with uniform formatting.

    Parameters
    ----------
    name : str
        The name of the logger instance (typically the module or process name).

    Returns
    -------
    logging.Logger
        A configured standard library Logger instance outputting to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

if __name__ == "__main__":
    pass