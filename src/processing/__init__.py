#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Signal Processing
==================
Audio segmentation and synthetic augmentation.
OBS.: This module only works with 1D array data.

Metadata
--------
:Author: Giovanni G. R. Milan
:Version: 2.0.0
"""

from .augmenter import (
    AudioAugmenter, 
    AugConfig
)
from .segmenter import (
    AudioSegmenter, 
    SegConfig
)

__all__ = [
    "AudioAugmenter",
    "AugConfig",
    "AudioSegmenter",
    "SegConfig",
]