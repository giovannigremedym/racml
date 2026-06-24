#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Feature Engineering
===========================
Off-the-shelf acoustic metrics extraction and dimensionality reduction to
isolate an optimal, lightweight subset of features.

OBS.: The module acoustic_metrics only works with 1D array data.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-05-15
:Version: 4.0.0
"""

from sklearn import set_config
set_config(transform_output="pandas")

from .acoustic_metrics import (
    AcousticMetrics, 
    AcMConfig
)


from .selector import (
    select_features,
    plot_correlation_matrix,
    plot_feature_separability,
    plot_gini_importances,
)

__all__ = [
    "AcMConfig",
    "AcousticMetrics",
    "select_features",
    "plot_correlation_matrix",
    "plot_feature_separability",
    "plot_gini_importances",
]