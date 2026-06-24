#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Rainfall_classification
================
Machine learning classification, hyperparameter tuning, and validation.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-19
:Version: 1.0.0
"""

from sklearn import set_config
set_config(transform_output="pandas")

from .classifiers import (
    ClassifierConfig,
    ClassifierFactory,
    optimize_model
)

from .validation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_multimodel_pr_curves,
)
__all__ = [
    "ClassifierConfig",
    "ClassifierFactory",
    "optimize_model",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_multimodel_pr_curves",
]