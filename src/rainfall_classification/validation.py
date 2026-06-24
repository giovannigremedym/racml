#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Model Validation & Evaluation
=====================================
Module for calculating rigorous acoustic classification metrics 
and generating publication-ready visualizations.

References:
- Bedoya et al. (2017): Accuracy = (Sensitivity + Specificity) / 2
- Xavier et al. (2024): FAR (False Alarm Rate) = FP / (FP + TN)

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-05-15
:Version: 4.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from src.utils import get_standard_logger

logger = get_standard_logger("ValidationModule")

def evaluate_model(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: Optional[np.ndarray] = None, 
    classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of metrics for rainfall classification.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Model predictions.
    y_proba : np.ndarray, optional
        Class probabilities for PR-AUC calculation.
    classes : list of str, optional
        Names of the classes in order.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing scalars (F1, Sens, Spec, FAR, Bedoya Acc) 
        and matrices (Confusion Matrix).
    """
    if classes is None:
        classes = [str(c) for c in np.unique(y_true)]
        
    cm = confusion_matrix(y_true, y_pred)
    ncm = confusion_matrix(y_true, y_pred, normalize='true')

    # 1. Base Totals per Class (Macro Average Strategy)
    FP = cm.sum(axis=0) - np.diag(cm) 
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # 2. Score Derivation
    with np.errstate(divide='ignore', invalid='ignore'):
        # Sensitivity (Recall)
        sens_per_class = np.nan_to_num(TP / (TP + FN), nan=0.0)
        # Specificity
        spec_per_class = np.nan_to_num(TN / (TN + FP), nan=0.0)
        # False Alarm Rate (Xavier 2024)
        far_per_class = np.nan_to_num(FP / (FP + TN), nan=0.0)

    # 3. Macro Aggregation (Weights all classes equally regardless of frequency)
    sensitivity = np.mean(sens_per_class)
    specificity = np.mean(spec_per_class)
    far = np.mean(far_per_class)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Bedoya Accuracy = (Sensitivity + Specificity) / 2
    accuracy = (sensitivity + specificity) / 2.0
    
    # 4. PR-AUC Calculation
    prauc = np.nan
    if y_proba is not None:
        try:
            if len(classes) > 2:
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                prauc = average_precision_score(y_true_bin, y_proba, average="macro")
            else:
                prauc = average_precision_score(y_true, y_proba[:, 1])
        except Exception as e:
            logger.warning(f"Could not calculate PR-AUC: {e}")

    logger.info(f"Evaluation complete.")
    
    return {
        "f1_macro": f1,
        "prauc_macro": prauc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "far": far,
        "conf_matrix": cm,
        "norm_conf_matrix": ncm,
        "classes": classes
    }


def plot_multimodel_pr_curves(
    y_true: np.ndarray,
    y_probs_dict: Dict[str, np.ndarray],
    palette: Dict[str, str],
    champion_models: Optional[List[str]] = None,
    title: Optional[str] = "Precision-Recall AUC per Class and Model"
) -> plt.Figure:
    """
    Plots Precision-Recall curves for multiple models across multiple classes.
    
    Generates a grid of subplots (one per class). Uses the provided palette 
    for class colors and different line styles to differentiate models. 
    Highlights champion models with thicker lines while maintaining visibility
    for comparison models. Legends are placed outside the grid.

    Parameters
    ----------
    y_true : np.ndarray
        1D array of true labels.
    y_probs_dict : Dict[str, np.ndarray]
        Dictionary mapping model names to their respective predicted 
        probability matrices (shape: [n_samples, n_classes]).
    palette : Dict[str, str]
        Dictionary mapping class names to HEX colors. The keys dictate 
        the target classes and the order of the subplots.
    champion_models : List[str], optional
        List of model names to highlight. Non-champions will be plotted 
        with slightly lower opacity and thinner lines to reduce visual noise.
    title : str, optional
        The overarching title of the graphic.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the subplot grid.
    """
    
    classes = list(palette.keys())
    n_classes = len(classes)
    
    # Binarize labels for One-vs-Rest evaluation
    y_true_bin = label_binarize(y_true, classes=classes)
    # Handle binary edge case dynamically if needed
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        
    # Define line styles for up to 4 models (extendable if needed)
    line_styles = ['-', '--', '-.', ':']
    models = list(y_probs_dict.keys())
        
    style_map = {model: line_styles[i % len(line_styles)] for i, model in enumerate(models)}
    champs = champion_models or []
    
    # Setup subplot grid. 
    # Base column width increased to 6.5 to accommodate external legends.
    cols = 3 if n_classes > 2 else 2
    rows = math.ceil(n_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 4), sharex=True, sharey=True)
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.05)
        
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Plotting loop per class to maintain color consistency
    for i, class_name in enumerate(classes):
        ax = axes_flat[i]
        class_color = palette[class_name]
        
        for model_name, y_probs in y_probs_dict.items():
            # Extract probabilities for the current class
            class_idx = classes.index(class_name)
            probs = y_probs[:, class_idx]
            
            # Calculate PR and AUC
            precision, recall, _ = precision_recall_curve(y_true_bin[:, class_idx], probs)
            pr_auc = auc(recall, precision)
            
            # Determine visual weight based on champion status
            is_champ = model_name in champs or not champs
            lw = 2.5 if is_champ else 1.5    # Increased from 1.2
            alpha = 1.0 if is_champ else 0.7 # Increased from 0.35
            zorder = 3 if is_champ else 2
            
            ax.plot(
                recall, 
                precision, 
                color=class_color, 
                linestyle=style_map[model_name], 
                linewidth=lw, 
                alpha=alpha,
                zorder=zorder,
                label=f"{model_name} (AUC = {pr_auc:.2f})"
            )
            
        ax.set_title(class_name, fontsize=14, color=class_color, fontweight='bold')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if i >= len(classes) - cols:
            ax.set_xlabel("Recall", fontsize=12)
        if i % cols == 0:
            ax.set_ylabel("Precision", fontsize=12)
            
        # Class-specific legend placed OUTSIDE the plot grid
        ax.legend(
            bbox_to_anchor=(1.02, 1), 
            loc="upper left", 
            fontsize=9, 
            framealpha=0.9,
            borderaxespad=0.
        )
        
    # Remove empty subplots
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])
        
    sns.despine(fig=fig)
    
    # Increase w_pad to ensure legends do not overlap with the next column's y-axis
    plt.tight_layout(w_pad=2.0)
    
    return fig


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: Optional[str] = "Confusion Matrix",
    is_normalized: bool = False
) -> plt.Figure:
    """
    Plots a standardized confusion matrix heatmap.

    Dynamically handles both raw count matrices and normalized (percentage) 
    matrices, adjusting the color scale limits and text formatting automatically
    via a unified interface.

    Parameters
    ----------
    cm_df : pd.DataFrame
        The confusion matrix data. Indices should represent True Labels, 
        and columns should represent Predicted Labels.
    title : str, optional
        The overarching title of the graphic.
    is_normalized : bool, default=False
        If True, treats the values as proportions (0.0 to 1.0) and formats 
        them with decimals. If False, formats them as integers.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the heatmap.
    """    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Adjust formatting and color limits based on normalization status
    if is_normalized:
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        fmt = "d"
        vmin, vmax = None, None  # Let Seaborn decide dynamically based on max count
        
    # Standard uniform colormap for confusion matrices
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    sns.heatmap(
        cm_df, 
        annot=True, 
        fmt=fmt, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax,
        square=True, 
        linewidths=1, 
        linecolor='#F0F0F0', 
        cbar_kws={"shrink": .8}, 
        ax=ax,
        annot_kws={"size": 12}
    )
    
    if title:
        ax.set_title(title, fontsize=16, pad=20)
        
    ax.set_ylabel("True Label", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
    
    # Rotate tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    
    plt.tight_layout()
    
    return fig


if __name__ == '__main__':
    pass