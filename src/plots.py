#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Feature Engineering Plots
=================================
Visualization tools tightly coupled with the feature selection pipeline.
Provides observability into multicollinearity and Random Forest Gini importances.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-05-15
:Version: 3.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from src.utils import get_standard_logger

logger = get_standard_logger("FeaturePlots")

# Configure publication-quality plot settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def plot_correlation_matrix(
    X: pd.DataFrame, 
    threshold: float = 0.85
) -> plt.Figure:
    """
    Plots a highly stylized correlation heatmap for the feature matrix.
    Useful for visualizing the dataset before dropping collinear features.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix (ideally right before the collinearity filter).
    threshold : float, default=0.85
        The Pearson correlation threshold used in the selector.
    """
    logger.info("Generating Correlation Matrix plot...")
    corr = X.corr().abs()
    
    # Generate a mask for the upper triangle to avoid visual duplication
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Custom diverging colormap (blue to red)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1.0, vmin=0, center=0.5,
        square=True, linewidths=.5, cbar_kws={"shrink": .75}, ax=ax
    )
    
    ax.set_title(f"Acoustic Features Collinearity Matrix (Threshold = {threshold})", pad=20, fontsize=16)
    
    plt.tight_layout()
    
    return fig

def plot_gini_importances(
    gini_dict: Dict[str, float], 
    top_n: int = 30
) -> plt.Figure:
    """
    Plots a horizontal bar chart of Random Forest Gini importances.

    Parameters
    ----------
    gini_dict : Dict[str, float]
        Dictionary mapping feature names to their Gini importance.
    top_n : int, default=30
        Maximum number of top features to display to avoid clutter.
    """
    logger.info("Generating Gini Importances plot...")
    
    # Sort dictionaries by importance (descending)
    sorted_features = sorted(gini_dict.items(), key=lambda item: item[1], reverse=True)
    plot_data = sorted_features[:top_n]
    
    names = [x[0] for x in plot_data]
    importances = [x[1] for x in plot_data]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Uniform color for all top features
    bars = ax.barh(names, importances, color='steelblue', edgecolor='black', alpha=0.8)
    
    ax.set_xlabel("Mean Decrease in Impurity (Gini)", fontsize=12)
    ax.set_ylabel("Acoustic Features", fontsize=12)
    ax.set_title(f"Random Forest Feature Selection (Top {top_n})", fontsize=16, pad=15)
    
    # Invert y-axis to have the most important feature at the top
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_feature_separability(
    X: pd.DataFrame,
    y: pd.Series,
    features_to_plot: List[str]
) -> plt.Figure:
    """
    Plots violin plots for the top selected features to visualize 
    class separability and data distribution.

    Parameters
    ----------
    X : pd.DataFrame
        The scaled and selected feature matrix.
    y : pd.Series
        The target labels.
    features_to_plot : List[str]
        List of top column names to plot (recommend 5 to 10 max).
    """
    logger.info("Generating Feature Separability plot...")
    
    # Validate features
    valid_features = [f for f in features_to_plot if f in X.columns]
    if not valid_features:
        logger.error("None of the requested features are in the DataFrame.")
        return

    plot_df = X[valid_features].copy()
    plot_df['Target'] = y.values
    
    # Melt dataframe for seaborn facet grid plotting
    melted_df = pd.melt(plot_df, id_vars=['Target'], value_vars=valid_features, 
                        var_name='Feature', value_name='Standardized Value')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # density_norm replaces the deprecated 'scale' parameter in newer seaborn versions
    sns.violinplot(
        data=melted_df, x='Feature', y='Standardized Value', hue='Target',
        split=False, inner="quart", palette="muted", ax=ax, density_norm="count"
    )
    
    ax.set_title("Class Separability of Selected Acoustic Features", fontsize=16, pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig

if __name__ == '__main__':
    pass