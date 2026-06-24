#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Feature Selectors
=========================
Contains a streamlined functional pipeline for statistical feature reduction,
handling zero-variance, multicollinearity, and Random Forest Gini isolation.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-05-15
:Version: 3.1.0
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_standard_logger

logger = get_standard_logger("FeatureSelector")

def select_features(
    X: pd.DataFrame, 
    y: pd.Series, 
    target_features: Optional[List[str]] = None, 
    corr_threshold: float = 0.85,
    gini_threshold: Union[str, float] = '1.25*mean',
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Executes an end-to-end feature selection and standardization pipeline.
    
    Performs safe imputation, drops zero-variance features, removes highly 
    collinear metrics, and ultimately filters the remaining feature space 
    using Random Forest Gini Impurity.

    Parameters
    ----------
    X : pd.DataFrame
        The input feature matrix.
    y : pd.Series
        The target labels used for fitting the Random Forest.
    target_features : list of str, optional
        A specific list of feature names to isolate. If provided, the statistical 
        pipeline is bypassed and only these features are returned (scaled).
    corr_threshold : float, default=0.85
        Pearson correlation threshold. Features correlated above this are dropped.
    gini_threshold : str or float, default='1.25*mean'
        The threshold to use for feature importances. To aggressively reduce 
        the number of features, increase the multiplier (e.g., '1.5*mean').
    random_state : int, default=42
        Seed for reproducibility in the Random Forest.

    Returns
    -------
    X_selected : pd.DataFrame
        The transformed, scaled, and heavily filtered feature matrix.
    feature_importances : dict
        A dictionary mapping the surviving feature names to their Gini importance.
    """
    X_work = X.copy()
    
    # 1. Safe Imputation: Isolate valid columns, clean Infs, and impute medians
    X_work.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_work.dropna(axis=1, how='all', inplace=True)
    
    medians = X_work.median()
    X_work.fillna(medians, inplace=True)
    valid_cols = X_work.columns.tolist()

    # 2. Standardization (Done early as some linear checks benefit from it)
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_work)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=valid_cols, index=X_work.index)

    # 3. Target Feature Bypass (Baseline Isolation)
    if target_features is not None:
        missing = [f for f in target_features if f not in valid_cols]
        if missing:
            logger.warning(f"Target features dropped or missing: {missing}")
        selected_cols = [f for f in target_features if f in valid_cols]
        
        # Return dummy importances since statistical selection was bypassed
        return X_scaled[selected_cols], {col: 1.0 for col in selected_cols}

    # 4. Zero Variance Filter
    variances = X_scaled.var()
    # 1e-5 is used instead of strict 0.0 to account for floating point inaccuracies
    non_zero_var_cols = variances[variances > 1e-5].index.tolist()
    X_filtered = X_scaled[non_zero_var_cols]
    logger.debug(f"Zero Variance step dropped {len(valid_cols) - len(non_zero_var_cols)} features.")

    # 5. Multicollinearity Filter (Pearson)
    corr_matrix = X_filtered.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
    
    uncorrelated_cols = [c for c in non_zero_var_cols if c not in to_drop]
    X_uncorr = X_filtered[uncorrelated_cols]
    logger.debug(f"Collinearity step dropped {len(to_drop)} features.")

    # 6. Random Forest Gini Selection
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_state, 
        class_weight='balanced', 
        n_jobs=-1
    )
    rf.fit(X_uncorr, y)
    
    importances = rf.feature_importances_
    feature_gini_dict = dict(zip(uncorrelated_cols, importances))
    
    # Dynamic Threshold Parsing
    if isinstance(gini_threshold, str):
        if 'mean' in gini_threshold:
            multiplier = float(gini_threshold.split('*')[0]) if '*' in gini_threshold else 1.0
            thresh_val = multiplier * np.mean(importances)
        elif 'median' in gini_threshold:
            thresh_val = np.median(importances)
        else:
            thresh_val = np.mean(importances)
    else:
        thresh_val = float(gini_threshold)

    # Final feature survival arena
    selected_cols = [col for col, imp in feature_gini_dict.items() if imp >= thresh_val]
    
    # Assemble final payload
    X_final = X_uncorr[selected_cols]
    final_gini_dict = {col: feature_gini_dict[col] for col in selected_cols}
    
    logger.info(f"Pipeline complete. Selected {len(selected_cols)} features out of {X.shape[1]}. (Gini Thresh: {thresh_val:.4f})")

    return X_final, final_gini_dict

# Configure publication-quality plot settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def plot_correlation_matrix(
    X: pd.DataFrame, 
    title: Optional[str] = "Acoustic Features Collinearity Matrix"
) -> plt.Figure:
    """
    Plots a highly stylized correlation heatmap for the feature matrix.
    Useful for visualizing the dataset before dropping collinear features.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix (ideally right before the collinearity filter).
    title: str, optional
        The title of graphics
    """
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
    
    ax.set_title(title, pad=20, fontsize=16)
    
    plt.tight_layout()
    
    return fig

def plot_gini_importances(
    gini_dict: Dict[str, float], 
    top_n: int = 30,
    title: Optional[str] = "Random Forest Feature Selection"
) -> plt.Figure:
    """
    Plots a lollipop chart of Random Forest Gini importances.

    This visualization provides a higher data-to-ink ratio than standard
    bar charts, offering a condensed and minimal view of feature importances
    while avoiding visual fatigue.

    Parameters
    ----------
    gini_dict : Dict[str, float]
        Dictionary mapping feature names to their Gini importance.
    top_n : int, default=30
        Maximum number of top features to display to avoid clutter.
    title : str, optional
        The title of the graphic.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the generated plot.
    """    
    # Sort dictionary by importance (descending)
    sorted_features = sorted(gini_dict.items(), key=lambda item: item[1], reverse=True)
    plot_data = sorted_features[:top_n]
    
    names = [x[0] for x in plot_data]
    importances = [x[1] for x in plot_data]
    
    # Dynamically scale the figure height to stay condensed based on the number of features
    fig_height = max(6, min(10, 0.3 * len(names) + 2))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Normalize importances to map them to a colormap for visual hierarchy
    norm = plt.Normalize(min(importances), max(importances))
    cmap = plt.cm.viridis_r 
    colors = cmap(norm(importances))
    
    # Draw the stems (horizontal lines)
    ax.hlines(y=names, xmin=0, xmax=importances, color=colors, alpha=1.0, linewidth=3)
    
    # Draw the markers (dots)
    ax.scatter(importances, names, color=colors, s=50, alpha=1.0, zorder=3)
    
    ax.set_xlabel("Mean Decrease in Impurity (Gini)", fontsize=12)
    ax.set_ylabel("Acoustic Features", fontsize=12)
    ax.set_title(title, fontsize=16, pad=15)
    
    # Invert y-axis to have the most important feature at the top
    ax.invert_yaxis()
    
    # Minimalist grid and spine removal for a cleaner aesthetic
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=False, ax=ax)
    
    # Remove y-axis ticks since the names already align perfectly
    ax.tick_params(axis='y', length=0)
    
    plt.tight_layout()
    
    return fig

def plot_feature_separability(
    X: pd.DataFrame,
    y: pd.Series,
    features_to_plot: List[str],
    title: Optional[str] = "Class Separability of Selected Acoustic Features",
    palette: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Plots an integrated grouped boxplot for the selected features.

    Utilizes a symmetrical log scale (symlog) to prevent extreme outliers 
    from flattening the interquartile range representations. Ensures that
    categorical variables follow the logical order defined by the palette.

    Parameters
    ----------
    X : pd.DataFrame
        The scaled and selected feature matrix.
    y : pd.Series
        The target labels.
    features_to_plot : List[str]
        List of column names to plot.
    title : str, optional
        The overarching title of the graphic.
    palette : dict, optional
        Dictionary mapping target classes to specific HEX colors. The keys
        of this dictionary will dictate the plotting order of the classes.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the plot.
    """
    
    # Validate features against the dataframe
    valid_features = [f for f in features_to_plot if f in X.columns]
    if not valid_features:
        raise ValueError("No valid features found in the provided DataFrame.")

    plot_df = X[valid_features].copy()
    plot_df['Target'] = y.values
    
    # Melt dataframe for seaborn grouped categorical plotting
    melted_df = pd.melt(
        plot_df, 
        id_vars=['Target'], 
        value_vars=valid_features, 
        var_name='Feature', 
        value_name='Value'
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Custom properties for outliers (fliers)
    flier_props = {
        'marker': 'o',
        'markerfacecolor': 'red',
        'markeredgecolor': 'none',
        'markersize': 4,
        'alpha': 0.6
    }
    
    # Extract order from the palette dictionary to maintain logical progression
    class_order = list(palette.keys()) if palette else None
    
    sns.boxplot(
        data=melted_df, 
        x='Feature', 
        y='Value', 
        hue='Target',
        hue_order=class_order,
        palette=palette,
        flierprops=flier_props,
        ax=ax,
        linewidth=1.2
    )
    
    # Apply symmetrical log scale to handle massive outliers gracefully
    ax.set_yscale('symlog')
    
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_ylabel("Value (Symlog Scale)", fontsize=12)
    ax.set_xlabel("")
    
    # Rotate X labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Move legend outside the plot area to avoid covering data
    ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Clean up aesthetics
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    sns.despine(ax=ax, bottom=True)
    
    plt.tight_layout()
    
    return fig


if __name__ == '__main__':
    pass