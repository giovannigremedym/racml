#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Classifiers Factory
===========================
Provides a centralized, extensible factory machine learning classifiers. 
Enforces strict handling of class imbalance and computational resource limits.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-19
:Version: 1.1.0
"""

import os
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, fields
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils import get_standard_logger

cls_logger = get_standard_logger("ClassifierFactory")
opt_logger = get_standard_logger("HyperparameterTuner")

@dataclass(frozen=True)
class ClassifierConfig:
    """
    Configuration DTO for Classifier instantiation.

    Parameters
    ----------
    model_type : str, default='rf'
        The type of model to instantiate (e.g., 'lr', 'sgd', 'rf', 'xgb').
    random_state : int, default=42
        Seed for reproducibility across identical runs.
    n_jobs : int, default=-1
        Number of CPU cores to utilize. Will be scaled automatically.
    model_hyperparams : dict of str to Any, optional
        Additional hyper-parameters to pass directly to the model's constructor.
    """
    model_type: str = 'rf'
    random_state: int = 42
    n_jobs: int = -1
    model_hyperparams: Optional[Dict[str, Any]] = None

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'ClassifierConfig':
        """
        Instantiates the configuration safely from a pool of arbitrary kwargs.

        Parameters
        ----------
        **kwargs : Any
            Arbitrary keyword arguments.

        Returns
        -------
        ClassifierConfig
            An immutable instance of the classifier configuration.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        """
        Normalizes mutable defaults, scales CPU usage safely, and enforces 
        domain boundaries after initialization.
        """
        # 1. Normalize Hyperparameters
        if self.model_hyperparams is None:
            object.__setattr__(self, 'model_hyperparams', {})
            
        # 2. CPU Scaling Protection (Reserve 1 core for the OS minimum)
        max_cores = os.cpu_count() or 2
        if self.n_jobs == -1:
            reserved_cores = 2 if max_cores >= 16 else 1
            safe_cores = max(1, max_cores - reserved_cores)
        else:
            safe_cores = min(max(1, self.n_jobs), max_cores)
            
        cls_logger.info(f"CPU Scaling: Requested {self.n_jobs} -> Allocated {safe_cores} cores.")
        object.__setattr__(self, 'n_jobs', safe_cores)


class ClassifierFactory:
    """
    Registry for Machine Learning models. 
    Implements the Open/Closed Principle via Decorators.
    """
    _registry: Dict[str, Callable[[ClassifierConfig], BaseEstimator]] = {}

    @classmethod
    def register(cls, model_name: str) -> Callable:
        """
        Decorator to register a new model builder function.

        Parameters
        ----------
        model_name : str
            The string identifier for the model (e.g., 'rf').

        Returns
        -------
        Callable
            The decorator function.
        """
        def wrapper(builder_func: Callable[[ClassifierConfig], BaseEstimator]) -> Callable:
            cls._registry[model_name.lower()] = builder_func
            return builder_func
        return wrapper

    @classmethod
    def build(cls, config: Optional[ClassifierConfig] = None, **kwargs: Any) -> BaseEstimator:
        """
        Builds the requested classifier by routing to the registered builder.

        Parameters
        ----------
        config : ClassifierConfig, optional
            The configuration object. If None, it is built from kwargs.
        **kwargs : Any
            Keyword arguments to construct the configuration on-the-fly.

        Returns
        -------
        BaseEstimator
            An un-fitted, scikit-learn compatible model instance.

        Raises
        ------
        ValueError
            If the requested `model_type` is not found in the registry.
        """
        if config is None:
            config = ClassifierConfig.from_kwargs(**kwargs)
            
        m_type = config.model_type.lower()
        if m_type not in cls._registry:
            raise ValueError(
                f"Model type '{m_type}' is not registered. "
                f"Available models: {list(cls._registry.keys())}"
            )
            
        cls_logger.info(f"Building registered model architecture: {m_type.upper()}")
        return cls._registry[m_type](config)

def optimize_model(
    estimator: BaseEstimator, 
    param_grid: Dict[str, list],
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scoring_metric: str = 'f1_macro',
    n_jobs: int = -1
) -> BaseEstimator:
    """
    Executes Grid Search optimization using a strictly predefined validation set.

    Utilizes the `PredefinedSplit` strategy to ensure the model is tuned on the 
    validation set without cross-validation contamination, preserving the 
    distribution of the original holdout data.

    Parameters
    ----------
    estimator : BaseEstimator
        The unfitted Scikit-Learn compatible classifier instance.
    param_grid : dict
        Dictionary with parameters names (str) as keys and lists of settings to try.
    X_train : pd.DataFrame
        The training feature matrix (can be augmented/resampled).
    y_train : pd.Series
        Target labels for the training set.
    X_val : pd.DataFrame
        Explicit validation feature matrix (pristine real-world distribution).
    y_val : pd.Series
        Target labels for the validation set.
    scoring_metric : str, default='f1_macro'
        The evaluation metric to optimize. 
    n_jobs : int, default=-1
        Number of CPU cores to use. Automatically leaves headroom for the OS.

    Returns
    -------
    champion_model : BaseEstimator
        A cloned instance of the estimator, fitted exclusively on X_train using
        the optimal hyperparameters discovered.
    """
    if not isinstance(param_grid, dict) or not param_grid:
        opt_logger.warning("Empty param_grid provided. Returning base estimator fitted on training data.")
        base_model = clone(estimator)
        base_model.fit(X_train, y_train)
        return base_model

    # Resource Management: Prevents system lockup during heavy GridSearch
    max_cores = os.cpu_count() or 2
    if n_jobs == -1:
        reserved_cores = 2 if max_cores >= 16 else 1
        safe_cores = max(1, max_cores - reserved_cores)
    else:
        safe_cores = min(max(1, n_jobs), max_cores)

    opt_logger.info(f"Starting optimization | Metric: {scoring_metric} | Cores: {safe_cores}")
    
    # PredefinedSplit Workaround: Concatenate only to satisfy GridSearchCV API
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    
    # Strategy: -1 for training indices (ignored during eval), 0 for validation indices
    test_fold = np.concatenate([
        np.full(X_train.shape[0], -1), 
        np.zeros(X_val.shape[0])
    ])
    cv_strategy = PredefinedSplit(test_fold)
    
    # Injection of specific fit parameters (e.g., for XGBoost Early Stopping)
    fit_params = {}
    if type(estimator).__name__ == 'XGBClassifier':
        # Passes the explicit validation set to the tree builder
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['verbose'] = False

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv_strategy,
        n_jobs=safe_cores,
        verbose=1,
        refit=False  # Mandatory: Prevents fitting on X_combined. We handle refit manually.
    )
    
    search.fit(X_combined, y_combined, **fit_params)
    
    opt_logger.info(f"Optimization Finished | Best {scoring_metric}: {search.best_score_:.4f}")
    opt_logger.debug(f"Champion Hyperparameters: {search.best_params_}")
    
    # Manual Refit: Ensures NO validation data leaks into the final weights
    opt_logger.info("Refitting the champion model strictly on the Training Set...")
    champion_model = clone(estimator).set_params(**search.best_params_)
    champion_model.fit(X_train, y_train)
    
    return champion_model


# =============================================================================
# MODEL BUILDERS (Registered dynamically)
# =============================================================================

@ClassifierFactory.register('lr')
def _build_logistic_regression(config: ClassifierConfig) -> BaseEstimator:
    """Builds a Logistic Regression classifier."""
    params = {
        'random_state': config.random_state, 
        'max_iter': 2000,
        'class_weight': 'balanced'
    }
    params.update(config.model_hyperparams)
    return LogisticRegression(n_jobs=config.n_jobs, **params)


@ClassifierFactory.register('sgd')
def _build_sgd(config: ClassifierConfig) -> BaseEstimator:
    """Builds a Stochastic Gradient Descent classifier."""
    params = {
        'random_state': config.random_state,
        'loss': 'log_loss',
        'class_weight': 'balanced'
    }
    params.update(config.model_hyperparams)
    return SGDClassifier(n_jobs=config.n_jobs, **params)


@ClassifierFactory.register('rf')
def _build_random_forest(config: ClassifierConfig) -> BaseEstimator:
    """Builds a Random Forest classifier."""
    params = {
        'random_state': config.random_state,
        'class_weight': 'balanced'
    }
    params.update(config.model_hyperparams)
    return RandomForestClassifier(n_jobs=config.n_jobs, **params)


@ClassifierFactory.register('xgb')
def _build_xgboost(config: ClassifierConfig) -> BaseEstimator:
    """Builds an XGBoost classifier, handling dependencies safely."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost is not installed. Run: pip install xgboost")
        raise ImportError("Missing required dependency: xgboost")

    params = {
        'random_state': config.random_state,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss'
    }
    params.update(config.model_hyperparams)
    return xgb.XGBClassifier(n_jobs=config.n_jobs, **params)

if __name__ == '__main__':
    pass