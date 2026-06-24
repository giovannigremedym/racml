#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Audio Augmenter
========================
Handles stochastic mathematical transformations on audio signals 
to perform data augmentation.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-07
:Version: 1.0.0
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class AugConfig:
    """
    Configuration DTO for the Audio Augmentation process.
    Implements strict Separation of Concerns (Normalization vs Validation).

    Parameters
    ----------
    sample_rate : int
        Target sample rate for analysis. Must be > 0. Default: 24000.
    random_state : int
        Seed for the random number generator to ensure reproducibility. Default: 42.
    noise_prob : float
        Probability [0.0, 1.0] of applying additive Gaussian noise. Default: 0.2.
    cutout_prob : float
        Probability [0.0, 1.0] of applying random cutout (silencing a segment). Default: 0.3.
    """
    sample_rate: int = 24000
    random_state: int = 42
    noise_prob: float = 0.2
    cutout_prob: float = 0.3

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'AugConfig':
        """
        Instantiates the configuration by filtering only valid fields from kwargs.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments containing potential configuration parameters.

        Returns
        -------
        AugConfig
            An immutable instance of the configuration object.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        """
        Post-initialization hook to trigger data normalization and validation.
        """
        self._normalize()
        self._validate()

    def _normalize(self) -> None:
        """
        Normalizes internal attributes by casting them to their expected types.
        Fails silently to allow the `_validate` method to catch persistent errors.
        """
        try:
            object.__setattr__(self, 'sample_rate', int(self.sample_rate))
            object.__setattr__(self, 'random_state', int(self.random_state))
            object.__setattr__(self, 'noise_prob', float(self.noise_prob))
            object.__setattr__(self, 'cutout_prob', float(self.cutout_prob))
        except (ValueError, TypeError):
            pass

    def _validate(self) -> None:
        """
        Validates the logical boundaries of the configuration attributes.

        Raises
        ------
        ValueError
            If `sample_rate` is not strictly positive, or if probability bounds 
            (`noise_prob`, `cutout_prob`) are not within the [0.0, 1.0] interval.
        """
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer. Got: {self.sample_rate}")
        if not (0.0 <= self.noise_prob <= 1.0):
            raise ValueError(f"noise_prob must be between 0.0 and 1.0. Got: {self.noise_prob}")
        if not (0.0 <= self.cutout_prob <= 1.0):
            raise ValueError(f"cutout_prob must be between 0.0 and 1.0. Got: {self.cutout_prob}")


class AudioAugmenter:
    """
    Worker class dedicated strictly to audio signal augmentation.

    Applies stochastic mathematical transformations (time shift, gain perturbation,
    Gaussian noise, and cutout) to a 1D numpy array.

    Parameters
    ----------
    config : AugConfig, optional
        The immutable configuration object. If None, default parameters are used.
    **kwargs : dict
        Additional arguments to pass to the configuration factory if `config` is None.

    Attributes
    ----------
    config : AugConfig
        The configuration object governing the augmentation probabilities and parameters.
    rng : numpy.random.Generator
        The initialized random number generator for reproducible stochastic operations.
    """
    def __init__(self, config: Optional[AugConfig] = None, **kwargs):
        self.config = config if config is not None else AugConfig.from_kwargs(**kwargs)
        self.rng = np.random.default_rng(seed=self.config.random_state)

    def process(self, y: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Applies random distortions to the signal based on configuration probabilities.

        Parameters
        ----------
        y : numpy.ndarray
            The 1D audio time-series array representing the acoustic signal.

        Returns
        -------
        Tuple[numpy.ndarray, str]
            A tuple containing:
            - The augmented audio array (dtype=np.float32).
            - A string logging the applied operations (e.g., 'shift=1200|noise=0.015').
              Returns "empty" if the input array is empty.
        """
        if y.size == 0:
            return y, "empty"

        logs = []
        y_mod = y.copy()
        
        # 1. Time Shift (Circular Scrolling) - Always applied for variation
        shift_amt = self.rng.integers(int(len(y) * 0.1), int(len(y) * 0.9))
        y_mod = np.roll(y_mod, shift_amt)
        logs.append(f"shift={shift_amt}")
        
        # 2. Gain Perturbation - Always applied
        gain = self.rng.uniform(0.7, 1.2)
        y_mod = y_mod * gain
        logs.append(f"gain={gain:.2f}")
        
        # 3. Additive Gaussian Noise - Probabilistic
        if self.rng.random() < self.config.noise_prob:
            max_amp = np.amax(np.abs(y_mod)) if len(y_mod) > 0 else 1.0
            noise_factor = self.rng.uniform(0.001, 0.02) 
            noise_amp = noise_factor * max_amp
            noise = self.rng.normal(scale=noise_amp, size=len(y_mod))
            y_mod = y_mod + noise
            logs.append(f"noise={noise_factor:.4f}")
        
        # 4. Random Cutout - Probabilistic
        if self.rng.random() < self.config.cutout_prob:
            zero_len = int(0.1 * self.config.sample_rate)
            if len(y_mod) > zero_len:
                zero_start = self.rng.integers(0, len(y_mod) - zero_len)
                y_mod[zero_start : zero_start + zero_len] = 0.0
                logs.append("cutout")

        return y_mod.astype(np.float32), "|".join(logs)


if __name__ == "__main__":
    pass