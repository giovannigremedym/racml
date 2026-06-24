#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Audio Segmenter
========================
Handles the segmentation of acoustic signals using a sliding window approach.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-07
:Version: 1.0.3
"""
import numpy as np
from dataclasses import dataclass, fields
from typing import List, Tuple, Optional

@dataclass(frozen=True)
class SegConfig:
    """
    Configuration DTO for the Audio Segmentation process.

    Implements strict Separation of Concerns (Normalization vs Validation).

    Parameters
    ----------
    segment_duration : float, default=10.0
        The length of each segmented chunk in seconds. Must be strictly positive.
    overlap : float, default=0.0
        The proportion of overlap between consecutive segments [0.0, 1.0).
    sample_rate : int, default=24000
        Target sample rate for calculating index strides. Must be strictly positive.
    """
    segment_duration: float = 10.0
    overlap: float = 0.0
    sample_rate: int = 24000

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'SegConfig':
        """
        Instantiates the configuration by filtering only valid fields from kwargs.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments containing potential configuration parameters.

        Returns
        -------
        SegConfig
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
            object.__setattr__(self, 'segment_duration', float(self.segment_duration))
            object.__setattr__(self, 'overlap', float(self.overlap))
            object.__setattr__(self, 'sample_rate', int(self.sample_rate))
        except (ValueError, TypeError):
            pass 

    def _validate(self) -> None:
        """
        Validates the logical boundaries of the configuration attributes.

        Raises
        ------
        ValueError
            If `segment_duration` or `sample_rate` are not strictly positive,
            or if `overlap` is not within the [0.0, 1.0) interval.
        """
        if self.segment_duration <= 0:
            raise ValueError(f"segment_duration must be strictly positive. Got: {self.segment_duration}")
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError(f"overlap must be between 0.0 and strictly less than 1.0. Got: {self.overlap}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer. Got: {self.sample_rate}")


class AudioSegmenter:
    """
    Worker class dedicated strictly to chopping a 1D signal into smaller segments.

    Uses a sliding window approach with configurable overlap to chunk audio 
    data efficiently.

    Parameters
    ----------
    config : SegConfig, optional
        The immutable configuration object. If None, default parameters are used.
    **kwargs : dict
        Additional arguments to pass to the configuration factory if `config` is None.

    Attributes
    ----------
    config : SegConfig
        The configuration object governing the segmentation parameters.
    segment_samples : int
        The calculated number of samples per segment based on duration and sample rate.
    stride_samples : int
        The calculated number of samples to advance the sliding window based on overlap.
    """
    def __init__(self, config: Optional[SegConfig] = None, **kwargs):
        self.config = config if config is not None else SegConfig.from_kwargs(**kwargs)
        
        # Pre-calculating sample metrics in the constructor saves CPU cycles during loop
        self.segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        self.stride_samples = int(self.segment_samples * (1.0 - self.config.overlap))

    def process(self, y: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Slices the audio array into segments based on duration and overlap.

        Parameters
        ----------
        y : numpy.ndarray
            The 1D audio time-series array to be segmented.

        Returns
        -------
        List[Tuple[numpy.ndarray, float]]
            A list of tuples representing the segments. Each tuple contains:
            - chunk : numpy.ndarray
                The sliced audio segment.
            - offset_sec : float
                The start time of the chunk relative to the original file in seconds.
        """
        chunks = []
        if y.size < self.segment_samples:
            return chunks

        # O(N) Sliding Window complexity
        for start in range(0, len(y) - self.segment_samples + 1, self.stride_samples):
            end = start + self.segment_samples
            chunk = y[start:end]
            offset_seconds = start / self.config.sample_rate
            chunks.append((chunk, offset_seconds))

        return chunks

if __name__ == '__main__':
    pass