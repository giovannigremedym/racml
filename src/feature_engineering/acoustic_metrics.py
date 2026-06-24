#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Acoustic Metrics
=======================
Exhaustive acoustic metric extraction for one sample.
Uses a specialized scalar extractor to bypass all NumPy ambiguity errors,
now powered by a dynamic, cached execution pipeline.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-05-14
:Version: 5.0.0
"""
from typing import Optional, Dict, Any, Callable, Tuple, List
import numpy as np
from scipy import signal
import scipy.stats as stats
import librosa
from librosa import feature as libfeat
from librosa import power_to_db
from maad import sound as maddsd 
from maad import features as maadfeat
from pywt import wavedec as pywtwavedec
from dataclasses import dataclass, fields

from src.utils import get_standard_logger


def metric_group(group_name: str) -> Callable:
    """
    Decorator that tags a class method as belonging to a specific metric group.

    This tag is later read during class instantiation to build a cached 
    execution pipeline, avoiding dynamic lookups during data processing.

    Parameters
    ----------
    group_name : str
        The unique identifier for the metric group (e.g., "temporal", "spectral").

    Returns
    -------
    Callable
        The decorated function with an injected `__metric_group__` attribute.
    """
    def decorator(func: Callable) -> Callable:
        func.__metric_group__ = group_name
        return func
    return decorator


@dataclass(frozen=True)
class AcMConfig:
    """
    Configuration DTO for the Acoustic Metrics Extraction process.

    Parameters
    ----------
    sample_rate : int, default=24000
        Target sample rate for the audio analysis. Must be strictly positive.
    fft_window_size : int, default=1024
        Size of the Fast Fourier Transform (FFT) window. Must be strictly positive.
    metric_groups : Tuple[str, ...], default=None
        Defines which metric groups to compute.
    """
    sample_rate: int = 24000
    fft_window_size: int = 1024
    metric_groups: Tuple[str, ...] = None

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'AcMConfig':
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        self._normalize()
        self._validate()
    
    def _normalize(self) -> None:
        try:
            if not isinstance(self.sample_rate, int):
                object.__setattr__(self, 'sample_rate', int(self.sample_rate))
            if not isinstance(self.fft_window_size, int):
                object.__setattr__(self, 'fft_window_size', int(self.fft_window_size))
            if self.metric_groups is not None:
                if isinstance(self.metric_groups, str):
                    object.__setattr__(self, 'metric_groups', (self.metric_groups.lower(),))
                else:
                    normalized_groups = tuple(str(g).lower() for g in self.metric_groups)
                    object.__setattr__(self, 'metric_groups', normalized_groups)
        except (ValueError, TypeError):
            pass

    def _validate(self) -> None:
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer. Got: {self.sample_rate}")
        
        if not isinstance(self.fft_window_size, int) or self.fft_window_size <= 0:
            raise ValueError(f"fft_window_size must be a positive integer. Got: {self.fft_window_size}")


class AcousticMetrics:
    """
    Worker class dedicated to the exhaustive extraction of acoustic metrics.
    """
    def __init__(self, config: Optional[AcMConfig] = None, **kwargs):
        if config is not None:
            if not isinstance(config, AcMConfig):
                raise TypeError(f"Config must be AcMConfig. Got: {type(config)}")
            else:
                self._metrics_config = config
        else:
            self._metrics_config = AcMConfig.from_kwargs(**kwargs)
        
        self.logger = get_standard_logger(self.__class__.__name__)
        
        # Pipeline cache: Prevents running dir(self) on every audio file.
        self._execution_pipeline: List[Callable] = []
        self._build_pipeline()

    @property
    def config(self) -> AcMConfig:
        return self._metrics_config

    def _build_pipeline(self) -> None:
        """
        Scans the class for decorated methods and registers them in the execution 
        pipeline if their group is active in the configuration.
        """
        available_groups = set()
        
        for attr_name in dir(self):
            method = getattr(self, attr_name)
            if callable(method) and hasattr(method, "__metric_group__"):
                group_name = getattr(method, "__metric_group__")
                available_groups.add(group_name)
                
                if self.config.metric_groups is None or group_name in self.config.metric_groups:
                    self._execution_pipeline.append(method)
                    self.logger.debug(f"Registered metric group: '{group_name}'")

        if self.config.metric_groups is not None:
            missing_groups = set(self.config.metric_groups) - available_groups
            if missing_groups:
                self.logger.warning(
                    f"Requested unknown metric groups: {missing_groups}. "
                    f"Available groups are: {available_groups}"
                )

    def calculate(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extracts selected acoustic metrics from a given audio signal.
        """
        res: Dict[str, float] = {}
        if y.size == 0:
            self.logger.warning("Empty audio array received.")
            return res
        
        # O(N) over the pre-cached pipeline. Zero reflection cost here.
        for extractor_func in self._execution_pipeline:
            try:
                group_results = extractor_func(y)
                res.update(group_results)
            except Exception as e:
                group_name = getattr(extractor_func, "__metric_group__", "unknown")
                self.logger.error(f"Failed to compute group '{group_name}': {e}")
                
        return res
    
    def _get_scalar(self, val: Any) -> float:
        """
        THE TRUTH EXTRACTOR: Guarantees a float return without triggering 
        any 'ambiguous array' boolean checks.
        """
        try:
            if isinstance(val, np.ndarray):
                if val.size == 0: 
                    return np.nan
            
                if val.dtype == bool:
                    if np.mean(val) > 5.0:
                        return 1.0
                    else:
                        return 0.0
            
                return float(np.nanmean(val))
            else:
                number = float(val)
            
                if np.isinf(number) or np.isnan(number):
                    return np.nan
                
                return number
        except:
            return np.nan
        
    # ==========================================
    # METRIC EXTRACTION GROUPS
    # ==========================================

    @metric_group("alpha")
    def _compute_alpha(self, y: np.ndarray) -> Dict[str, float]:
        res: Dict[str, float] = {}
        expected_metrics = [
            'EAS', 'ECU', 'ECV', 'num_peaks', 'NDSI', 'ratioBA', 
            'anthro_energy', 'bio_energy', 'BI', 'ADI', 'AEI', 
            'roughness', 'tfsd'
        ]

        sr = self.config.sample_rate
        n_fft = self.config.fft_window_size
        hop = n_fft // 2

        try:
            Sxx_power, tn, fn, _ = maddsd.spectrogram(y, sr, nperseg=n_fft, noverlap=hop, mode='psd')
            Sxx_power = np.maximum(Sxx_power, 1e-12)
            
            dt = tn[1] - tn[0] if len(tn) > 1 else 1.0
            psd_avg = np.mean(Sxx_power, axis=1)
        except Exception as e:
            self.logger.error(f"Spectrogram baseline failed for alpha indices: {e}")
            return res
            
        try:
            spectral_entropy = maadfeat.spectral_entropy(Sxx_power, fn)
            res['EAS'] = self._get_scalar(spectral_entropy[0])
            res['ECU'] = self._get_scalar(spectral_entropy[1])
            res['ECV'] = self._get_scalar(spectral_entropy[2])

            res['num_peaks'] = self._get_scalar(maadfeat.number_of_peaks(psd_avg, fn, mbins=None, threshold=None))

            ndsi_res = maadfeat.soundscape_index(Sxx_power, fn)
            res['NDSI'] = self._get_scalar(ndsi_res[0])
            res['ratioBA'] = self._get_scalar(ndsi_res[1])
            res['anthro_energy'] = self._get_scalar(ndsi_res[2])
            res['bio_energy'] = self._get_scalar(ndsi_res[3])

            res['BI'] = self._get_scalar(maadfeat.bioacoustics_index(Sxx_power, fn))
            res['ADI'] = self._get_scalar(maadfeat.acoustic_diversity_index(Sxx_power, fn))
            res['AEI'] = self._get_scalar(maadfeat.acoustic_eveness_index(Sxx_power, fn))
            
            res['roughness'] = self._get_scalar(maadfeat.surface_roughness(Sxx_power, fn))
            res['tfsd'] = self._get_scalar(maadfeat.tfsd(Sxx_power, fn, tn))
        except Exception:
            pass

        for metric in expected_metrics:
            if metric not in res:
                res[metric] = np.nan

        return res
            
    @metric_group("temporal")
    def _compute_temporal(self, y: np.ndarray) -> Dict[str, float]:
        res: Dict[str, float] = {}
        expected_metrics = [
            'temp_entropy', 'ACI', 'AGI', 'temp_skew', 'temp_kurtosis', 
            'rms_mean', 'rms_std', 'zcr_mean', 'zcr_std', 'mae'
        ]

        sr = self.config.sample_rate
        n_fft = self.config.fft_window_size
        hop = n_fft // 2

        try:
            Sxx_power, tn, fn, _ = maddsd.spectrogram(y, sr, nperseg=n_fft, noverlap=hop, mode='psd')
            Sxx_power = np.maximum(Sxx_power, 1e-12)
            Sxx_mag = np.sqrt(Sxx_power)
            dt = tn[1] - tn[0] if len(tn) > 1 else 1.0
        except Exception as e:
            self.logger.error(f"Spectrogram baseline failed for temporal indices: {e}")
            return res
        
        try:
            res['temp_entropy'] = self._get_scalar(maadfeat.temporal_entropy(y))
            res['ACI'] = self._get_scalar(maadfeat.acoustic_complexity_index(Sxx_mag)[2])
            res['AGI'] = self._get_scalar(maadfeat.acoustic_gradient_index(Sxx_power, dt)[3])

            t_moments = maadfeat.temporal_moments(y)
            res['temp_skew'] = self._get_scalar(t_moments[2])
            res['temp_kurtosis'] = self._get_scalar(t_moments[3])

            rms = libfeat.rms(y=y, frame_length=n_fft, hop_length=hop)
            res['rms_mean'] = self._get_scalar(np.mean(rms))
            res['rms_std'] = self._get_scalar(np.std(rms))
    
            zcr = libfeat.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
            res['zcr_mean'] = self._get_scalar(np.mean(zcr))
            res['zcr_std'] = self._get_scalar(np.std(zcr))
        
            res['mae'] = self._get_scalar(np.mean(np.abs(y)))
        except Exception:
            pass

        for metric in expected_metrics:
            if metric not in res:
                res[metric] = np.nan

        return res

    @metric_group("spectral")
    def _compute_spectral(self, y: np.ndarray) -> Dict[str, float]:
        res: Dict[str, float] = {}
        expected_metrics = [
            'psd_mean', 'psd_std', 'spec_slope', 'spec_mean', 'spec_std', 
            'spec_skew', 'spec_kurtosis','spec_centroid', 'spec_flatness', 
            'spec_rolloff', 'spec_bandwidth_mean', 'spec_bandwidth_50%', 'spec_bandwidth_90%'
        ]

        sr = self.config.sample_rate
        n_fft = self.config.fft_window_size
        hop = n_fft // 2

        try:
            Sxx_power, _, fn, _ = maddsd.spectrogram(y, sr, nperseg=n_fft, noverlap=hop, mode='psd')
            Sxx_power = np.maximum(Sxx_power, 1e-12)
            Sxx_mag = np.sqrt(Sxx_power)
            psd_avg = np.mean(Sxx_power, axis=1)
        except Exception as e:
            self.logger.error(f"Spectrogram baseline failed for spectral indices: {e}")
            return res
        
        try:
            res['psd_mean'] = self._get_scalar(np.mean(psd_avg))
            res['psd_std'] = self._get_scalar(np.std(psd_avg))

            res['spec_slope'] = self._get_scalar(stats.linregress(fn, psd_avg)[0])

            s_moments = maadfeat.spectral_moments(psd_avg)
            res['spec_mean'] = self._get_scalar(s_moments[0])
            res['spec_std'] = self._get_scalar(s_moments[1])
            res['spec_skew'] = self._get_scalar(s_moments[2])
            res['spec_kurtosis'] = self._get_scalar(s_moments[3])

            res['spec_centroid'] = self._get_scalar(np.mean(libfeat.spectral_centroid(S=Sxx_mag, sr=sr)))
            res['spec_flatness'] = self._get_scalar(np.mean(libfeat.spectral_flatness(S=Sxx_mag)))
            res['spec_rolloff'] = self._get_scalar(np.mean(libfeat.spectral_rolloff(S=Sxx_mag, sr=sr)))
            
            bw_50, bw_90 = maadfeat.spectral_bandwidth(y, sr, nperseg=n_fft)
            res['spec_bandwidth_50%'] = self._get_scalar(bw_50)
            res['spec_bandwidth_90%'] = self._get_scalar(bw_90)

            spec_bdw = libfeat.spectral_bandwidth(S=Sxx_mag, sr=sr)
            res['spec_bandwidth_mean'] = self._get_scalar(np.mean(spec_bdw))
        except Exception:
            pass
        
        for metric in expected_metrics:
            if metric not in res:
                res[metric] = np.nan
        
        return res

    @metric_group("mfcc")
    def _compute_mfcc(self, y: np.ndarray) -> Dict[str, float]:
        res: Dict[str, float] = {}
        expected_metrics = ['mfcc_mean', 'mfcc_std']
        for i in range(1, 21):
            expected_metrics.extend([f'mfcc_{i}', f'mfcc_delta_{i}'])

        sr = self.config.sample_rate
        n_fft = self.config.fft_window_size
        hop = n_fft // 2

        try:
            # Librosa pipeline for MFCCs
            mel = libfeat.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
            mfccs = libfeat.mfcc(S=power_to_db(mel), n_mfcc=20)
            mfccs_delta = libfeat.delta(mfccs)
            
            mfcc_m = np.mean(mfccs, axis=1)
            mfcc_d_m = np.mean(mfccs_delta, axis=1)
            
            for i in range(20): 
                res[f'mfcc_{i+1}'] = self._get_scalar(mfcc_m[i])
                res[f'mfcc_delta_{i+1}'] = self._get_scalar(mfcc_d_m[i])

            res['mfcc_mean'] = self._get_scalar(np.mean(mfcc_m))
            res['mfcc_std'] = self._get_scalar(np.std(mfcc_m))
        except Exception:
            pass

        for metric in expected_metrics:
            if metric not in res:
                res[metric] = np.nan

        return res

    @metric_group("wavelet")
    def _compute_wavelet(self, y: np.ndarray) -> Dict[str, float]:
        res: Dict[str, float] = {}
        expected_metrics = ['wav_energy_mean', 'wav_approx_energy', 'wav_approx_std']
        for lvl in range(1, 6):
            expected_metrics.extend([f'wav_detail_lvl{lvl}_energy', f'wav_detail_lvl{lvl}_std'])

        try: 
            coeffs = pywtwavedec(y, 'db4', level=5)
            
            ca5 = coeffs[0]
            ca5_energy = np.sum(ca5**2) / len(ca5)
            res['wav_approx_energy'] = self._get_scalar(ca5_energy)
            res['wav_approx_std'] = self._get_scalar(np.std(ca5))

            details = coeffs[1:]
            wav_energys = [ca5_energy]

            for i, detail in enumerate(details):
                level = 5 - i
                detail_energy = np.sum(detail**2) / len(detail)
                wav_energys.append(detail_energy)

                res[f'wav_detail_lvl{level}_energy'] = self._get_scalar(detail_energy)
                res[f'wav_detail_lvl{level}_std'] = self._get_scalar(np.std(detail))

            res['wav_energy_mean'] = self._get_scalar(np.mean(wav_energys))
        except Exception:
            pass

        for metric in expected_metrics:
            if metric not in res:
                res[metric] = np.nan

        return res


if __name__ == '__main__':
    pass