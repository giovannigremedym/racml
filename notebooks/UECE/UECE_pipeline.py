import sys
import gc
from pathlib import Path
from typing import Dict, Any, List

# ====================================================================
# 1. PATH INJECTION FOR WORKERS
# Child processes must discover the project root independently.
# ====================================================================
current_path = Path.cwd()
for p in [current_path, current_path.parent, current_path.parent.parent]:
    if (p / "rainfall_acoustic_classification").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

# ====================================================================
# 2. LIBRARY IMPORTS AND CONFIGURATION
# ====================================================================
from rainfall_acoustic_classification.core import load_audio_sample
from rainfall_acoustic_classification.processing import (
    AudioAugmenter, AudioSegmenter, AcousticMetrics
)

augmenter = AudioAugmenter(sample_rate=24000, noise_prob=0.2, cutout_prob=0.3, random_state=42)
segmenter = AudioSegmenter(sample_rate=24000,segment_duration=10.0, overlap=0.5)
metrics_ext = AcousticMetrics(sample_rate=24000, fft_window_size=1024, metric_groups=["alpha", "temporal", "spectral", "mfcc", "wavelet"])

def audio_processing_worker(row_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Worker function to execute signal processing and feature extraction 
    for a single audio file.

    This function loads the audio, applies stochastic augmentation (if flagged),
    segments the signal, and extracts acoustic metrics for each segment.

    Parameters
    ----------
    row_dict : dict
        A dictionary representing a single row from the metadata DataFrame. Must contain
        at least the 'file_path', 'split', and 'should_augment' keys.

    Returns
    -------
    list of dict
        A list of dictionaries, where each dictionary contains the extracted features
        and metadata for a specific segment of the original audio file. Returns an
        empty list if the audio loading fails.
    """
    sample = load_audio_sample(row_dict.get('file_path'), sample_rate=24000)
    if not sample: 
        return []

    y = sample.audio_data
    aug_log = "raw"

    # Apply stochastic augmentation strictly if the flag is True
    if row_dict.get('split') == 'train' and row_dict.get('should_augment', False):
        y, aug_log = augmenter.process(y)

    results = []
    chunks = segmenter.process(y)

    for idx, (chunk_array, offset) in enumerate(chunks):
        features = metrics_ext.calculate(chunk_array)

        segment_data = row_dict.copy()
        segment_data.update({
            'segment_idx': idx, 
            'offset_sec': offset, 
            'aug_params': aug_log
        })
        segment_data.update(features)
        results.append(segment_data)

    # =======================================================
    # AGGRESSIVE GARBAGE COLLECTION
    # =======================================================
    del y
    del chunks
    del sample
    gc.collect() 

    return results
