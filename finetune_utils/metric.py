import math

import torch
import librosa
import numpy as np

# Define the cost function for calculating DTW
def log_spec_dB_dist(x, y):
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))


# Calculate mcd 
def compute_mcd(ref_mel, pred_mel, n_mfcc=13, sr=22050):
    """
    Function to calculate MCD with DTW for handling mismatch length. ref and pred should be mel spectrogram with
    shape (n_mels, frames)
    """
    if isinstance(ref_mel, torch.Tensor):
        ref_mel = ref_mel.squeeze().cpu().numpy()
    if isinstance(pred_mel, torch.Tensor):
        pred_mel = pred_mel.squeeze().cpu().numpy()

    ref_mfcc = librosa.feature.mfcc(S=ref_mel, sr=sr, n_mfcc=n_mfcc)
    pred_mfcc = librosa.feature.mfcc(S=pred_mel, sr=sr, n_mfcc=n_mfcc)
    
    dtw_cost, dtw_min_path = librosa.sequence.dtw(
        ref_mfcc, pred_mfcc, metric=log_spec_dB_dist
    )
    # Sum up the costs over the path
    path_cost_matrix = dtw_cost[dtw_min_path[:, 0], dtw_min_path[:, 1]]
    path_cost = np.sum(path_cost_matrix)

    # Average over path length
    path_length = dtw_min_path.shape[0]
    reduced_dtw_cost = path_cost / path_length

    # Average over number of frames
    frames = pred_mel.shape[1]
    mcd = reduced_dtw_cost / frames

    return mcd
