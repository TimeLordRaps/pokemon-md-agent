"""Perceptual hashing utilities for sprite comparison in VISION-GRID container.

This module provides deterministic perceptual hashing for sprites using fixed-size
grayscale downsampling and DCT-based hashing, ensuring consistent hashes regardless
of input image dimensions.
"""

import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Fixed hash size for deterministic behavior
PHASH_SIZE = 32  # 32x32 grayscale downsample
DCT_SIZE = 8     # 8x8 low-frequency DCT components


def compute_phash(image: np.ndarray) -> np.ndarray:
    """Compute deterministic perceptual hash for sprite comparison.

    Uses fixed-size (32x32) grayscale downsampling and DCT to create a binary
    hash that's consistent regardless of input image dimensions.

    Args:
        image: Input image as numpy array (H, W, C) or (H, W)

    Returns:
        Binary hash array of shape (64,) representing 8x8 DCT components

    Raises:
        ValueError: If image is invalid or cannot be processed
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if image.size == 0:
        raise ValueError("Input image is empty")

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # Convert RGB/RGBA to grayscale using luminance weights
        if image.shape[2] == 4:  # RGBA
            image = image[..., :3]  # Remove alpha channel
        if image.shape[2] == 3:  # RGB
            # Use standard luminance conversion: 0.299*R + 0.587*G + 0.114*B
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        else:
            # Single channel, treat as grayscale
            gray = image[..., 0]
    elif len(image.shape) == 2:
        gray = image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Ensure float type for DCT
    gray = gray.astype(np.float32)

    # Resize to fixed 32x32 for deterministic behavior
    from scipy.ndimage import zoom
    zoom_factors = (PHASH_SIZE / gray.shape[0], PHASH_SIZE / gray.shape[1])
    resized = zoom(gray, zoom_factors, order=1)  # Linear interpolation

    # Apply 2D DCT
    dct_result = _dct2d(resized)

    # Extract low-frequency 8x8 components (top-left corner)
    low_freq = dct_result[:DCT_SIZE, :DCT_SIZE]

    # Calculate median as threshold (more robust than mean for DCT)
    median_val = np.median(low_freq)

    # Create binary hash: 1 if above median, 0 if below
    binary_hash = (low_freq > median_val).astype(np.uint8)

    # Flatten to 1D array
    hash_array = binary_hash.flatten()

    logger.debug(f"Computed pHash with {hash_array.sum()} bits set out of {len(hash_array)}")
    return hash_array


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Calculate Hamming distance between two binary hashes.

    Args:
        a: First hash array
        b: Second hash array

    Returns:
        Hamming distance (number of differing bits)

    Raises:
        ValueError: If hash arrays have different shapes or dtypes
    """
    if a.shape != b.shape:
        raise ValueError(f"Hash shapes must match: {a.shape} vs {b.shape}")

    if a.dtype != b.dtype:
        raise ValueError(f"Hash dtypes must match: {a.dtype} vs {b.dtype}")

    # XOR and count bits
    xor_result = np.bitwise_xor(a, b)
    distance = np.sum(xor_result)

    return int(distance)


def is_near_duplicate(a: np.ndarray, b: np.ndarray, threshold: int = 8) -> bool:
    """Check if two binary hashes are near duplicates within Hamming distance threshold.

    Args:
        a: First hash array
        b: Second hash array
        threshold: Maximum Hamming distance for near-duplicate认定 (default: 8)

    Returns:
        True if hashes are near duplicates (distance <= threshold), False otherwise

    Raises:
        ValueError: If hash arrays have different shapes or dtypes
    """
    # Validate dtypes
    if a.dtype != b.dtype:
        raise ValueError(f"Hash dtypes must match: {a.dtype} vs {b.dtype}")

    # Validate shapes
    if a.shape != b.shape:
        raise ValueError(f"Hash shapes must match: {a.shape} vs {b.shape}")

    # Calculate distance and check threshold
    distance = hamming_distance(a, b)
    return distance <= threshold


def _dct2d(image: np.ndarray) -> np.ndarray:
    """Compute 2D Discrete Cosine Transform using scipy.

    Args:
        image: 2D numpy array

    Returns:
        2D DCT result
    """
    from scipy.fft import dct
    # Apply DCT row-wise then column-wise using scipy
    dct_rows = dct(image, axis=1)
    dct_full = dct(dct_rows, axis=0)
    return dct_full


def _dct1d(signal: np.ndarray) -> np.ndarray:
    """Compute 1D Discrete Cosine Transform using scipy.

    Args:
        signal: 1D numpy array

    Returns:
        1D DCT result
    """
    from scipy.fft import dct
    return dct(signal)


def _dct1d(signal: np.ndarray) -> np.ndarray:
    """Compute 1D Discrete Cosine Transform using numpy.

    Args:
        signal: 1D numpy array

    Returns:
        1D DCT result
    """
    N = len(signal)
    result = np.zeros(N, dtype=np.float32)

    for k in range(N):
        sum_val = 0.0
        for n in range(N):
            sum_val += signal[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        # Apply DCT-II normalization
        if k == 0:
            result[k] = sum_val * np.sqrt(1.0 / N)
        else:
            result[k] = sum_val * np.sqrt(2.0 / N)

    return result