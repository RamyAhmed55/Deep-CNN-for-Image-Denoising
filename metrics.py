import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """
    img1, img2: float numpy arrays in [0,1]
    """
    mse = np.mean((img1 - img2) ** 2, dtype=np.float64)
    if mse == 0:
        return 99.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """
    img1, img2: grayscale float numpy arrays in [0,1]
    """
    return ssim(img1, img2, data_range=data_range)


def summarize_metrics(psnr_list, ssim_list):
    avg_psnr = float(np.mean(psnr_list)) if len(psnr_list) > 0 else 0.0
    avg_ssim = float(np.mean(ssim_list)) if len(ssim_list) > 0 else 0.0
    return avg_psnr, avg_ssim