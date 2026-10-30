#!/usr/bin/env python3
"""
enhance_methods.py

Feature-preserving enhancement candidates for one grey RAW image.
All candidates work in float32 on the 0-255 scale and return float32,
so no re-quantization is added before the validation step.

Candidates:
  bm3d        BM3D denoising, noise sigma estimated from the image
  nlm         OpenCV non-local means, h = noise sigma
  bm3d_rl     BM3D, then mild Richardson-Lucy deconvolution (negative control: sharpening
              invents structure; every tested PSF/iteration setting failed validation)
  bm3d_x2     BM3D, then Lanczos x2 interpolation (adds pixels, not information)
"""

import bm3d
import cv2
import numpy as np
from skimage.restoration import estimate_sigma, richardson_lucy


def estimate_noise_sigma(gray):
    """Gaussian noise sigma (grey levels) from the wavelet median absolute deviation."""
    return float(estimate_sigma(gray.astype(np.float32)))


def denoise_bm3d(gray, sigma):
    """BM3D on the 0-1 scale with the estimated noise sigma."""
    out = bm3d.bm3d(gray.astype(np.float32) / 255.0, sigma_psd=sigma / 255.0)
    return np.clip(out * 255.0, 0, 255).astype(np.float32)


def denoise_nlm(gray, sigma):
    """OpenCV non-local means with filter strength h = sigma."""
    out = cv2.fastNlMeansDenoising(gray.astype(np.uint8), None, h=float(sigma),
                                   templateWindowSize=7, searchWindowSize=21)
    return out.astype(np.float32)


def gaussian_psf(psf_sigma, size=7):
    ax = np.arange(size) - size // 2
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * psf_sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def deconvolve_rl(image, psf_sigma, num_iter):
    """Richardson-Lucy deconvolution with a Gaussian PSF (light sharpening)."""
    out = richardson_lucy(np.clip(image, 1, 255) / 255.0, gaussian_psf(psf_sigma), num_iter=num_iter, clip=False)
    return np.clip(out * 255.0, 0, 255).astype(np.float32)


def upscale_lanczos(image, factor):
    h, w = image.shape
    return cv2.resize(image, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4)


def run_candidates(gray, rl_psf_sigma=0.7, rl_iter=2, upscale=2):
    """Return (sigma, {name: (image, description)})."""
    sigma = estimate_noise_sigma(gray)
    den_bm3d = denoise_bm3d(gray, sigma)
    candidates = {
        "bm3d": (den_bm3d, f"BM3D denoising (sigma={sigma:.2f})"),
        "nlm": (denoise_nlm(gray, sigma), f"Non-local means (h={sigma:.2f})"),
        "bm3d_rl": (deconvolve_rl(den_bm3d, rl_psf_sigma, rl_iter),
                    f"BM3D + Richardson-Lucy (Gaussian PSF sigma={rl_psf_sigma}, {rl_iter} iterations)"),
        f"bm3d_x{upscale}": (upscale_lanczos(den_bm3d, upscale), f"BM3D + Lanczos x{upscale}"),
    }
    return sigma, candidates
