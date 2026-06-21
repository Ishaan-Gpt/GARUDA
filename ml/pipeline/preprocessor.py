"""
GARUDA ML Pipeline — Image Preprocessor
========================================
Handles all image quality challenges in Indian traffic surveillance:
  - Low light (night, tunnel)
  - Rain and fog (Bangalore monsoon)
  - Motion blur (fast vehicles)
  - Sensor noise
  - Overexposure (direct sunlight)

Uses CLAHE, Wiener-deblur, and adaptive gamma correction.
Augmentation pipeline uses Albumentations for training data robustness.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Production-grade preprocessing for traffic surveillance images.

    For INFERENCE: enhance_low_light → reduce_noise → normalize_exposure
    For TRAINING: full augmentation pipeline with rain/fog/blur/noise simulation
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self._augment = None  # Lazy-load albumentations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray, enhance: bool = True, is_video: bool = False) -> np.ndarray:
        """
        Full preprocessing pipeline for inference.
        Optimized to downscale before CLAHE/Denoise to save CPU cycles,
        and bypass denoising completely for video or during daylight.

        Args:
            image: BGR image (OpenCV format)
            enhance: Apply quality enhancement (CLAHE + noise reduction).
                     Set False for speed-critical live video paths.
            is_video: Skip denoising completely if true.

        Returns:
            Preprocessed BGR image (same size as input)
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image provided to preprocessor")

        orig_h, orig_w = image.shape[:2]

        # Calculate average brightness to detect daylight
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        # Only denoise if: enhance is True, NOT video, and brightness is extremely low (< 60)
        should_denoise = enhance and not is_video and (mean_brightness < 60)

        # Downscale for enhancement if target size exceeded
        downscaled = image
        scale_factor = 1.0
        max_dim = 640
        if enhance and (orig_w > max_dim or orig_h > max_dim):
            scale_factor = max_dim / max(orig_w, orig_h)
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Apply enhancements on the downscaled image
        if enhance:
            downscaled = self._enhance_low_light(downscaled)
            if should_denoise:
                downscaled = self._reduce_noise(downscaled)

        downscaled = self._normalize_exposure(downscaled)

        # Upscale back to original resolution if we downscaled
        if scale_factor != 1.0:
            processed = cv2.resize(downscaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            processed = downscaled

        return processed

    def resize_for_model(self, image: np.ndarray,
                          size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Resize with letterboxing to preserve aspect ratio"""
        target = size or self.target_size
        h, w = image.shape[:2]
        th, tw = target

        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target
        pad_w = tw - new_w
        pad_h = th - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))

    def augment_for_training(self, image: np.ndarray,
                              bboxes: Optional[List] = None,
                              labels: Optional[List] = None) -> Dict:
        """
        Apply Albumentations augmentation for training data.
        Simulates Bangalore-specific conditions: monsoon rain, fog, motion blur.

        Returns dict with 'image' (and optionally 'bboxes', 'category_ids')
        """
        pipeline = self._get_augmentation_pipeline()

        if bboxes and labels:
            result = pipeline(image=image, bboxes=bboxes, category_ids=labels)
        else:
            result = pipeline(image=image)

        return result

    # ------------------------------------------------------------------
    # Enhancement Methods
    # ------------------------------------------------------------------

    def _enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel.
        Enhances local contrast without over-brightening already-lit regions.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Non-local means denoising — effective for CMOS sensor noise.
        Note: ~15ms per 640x480 frame on CPU. Skip for live video > 20fps.
        """
        return cv2.fastNlMeansDenoisingColored(image, None, h=5, hColor=5,
                                               templateWindowSize=7,
                                               searchWindowSize=21)

    def _normalize_exposure(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive gamma correction based on mean brightness.
        Dark images (night/tunnel): gamma boost.
        Overexposed images (noon sun): gamma reduction.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        if mean_brightness < 60:      # Very dark — heavy boost
            gamma = 0.4
        elif mean_brightness < 100:   # Dark — moderate boost
            gamma = 0.65
        elif mean_brightness > 210:   # Overexposed — reduce
            gamma = 1.6
        else:
            return image  # Normal exposure — no change

        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8
        )
        return cv2.LUT(image, table)

    def deblur_motion(self, image: np.ndarray,
                       kernel_size: int = 15,
                       angle: float = 0.0) -> np.ndarray:
        """
        Directional motion deblur using unsharp masking.
        Use when capture has motion blur from fast vehicles.

        Args:
            kernel_size: Blur kernel size (larger = more blur removal)
            angle: Motion direction in degrees (0 = horizontal)
        """
        # Build motion kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

        if angle != 0:
            center = (kernel_size // 2, kernel_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

        blurred = cv2.filter2D(image, -1, kernel)
        # Unsharp mask: original * 1.5 - blurred * 0.5
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    def enhance_for_ocr(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Specialized enhancement for license plate OCR accuracy.
        Upscales small plates, applies CLAHE + adaptive threshold.
        """
        h, w = plate_image.shape[:2]

        # Upscale small plates
        if h < 60:
            scale = 60.0 / h
            plate_image = cv2.resize(plate_image,
                                     (int(w * scale), 60),
                                     interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # Choose threshold based on contrast
        if gray.std() < 40:
            # Low contrast — use adaptive
            result = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, result = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # Augmentation Pipeline (lazy-loaded)
    # ------------------------------------------------------------------

    def _get_augmentation_pipeline(self):
        """Build Albumentations pipeline on first use"""
        if self._augment is not None:
            return self._augment

        try:
            import albumentations as A

            self._augment = A.Compose([
                # Bangalore monsoon conditions
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=20, drop_width=1,
                    drop_color=(200, 200, 200), blur_value=3,
                    brightness_coefficient=0.75,
                    p=0.3
                ),
                A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.25,
                            alpha_coef=0.1, p=0.2),

                # Day/night variation
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,
                    contrast_limit=0.3,
                    p=0.5
                ),

                # Fast vehicle motion blur
                A.MotionBlur(blur_limit=7, p=0.3),

                # Camera sensor noise
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

                # Shadow simulation (trees, buildings)
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=0.25
                ),

                # CLAHE augmentation
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

                # Color variation (different times of day)
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.3
                ),

                # Sharpening for high-speed captures
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            ])
            logger.info("Albumentations augmentation pipeline initialized")

        except ImportError:
            logger.warning("albumentations not installed. Training augmentation disabled.")
            # No-op pipeline
            self._augment = _NoOpAugment()

        return self._augment


class _NoOpAugment:
    """Fallback when albumentations is not installed"""
    def __call__(self, image, **kwargs):
        return {"image": image, **kwargs}
