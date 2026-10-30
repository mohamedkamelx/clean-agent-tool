import hashlib
import os
from pathlib import Path
from typing import Union
import cv2
import numpy as np
from scipy.special import logsumexp
import torch
import deepgaze_pytorch


class DeepGazeHeatmap:
    """
    Wrapper class around DeepGaze IIE for computing saliency visual attention heatmaps.
    Supports in-memory heatmap matrix extraction without disk I/O, as well as
    visual overlay generation for testing.
    """

    def __init__(self, device: str = None, cache_dir: str = ".cache/heatmaps", use_cache: bool = True):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"[DeepGazeHeatmap] Initializing DeepGazeIIE on device: {self.device}")
        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(self.device)
        self.model.eval()
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _image_cache_key(self, image_or_path: Union[str, Path, np.ndarray]) -> str:
        h = hashlib.sha256()
        if isinstance(image_or_path, (str, Path)):
            with open(str(image_or_path), "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        else:
            h.update(np.ascontiguousarray(image_or_path).tobytes())
            h.update(str(image_or_path.shape).encode())
        return h.hexdigest()

    def _prepare_image(self, image_or_path: Union[str, Path, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads and prepares image as BGR and RGB numpy arrays.
        """
        if isinstance(image_or_path, (str, Path)):
            path_str = str(image_or_path)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image not found at path: {path_str}")
            image_bgr = cv2.imread(path_str)
            if image_bgr is None:
                raise ValueError(f"Could not decode image at path: {path_str}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_or_path, np.ndarray):
            if image_or_path.ndim == 2:
                image_bgr = cv2.cvtColor(image_or_path, cv2.COLOR_GRAY2BGR)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            elif image_or_path.shape[2] == 4:
                # RGBA
                image_bgr = cv2.cvtColor(image_or_path, cv2.COLOR_RGBA2BGR)
                image_rgb = cv2.cvtColor(image_or_path, cv2.COLOR_RGBA2RGB)
            else:
                image_bgr = image_or_path.copy()
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_or_path)}")

        return image_bgr, image_rgb

    def get_heatmap(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        output_format: str = "float32",
        normalize: str = "minmax",
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Compute and return the 2D visual attention saliency heatmap matrix.
        No image file is written to disk unless a heatmap cache is enabled.

        Args:
            image_or_path: File path (str/Path) or numpy array image.
            output_format: 'float32' or 'uint8' ([0, 255]).
            normalize:
                'minmax'      -> [0, 1] per image (visualisation; not comparable across screens)
                'probability' -> sums to 1.0 (statistical metrics; comparable across screens)
            use_cache: use the on-disk probability heatmap cache when possible.

        Returns:
            np.ndarray: 2D numpy array of shape (H, W) representing visual attention.
        """
        if normalize not in {"minmax", "probability"}:
            raise ValueError(f"normalize must be 'minmax' or 'probability', got {normalize!r}")

        cache_key = None
        if self.use_cache and use_cache:
            try:
                cache_key = self._image_cache_key(image_or_path)
                cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
                if os.path.exists(cache_path):
                    saliency_prob = np.load(cache_path).astype(np.float32)
                    return self._format_heatmap(saliency_prob, output_format, normalize)
            except Exception:
                cache_path = None
        else:
            cache_path = None

        _, image_rgb = self._prepare_image(image_or_path)
        height, width = image_rgb.shape[:2]

        # Format tensor for DeepGaze: [1, 3, H, W]
        image_tensor = (
            torch.tensor(image_rgb.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        # Uniform centerbias (kept for reproducibility of the original pipeline).
        centerbias_template = np.zeros((height, width), dtype=np.float32)
        centerbias_template -= logsumexp(centerbias_template)
        centerbias_tensor = (
            torch.tensor(centerbias_template, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            log_density_prediction = self.model(image_tensor, centerbias_tensor)

        log_density_map = log_density_prediction.cpu().numpy().squeeze()
        saliency_map = np.exp(log_density_map)

        if saliency_map.shape != (height, width):
            saliency_map = cv2.resize(saliency_map, (width, height), interpolation=cv2.INTER_LINEAR)

        saliency_map = np.maximum(saliency_map, 0.0).astype(np.float32)
        total = float(np.sum(saliency_map))
        if total <= 0:
            saliency_prob = np.full((height, width), 1.0 / max(1, height * width), dtype=np.float32)
        else:
            saliency_prob = (saliency_map / total).astype(np.float32)

        if cache_path:
            try:
                np.save(cache_path, saliency_prob)
            except Exception:
                pass

        return self._format_heatmap(saliency_prob, output_format, normalize)

    @staticmethod
    def _format_heatmap(saliency_prob: np.ndarray, output_format: str, normalize: str) -> np.ndarray:
        if normalize == "probability":
            saliency = saliency_prob.astype(np.float32)
        else:
            min_val, max_val = float(saliency_prob.min()), float(saliency_prob.max())
            if max_val > min_val:
                saliency = ((saliency_prob - min_val) / (max_val - min_val)).astype(np.float32)
            else:
                saliency = np.zeros_like(saliency_prob, dtype=np.float32)

        if output_format == "uint8":
            return (np.clip(saliency, 0.0, 1.0) * 255.0).astype(np.uint8)
        return saliency.astype(np.float32)

    def generate_overlay(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        output_path: str = "attention_heatmap_output.jpg",
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Generate and save a visual overlay combining the original screenshot
        with the JET colormap attention heatmap. Primarily for testing and inspection.

        Args:
            image_or_path: File path or numpy image.
            output_path: Destination path for the saved image.
            alpha: Transparency factor for overlay blend (0.0 - 1.0).

        Returns:
            np.ndarray: Blended BGR image.
        """
        image_bgr, _ = self._prepare_image(image_or_path)
        height, width = image_bgr.shape[:2]

        # Get normalized uint8 heatmap [0, 255]
        saliency_uint8 = self.get_heatmap(image_or_path, output_format="uint8")

        # Apply JET colormap (Red/Yellow = High Attention, Blue = Low Attention)
        heatmap_colored = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
        if heatmap_colored.shape[:2] != (height, width):
            heatmap_colored = cv2.resize(heatmap_colored, (width, height))

        # Blend original screenshot with attention heatmap
        output_overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)

        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(output_path, output_overlay)
            print(f"[DeepGazeHeatmap] Heatmap overlay saved successfully to '{output_path}'")

        return output_overlay


if __name__ == '__main__':
    # Standalone execution test preserving existing workflow
    print("[Standalone Test] Initializing DeepGazeHeatmap...")
    estimator = DeepGazeHeatmap()
    test_image = 'test1.png'
    output_image = 'attention_heatmap_output.jpg'
    estimator.generate_overlay(test_image, output_image)
    print(f"Heatmap generated and saved successfully as '{output_image}'!")
