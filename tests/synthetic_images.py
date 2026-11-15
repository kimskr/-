"""
synthetic_images.py — Generated test images for detector unit tests.
Owner: DEV1

These are NOT a replacement for real fixtures photographed with the UC852
camera (tests/fixtures/pass|fail). They give the unit tests a deterministic,
known ground truth so that the detectors' logic can be tested in CI.
"""
import cv2
import numpy as np

H, W = 800, 1280


def clean_metal(seed: int = 0, level: int = 140) -> np.ndarray:
    """Brushed-metal-like surface: flat gray + fine noise + faint horizontal streaks."""
    rng = np.random.default_rng(seed)
    base = np.full((H, W), level, np.float32)
    streaks = rng.normal(0, 2.0, (H, 1)).astype(np.float32)     # brushing direction
    noise = rng.normal(0, 3.0, (H, W)).astype(np.float32)
    gray = np.clip(base + streaks + noise, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Slight blue-gray tint so the surface is not perfectly neutral
    bgr[:, :, 0] = np.clip(bgr[:, :, 0].astype(np.int16) + 8, 0, 255).astype(np.uint8)
    return bgr


def clean_coated(seed: int = 0) -> np.ndarray:
    """Coated part: saturated blue paint with mild noise."""
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.float32)
    img[:] = (170, 90, 40)  # BGR — blue coating
    img += rng.normal(0, 3.0, (H, W, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def with_scratch(img: np.ndarray, depth: int = 60, thickness: int = 2,
                 p1=(300, 250), p2=(800, 420)) -> np.ndarray:
    """Thin dark line — a scratch."""
    out = img.copy()
    color = tuple(int(c) for c in np.clip(np.array(out[p1[1], p1[0]], int) - depth, 0, 255))
    cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
    return out


def with_dust(img: np.ndarray, n: int = 4, radius: int = 3, delta: int = 70,
              seed: int = 1) -> np.ndarray:
    """Small bright specks — dust particles."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    for _ in range(n):
        x, y = int(rng.integers(100, W - 100)), int(rng.integers(100, H - 100))
        color = tuple(int(c) for c in np.clip(np.array(out[y, x], int) + delta, 0, 255))
        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
    return out


def with_color_patch(img: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Dark, unsaturated rectangle — missing coating exposing bare substrate."""
    out = img.copy()
    side_w, side_h = int(W * np.sqrt(frac)), int(H * np.sqrt(frac))
    x, y = (W - side_w) // 2, (H - side_h) // 2
    out[y:y + side_h, x:x + side_w] = (45, 45, 45)
    return out


def exposure(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale brightness: factor < 1 darkens, > 1 brightens."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def with_glare(img: np.ndarray) -> np.ndarray:
    """Half the frame blown out by a reflection — uneven lighting.
    Use a darker base (e.g. clean_metal(level=100)) so the std exceeds std_max."""
    out = img.copy()
    out[:, W // 2:] = 250
    return out


def part_on_background(seed: int = 0):
    """Metal disc on a dark conveyor. Returns (image, roi inside the disc)."""
    img = np.full((H, W, 3), 25, np.uint8)
    metal = clean_metal(seed)
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (W // 2, H // 2), 350, 255, -1)
    img[mask > 0] = metal[mask > 0]
    roi = [W // 2 - 240, H // 2 - 240, 480, 480]
    return img, roi
