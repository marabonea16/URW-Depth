"""
Augmentare meteo sintetica (ceata, ploaie, zapada) cu severitate continua [0,1].
Extras din evaluate_weather.py pentru reutilizare in demo-ul HuggingFace Space.
"""
import numpy as np
from PIL import Image

SEVERITY_PRESETS = {
    "mild": 0.25,
    "moderate": 0.45,
    "severe": 0.65,
}


def apply_fog(img: Image.Image, severity: float = 0.45) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    fog_color = np.array([220, 220, 220], dtype=np.float32)
    h = arr.shape[0]
    gradient = np.linspace(severity, severity * 0.3, h, dtype=np.float32)[:, None, None]
    arr = arr * (1 - gradient) + fog_color * gradient
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_rain(img: Image.Image, severity: float = 0.45, seed: int = 42) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    num_streaks = int(severity * 600)
    rng = np.random.RandomState(seed)
    for _ in range(num_streaks):
        x = rng.randint(0, w)
        y = rng.randint(0, h - 20)
        length = rng.randint(10, 25)
        alpha = rng.uniform(0.3, 0.6)
        for k in range(length):
            yi = min(y + k, h - 1)
            xi = min(x + k // 3, w - 1)
            arr[yi, xi] = arr[yi, xi] * (1 - alpha) + 200 * alpha
    arr = arr * (1 - severity * 0.15) + 128 * severity * 0.15
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_snow(img: Image.Image, severity: float = 0.45, seed: int = 42) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2, keepdims=True)
    arr = arr * (1 - severity * 0.4) + gray * severity * 0.4
    rng = np.random.RandomState(seed)
    num_flakes = int(severity * 800)
    ys = rng.randint(0, h, num_flakes)
    xs = rng.randint(0, w, num_flakes)
    alphas = rng.uniform(0.5, 1.0, num_flakes)
    for y, x, a in zip(ys, xs, alphas):
        arr[y, x] = arr[y, x] * (1 - a) + 255 * a
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


WEATHER_FNS = {"fog": apply_fog, "rain": apply_rain, "snow": apply_snow}


def apply_weather(img: Image.Image, weather_type: str, severity: float) -> Image.Image:
    if weather_type == "none" or severity <= 0:
        return img
    return WEATHER_FNS[weather_type](img, severity)
