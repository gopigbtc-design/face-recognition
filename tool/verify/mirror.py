"""Faithful Python port of lib/core/yuv_sampler.dart + worker preprocessing,
used to verify geometry and model I/O against real TFLite models."""
import math
import numpy as np


def rgb_to_nv21(rgb):
    """Encode RGB (H,W,3) -> NV21 bytes (BT.601 full range, inverse of decoder)."""
    h, w, _ = rgb.shape
    f = rgb.astype(np.float64)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) / 1.772 + 128
    v = (r - y) / 1.402 + 128
    us = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    vs = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    vu = np.stack([vs, us], axis=-1).reshape(h // 2, w)
    out = np.concatenate([y.reshape(-1), vu.reshape(-1)])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def sample_rgb(src, w, h, stride, rotation, cx, cy, side, angle, size):
    """Line-by-line port of sampleRgb() (vectorised, same formulas)."""
    uv_off = stride * h
    has_uv = len(src) >= uv_off + stride * (h // 2 - 1) + w
    rot = rotation % 360
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    step = side / size
    half = side / 2
    vv, uu = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dy = (vv + 0.5) * step - half
    dx = (uu + 0.5) * step - half
    ux = cx + dx * cos_a - dy * sin_a
    uy = cy + dx * sin_a + dy * cos_a
    if rot == 90:
        rx, ry = uy, h - ux
    elif rot == 180:
        rx, ry = w - ux, h - uy
    elif rot == 270:
        rx, ry = w - uy, ux
    else:
        rx, ry = ux, uy
    fx = np.clip(rx - 0.5, 0, w - 1)
    fy = np.clip(ry - 0.5, 0, h - 1)
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = np.where(x0 + 1 < w, x0 + 1, x0)
    y1 = np.where(y0 + 1 < h, y0 + 1, y0)
    ax = fx - x0
    ay = fy - y0
    s = src.astype(np.float64)
    r0 = y0 * stride
    r1 = y1 * stride
    top = s[r0 + x0] + (s[r0 + x1] - s[r0 + x0]) * ax
    bot = s[r1 + x0] + (s[r1 + x1] - s[r1 + x0]) * ax
    yy = top + (bot - top) * ay
    if has_uv:
        xi = np.clip((fx + 0.5).astype(int), 0, w - 1)
        yi = np.clip((fy + 0.5).astype(int), 0, h - 1)
        uv_idx = uv_off + (yi >> 1) * stride + (xi & ~1)
        v = s[uv_idx] - 128.0
        u = s[uv_idx + 1] - 128.0
        r = yy + 1.402 * v
        g = yy - 0.344136 * u - 0.714136 * v
        b = yy + 1.772 * u
    else:
        r = g = b = yy
    out = np.stack([r, g, b], axis=-1)
    # Dart: clamp then .round() (round half away from zero; values >= 0 here)
    out = np.where(out < 0, 0, np.where(out > 255, 255, np.floor(out + 0.5)))
    return out.astype(np.uint8)


def laplacian_score(rgb, threshold=50):
    g = (rgb[..., 0] * 0.3 + rgb[..., 1] * 0.59 + rgb[..., 2] * 0.11).astype(np.int64)
    c = g[1:-1, 1:-1]
    v = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * c
    return int((v > threshold).sum())


def l2n(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v * 0


class Models:
    def __init__(self, d):
        from ai_edge_litert.interpreter import Interpreter
        self.fn = Interpreter(model_path=f"{d}/MobileFaceNet.tflite", num_threads=2)
        self.fn.allocate_tensors()
        self.fas = Interpreter(model_path=f"{d}/FaceAntiSpoofing.tflite", num_threads=2)
        self.fas.allocate_tensors()

    def embed(self, rgb112):
        x = (rgb112.astype(np.float32) - 127.5) / 128.0
        batch = np.stack([x, x[:, ::-1, :]])  # slot 1 = mirrored, as in Dart
        i = self.fn.get_input_details()[0]["index"]
        self.fn.set_tensor(i, batch)
        self.fn.invoke()
        o = self.fn.get_tensor(self.fn.get_output_details()[0]["index"])
        return l2n(l2n(o[0]) + l2n(o[1]))

    def spoof(self, rgb256):
        x = (rgb256.astype(np.float32) / 255.0)[None]
        self.fas.set_tensor(self.fas.get_input_details()[0]["index"], x)
        self.fas.invoke()
        outs = {d["name"]: self.fas.get_tensor(d["index"]) for d in self.fas.get_output_details()}
        cls, mask = outs["Identity"][0], outs["Identity_1"][0]
        return float((np.abs(cls) * mask).sum())
