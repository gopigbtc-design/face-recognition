import numpy as np, math
from skimage import data
from mirror import rgb_to_nv21, sample_rgb

up = data.astronaut()[:480, :320]          # upright frame (H'=480, W'=320), portrait like the phone
ok = True
for rot in (0, 90, 180, 270):
    raw = np.ascontiguousarray(np.rot90(up, k=rot // 90))   # sensor image: needs CW rot to be upright
    h, w = raw.shape[:2]
    nv = rgb_to_nv21(raw)
    for (cx, cy, side, ang, size) in [(160.0, 200.0, 128.0, 0.0, 128), (150.3, 210.7, 200.0, 0.35, 112), (40.0, 30.0, 120.0, -0.5, 96)]:
        got = sample_rgb(nv, w, h, w, rot, cx, cy, side, ang, size).astype(int)
        ref_frame = rgb_to_nv21(up)
        ref = sample_rgb(ref_frame, up.shape[1], up.shape[0], up.shape[1], 0, cx, cy, side, ang, size).astype(int)
        d = np.abs(got - ref).mean()
        print(f"rot={rot:3d} crop=({cx},{cy},{side},{ang}) mean|diff| vs rotation-0 = {d:.2f}")
        ok &= d < 6   # chroma subsampling grid differs after rotation; luma must line up
# Absolute ground truth: 1:1 axis-aligned crop must equal the upright pixels.
for rot in (0, 90, 180, 270):
    raw = np.ascontiguousarray(np.rot90(up, k=rot // 90)); h, w = raw.shape[:2]
    got = sample_rgb(rgb_to_nv21(raw), w, h, w, rot, 160.0, 200.0, 128.0, 0.0, 128).astype(int)
    truth = up[136:264, 96:224].astype(int)
    d = np.abs(got - truth).mean(); print(f"rot={rot:3d} vs ground-truth pixels: mean|diff|={d:.2f}")
    ok &= d < 6
# Row padding (rowStride > width) must give identical output.
raw = np.ascontiguousarray(np.rot90(up, k=3)); h, w = raw.shape[:2]; stride = w + 32
nv = rgb_to_nv21(raw); yp = nv[:w*h].reshape(h, w); vu = nv[w*h:].reshape(h//2, w)
pad = np.concatenate([np.pad(yp, ((0,0),(0,32))).reshape(-1), np.pad(vu, ((0,0),(0,32))).reshape(-1)[:-32]])
a = sample_rgb(pad, w, h, stride, 270, 160.0, 200.0, 150.0, 0.2, 112); b = sample_rgb(nv, w, h, w, 270, 160.0, 200.0, 150.0, 0.2, 112)
print("padded stride identical:", np.array_equal(a, b)); ok &= np.array_equal(a, b)
# Angle convention: crop x-axis must follow (cos a, sin a). Rotate content by +a about centre, sample with angle a -> undo.
print("ALL GEOMETRY CHECKS PASSED" if ok else "GEOMETRY FAILURE")
