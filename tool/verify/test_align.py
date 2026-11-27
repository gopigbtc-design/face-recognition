import numpy as np, math
from skimage import data, transform, color
from mirror import *; from align import *
M = Models(__import__('os').path.join(__import__('os').path.dirname(__file__), '../../assets/models'))
im = data.astronaut(); nv = rgb_to_nv21(im)
LM = np.array([[203.8, 100.8], [244.5, 103.8], [223.0, 126.0], [205.0, 141.0], [244.5, 142.0]])
cx, cy, side, ang = similarity(TEMPLATE, LM)
print(f"aligned crop: c=({cx:.1f},{cy:.1f}) side={side:.1f} angle={math.degrees(ang):.1f}deg")
from skimage import io; 
rng = np.random.default_rng(0)
for jit in (0.0, 1.0, 2.0, 3.0):
    E = []
    for k in range(20):
        c = similarity(TEMPLATE, LM + rng.normal(0, jit, LM.shape) if jit else LM)
        E.append(M.embed(sample_rgb(nv, 512, 512, 512, 0, *c, 112)))
    E = np.array(E); S = E @ E.T; iu = np.triu_indices(len(E), 1)
    print(f"landmark jitter sd={jit}px: pairwise sim min={S[iu].min():.3f} mean={S[iu].mean():.3f}")
base = M.embed(sample_rgb(nv, 512, 512, 512, 0, cx, cy, side, ang, 112))
th = 15; rimg = (transform.rotate(im, th, center=(223, 120)) * 255).astype(np.uint8)
R = np.array([[math.cos(math.radians(-th)), -math.sin(math.radians(-th))], [math.sin(math.radians(-th)), math.cos(math.radians(-th))]])
LMr = (LM - [223, 120]) @ R.T + [223, 120]
print("tilted 15deg photo, aligned via landmarks:", round(float(base @ M.embed(sample_rgb(rgb_to_nv21(rimg), 512, 512, 512, 0, *similarity(TEMPLATE, LMr), 112))), 3))
sm = (transform.rescale(im, 0.6, channel_axis=-1, order=1) * 255).astype(np.uint8)[:306, :306]
print("same face at 60% scale (further away):", round(float(base @ M.embed(sample_rgb(rgb_to_nv21(sm), 306, 306, 306, 0, *similarity(TEMPLATE, LM * 0.6), 112))), 3))
