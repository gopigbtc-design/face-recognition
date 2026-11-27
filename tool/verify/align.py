import numpy as np, math
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
def similarity(src, dst):
    """Port of Dart alignToTemplate(): least-squares s,R,t with dst = s R src + t."""
    ps, qs = src.mean(0), dst.mean(0)
    p, q = src - ps, dst - qs
    a = (p[:, 0] * q[:, 0] + p[:, 1] * q[:, 1]).sum()
    b = (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum()
    n = (p ** 2).sum()
    sc, ss = a / n, b / n
    s = math.hypot(sc, ss); ang = math.atan2(ss, sc)
    tx = qs[0] - (sc * ps[0] - ss * ps[1]); ty = qs[1] - (ss * ps[0] + sc * ps[1])
    cx = sc * 56 - ss * 56 + tx; cy = ss * 56 + sc * 56 + ty
    return cx, cy, 112 * s, ang
