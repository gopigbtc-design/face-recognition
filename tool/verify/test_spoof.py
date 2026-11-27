import numpy as np
from skimage import data, color, transform
from mirror import *
M = Models(__import__('os').path.join(__import__('os').path.dirname(__file__), '../../assets/models'))
def raw(rgb):
    x = (rgb.astype(np.float32) / 255.0)[None]
    M.fas.set_tensor(M.fas.get_input_details()[0]["index"], x); M.fas.invoke()
    o = {d["name"]: M.fas.get_tensor(d["index"])[0] for d in M.fas.get_output_details()}
    return o["Identity"], o["Identity_1"]
im = data.astronaut(); nv = rgb_to_nv21(im)
for name, crop in [("box 130", (230,112,130)), ("box 160", (226,110,160)), ("box 200", (226,115,200))]:
    rgb = sample_rgb(nv, 512, 512, 512, 0, *crop, 0.0, 256)
    c, m = raw(rgb); print(name, "cls", np.round(c, 2), "mask", np.round(m, 2), "score", round(float((abs(c)*m).sum()), 3), "lap", laplacian_score(rgb))
    c, m = raw(rgb[..., ::-1].copy()); print("   BGR order score", round(float((abs(c)*m).sum()), 3))
for name, rgb in [("noise", np.random.default_rng(0).integers(0,256,(256,256,3)).astype(np.uint8)), ("flat grey", np.full((256,256,3),128,np.uint8)), ("cameraman face", (transform.resize(color.gray2rgb(data.camera())[50:150,205:305],(256,256))*255).astype(np.uint8))]:
    c, m = raw(rgb); print(name, "score", round(float((abs(c)*m).sum()), 3), "mask", np.round(m,2))
