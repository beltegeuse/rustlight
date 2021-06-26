import pyexr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 

COLOR_MAP = 'jet'
NP_INT_TYPES = [np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64]

def falsecolor(error, max_val):
    """Compute false color heatmap."""

    cmap = plt.get_cmap(COLOR_MAP)
    mean = np.mean(error, axis=2)
    print(mean)
    val = np.clip(mean / max_val, 0, 1)
    return cmap(val)

def hdr_to_ldr(path_dir, img, imgname):
    """HDR to LDR conversion for web display."""

    # Image already in ldr
    if (img.dtype in NP_INT_TYPES):
        ldr = Image.fromarray(img.astype(np.uint8))
    else:
        ldr = Image.fromarray(
            (pyexr.tonemap(img) * 255).astype(np.uint8))
    ldr_fname = '{}.png'.format(imgname)
    ldr_fname = os.path.basename(ldr_fname)
    ldr_fname = ldr_fname.replace(" ", "_")
    ldr_fname = ldr_fname.replace("+", "_")
    ldr_path = os.path.join(path_dir, ldr_fname)
    print(ldr_path)
    ldr.save(ldr_path)

inputDir = "/home/agruson/projects/point-normal/figures/figures/images/steps/"
outputDir = "/home/agruson/projects/point-normal/figures/figures/images/out/steps"
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

imgs = ["budda_steps.exr", "retro_steps.exr", "retro_plane_steps.exr", "point_steps.exr", "plane_steps.exr"]
for img in imgs:
    name = img.split(".")[0]
    img = pyexr.read(os.path.join(inputDir, img))
    cmap = falsecolor(img, 12)
    hdr_to_ldr(outputDir, cmap, name)