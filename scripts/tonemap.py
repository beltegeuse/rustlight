import pyexr
import PIL.Image
import numpy as np
import sys

# For color maps
import matplotlib.pyplot as plt
COLOR_MAP = 'viridis'

# Can be commented if not needed
#from skimage.measure import compare_ssim, compare_psnr

def compute_metric(ref, test, metric, eps=1e-2):
    """Compute desired metric."""

    diff = np.array(ref - test)
    if (metric == 'l1'):      # Absolute error
        error = np.abs(diff)
    elif (metric == 'l2'):    # Squared error
        error = diff * diff
    elif (metric == 'mrse'):  # Relative squared error
        error = diff * diff / (ref * ref + eps)
    elif (metric == 'mape'):  # Relative absolute error
        error = np.abs(diff) / (ref + eps)
    elif (metric == 'smape'):  # Symmetric absolute error
        error = 2 * np.abs(diff) / (ref + test + eps)
    else:
        raise ValueError('Invalid metric')

    return error

def falsecolor(error, clip, eps=1e-2):
    """Compute false color heatmap."""

    cmap = plt.get_cmap(COLOR_MAP)

    # Sum RGB
    mean = np.mean(error, axis=2)

    # Scale by clipping
    min_val, max_val = clip
    val = np.clip((mean - min_val) / (max_val - min_val), 0, 1)
    return cmap(val)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Batch test')
    parser.add_argument('inputs', help='images inputs', type=str,  metavar='O', nargs="+")
    parser.add_argument('-e',   '--exposure', help='images inputs', type=int, default=0, required=False)
    parser.add_argument('-m',   '--metric', help='difference metric',
                        choices=['l1', 'l2', 'mrse', 'mape', 'smape'], type=str) # dssim
    parser.add_argument('-eps', '--epsilon',
                        help='epsilon value', type=float, default=1e-2)
    parser.add_argument('-c',   '--clip',
                        help='clipping values for min/max', nargs=2, type=float, default=[0, 1])
    parser.add_argument('-fc',  '--falsecolor',
                        help='false color heatmap output file', type=str)
    parser.add_argument('-r',   '--ref',
                        help='reference image filename', type=str)
    parser.add_argument('-p',   '--plain',
                        help='output error as plain text', action='store_true')

    args = parser.parse_args()
    ref = None
    if args.metric:
        # Read the reference
        if args.ref is None:
            print("Need to provide reference to compute error images (with -r)")
            sys.exit(1)
        ref = pyexr.read(args.ref)
        if args.exposure != 0:
            ref *= pow(2, args.exposure)

    for img_name in args.inputs:
        i = pyexr.read(img_name)
        if args.exposure != 0:
            i *= pow(2, args.exposure)

        print(f'img: {img_name}')
        if args.metric:
            m = args.metric
            err_img = compute_metric(ref, i, m, args.epsilon)

            # Show the error mean
            err_mean = np.mean(err_img)
            print(' - {}  = {:.6f}'.format(m, err_mean))

            if args.clip != [0, 1]:
                print('Clipping values in range: [{:.2f}, {:.2f}]'.format(
                    args.clip[0], args.clip[1]))
            fc = falsecolor(err_img, args.clip, args.epsilon)
            o = img_name.replace(".exr", f"_{m}.png")
            plt.imsave(o, fc)


        # Output name (replace name to png)
        o = img_name.replace(".exr", ".png")


        # Convert to png (gamma curve + clip high values) [Tonemapping]
        i = np.power(i, 1.0 / 2.2)
        i = np.clip((i*255.0), 0.0, 255.0).astype(np.uint8)
        pil_img = PIL.Image.fromarray(i)

        pil_img.save(o)
