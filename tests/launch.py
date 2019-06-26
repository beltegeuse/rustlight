import os
import sys
import argparse
import subprocess


class RenderTest:
    def __init__(self, ref, scene_file, techniques, extra={}):
        self.ref = ref
        self.scene_file = scene_file
        self.techniques = techniques
        self.extra = extra


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch analysis of rendered images.')
    parser.add_argument('-r',   '--root', type=str,
                        help='root directory for HTML visualizer')
    parser.add_argument('-sc',   '--skipcompute', action="store_true",
                        help='skip the computation')
    parser.add_argument('-t', '--techniques', type=str, nargs='+',
                        help='only render the listed techniques')
    parser.add_argument('-s', '--scenes', type=str, nargs="+",
                        help='only render the listed scenes')
    parser.add_argument('--time', type=int, default=30, required=True,
                        help='time for the rendering')
    args = parser.parse_args()

    DEFAULT_SPP = 32
    DEFAULT_ANALYSE_SCRIPT = ["python", "interactive-viewer/tools/analyze.py"]
    DEFAULT_SCENE_SCRIPT = ["python", "interactive-viewer/tools/scene.py"]
    SCENE_DIR = "/home/beltegeuse/projects/pbrt_rs/data/pbrt/"
    EXT = ".exr"
    DEFAULT_COMMAND = ['cargo', 'run', '--release',
                       '--features', 'default openexr', '--']
    GI_ALGO = ["path", "path-explicit", "light-explicit", "pssmlt", "vpl"]
    tests = {
        "cbox_ao":
        RenderTest("ref_cbox_ao_1_0", "cornell-box/scene.pbrt",
                   ["ao"], {"-d": "1.0"}),
        "cbox_path":
        RenderTest("ref_cbox_path", "cornell-box/scene.pbrt",
                   GI_ALGO[:]),
        "staircase_path":
        RenderTest("ref_staircase_path", "staircase/scene.pbrt",
                   GI_ALGO[:]),
    }
    for (n, t) in tests.items():
        # Checking if we need to skip the scene
        if(args.scenes):
            if(not (n in args.scenes)):
                print("SKIP:", n)
                continue
            print("{}".format(n))

        # Create the output directory if needed
        if(not os.path.exists(n)):
            os.makedirs(n)
            print("Create: {}".format(n))
        os.chdir("..")
        scene = SCENE_DIR + os.path.sep + t.scene_file

        # Iterate on all the technique
        # to generated the rendering outputs
        rendered_technique = 0
        for a in t.techniques:
            if(args.skipcompute):
                # We increate the number of rendered technique
                # to force to update the HTML
                rendered_technique += 1
                continue

            # Checking if we need to skip the technique
            if(args.techniques):
                if(not (a in args.techniques)):
                    print("SKIP:", a)
                    continue
                print(" - {}".format(a))
            rendered_technique += 1

            # Create directory for each techniques
            output_name_dir = os.path.join("tests", n, a + "_partial")
            if os.path.exists(output_name_dir):
                import shutil
                shutil.rmtree(output_name_dir)
            os.makedirs(output_name_dir)
            print("Create: {}".format(output_name_dir))
            output_name = os.path.join(output_name_dir, a + ".exr")

            # SPP, some technique have special number of SPP
            spp = DEFAULT_SPP
            if(a == "vpl"):
                spp = 1

            COMMAND = DEFAULT_COMMAND[:]
            COMMAND += ['-a', str(args.time), '-n',
                        str(spp), "-o", output_name, scene, a]
            for (n_e, v_e) in t.extra.items():
                COMMAND += [n_e, v_e]
            print("LAUNCH:", " ".join(COMMAND))
            subprocess.run(COMMAND)
        os.chdir("tests")

        if(rendered_technique == 0):
            print("SKIP: Backing HTML")
            continue
        # Update the main HTML scene (remove and add)
        SCENE_SCRIPT = DEFAULT_SCENE_SCRIPT[:]
        SCENE_SCRIPT += ["-r", "interactive-viewer/"]
        subprocess.run(SCENE_SCRIPT + ["remove", "-n", n])
        subprocess.run(SCENE_SCRIPT + ["add", "-n", n])

        # Update the HTML scenes results
        SCRIPT = DEFAULT_ANALYSE_SCRIPT[:]
        SCRIPT += ["-d", "interactive-viewer/scenes/"+n+os.path.sep]
        SCRIPT += ["-r", "ref/"+t.ref+".exr"]
        SCRIPT += ["-m", "l1", "l2", "mape"]
        SCRIPT += ['-np']  # NP false color
        SCRIPT += ["-n"]
        for a in t.techniques:
            SCRIPT += [a]
        SCRIPT += ["-p"]
        for a in t.techniques:
            SCRIPT += [os.path.join(n, a+"_partial")]
        SCRIPT += ['-t']
        for a in t.techniques:
            import glob
            partial_dir = os.path.join(n, a+"_partial")
            glob_expr = os.path.join(partial_dir, '{}_[0-9]*.exr'.format(a))
            img = glob.glob(glob_expr)
            if (len(img) == 0):
                raise Exception(
                    'Could not find files matching {}'.format(glob_expr))
            img = max(img, key=os.path.getctime)
            SCRIPT += [img]
        print("LAUNCHED:", " ".join(SCRIPT))
        subprocess.run(SCRIPT)
