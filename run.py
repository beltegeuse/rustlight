import subprocess
import shutil
import os
import shlex


def run(command):
    process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.communicate()[1]


# Default configuration
THREADS = "1"
RAND = "independent:0"

class SPP:
    def __init__(self, v):
        self.v = v
    def args(self):
        return ['-n', str(self.v)]
    def capture(self, l, info):
        pass
    
class EQTIME:
    def __init__(self, time, v=1):
        self.time = time
        self.v = v
    def args(self):
        return ['-z', str(self.time), '-n', str(self.v)]
    def capture(self, l, info):
        if 'rustlight::integrators::equal_time - Number spp:' in l:
            info["spp"] = l.split(' ')[-1]


def run_variations(output, scene, media_config, variations, config, delete=True):
    # Transform path to absolut ones
    #output = os.path.abspath(output)
    #scene = os.path.abspath(scene)

    if delete:
        if os.path.exists(output):
            shutil.rmtree(output)
    if not os.path.exists(output):
        os.mkdir(output)

    base_command = ["cargo", "run", "--release",
                    "--example=cli", '--features', 'embree openexr', "--"]
    base_command += ['-r', f'{RAND}',
                     '-m', f'{media_config}',
                     '-t', f'{THREADS}',
                     '-l', f'{output}/out.log']
    base_command += config["spp"].args()
    if "options" in config:
        base_command += ['-x', config['options']]
    if "scale" in config: # Scale the image
        base_command += ['-s', str(config['scale'])]
    infos = []

    for variation in variations:
        # Copy base command
        command = base_command[:]

        # Output, scene, integrator
        command += [
            '-o', f'{output}/{variation["name"]}.exr',
            scene,
            'point_normal',
        ]
        command += shlex.split(variation["options"])
        print('=======================')
        print(f'[RUN] {variation["name"]}')
        print(' '.join(command))
        
        log = run(command)
        log = str(log).split('\\n')
        # Output the log (lines by lines)
        print("\n".join(log))

        # Extract relevant informations
        current_info = {"details":[], "name":os.path.basename(variation["name"])}
        info_integrator = False
        for l in log:
            # Strategy information
            if "INFO cli - Strategy:" in l:
                info_integrator = True
            if "INFO rustlight::integrators - Build acceleration data structure..." in l:
                info_integrator = False
            if info_integrator:
                current_info["details"] += [l]
            # For the time            
            if " Elapsed Integrator" in l:
                current_info["time"] = l.split(' ')[-2]
            config["spp"].capture(l, current_info)
        infos += [current_info]
    return infos

def canonical_experiment():
    # Configuration
    ISO_CONFIG = {
        #'spp' : SPP(4)
        'spp' : EQTIME(0.5, 1)

    }
    ANISO_CONFIG = {
        #'spp' : SPP(16) 
        'spp' : EQTIME(2.0, 1)
    }


    # Collect all infos
    infos = []
    exp = []

    # Isotropic-point
    exp += ["iso-point"]
    variations = [
        {"name": "eq", "options": '-s eq_ex'},
        {"name": "eq_warp_TB",
            "options": '-s eq_warp_ex -w "T" -t "B"'},
        {"name": "eq_tr_taylor_o6",
            "options": '-s eq_tr_taylor_ex -o 6'},
    ]
    infos.append(run_variations("results/point",
                "scene/point-normal/point.xml", "0.5:0.2", variations, ISO_CONFIG))

    # # Anisotropic-point
    exp += ["aniso-point"]
    variations = [
        {"name": "eq", "options": '-s eq_ex'},
        {"name": "eq_warp_PB",
            "options": '-s eq_warp_ex -w "P" -t "B"'},
        {"name": "eq_warp_PTB",
            "options": '-s eq_warp_ex -w "PT" -t "B"'},
        {"name": "eq_warp_TPB",
            "options": '-s eq_warp_ex -w "TP" -t "B"'},
        {"name": "eq_phase_taylor_o6",
            "options": '-s eq_phase_taylor_ex -o 6'},
        {"name": "eq_best",
            "options": '-s eq_best_ex'},
    ]
    infos.append(run_variations("results/point_g_9",
                "scene/point-normal/point.xml", "0.5:0.2:0.9", variations, ANISO_CONFIG))

    # Isotropic-PN
    exp += ["iso-pn"]
    variations = [
        {"name": "eq", "options": '-s eq_clamped_ex'},
        {"name": "pn",
            "options": '-s pn_ex'},
        {"name": "pn_warp_TB",
            "options": '-s pn_warp_ex -w "T" -t "B"'},
        {"name": "pn_tr_taylor_o6",
            "options": '-s pn_tr_taylor_ex -o 6'},
    ]
    infos.append(run_variations("results/point-normal",
                "scene/point-normal/point_normal.xml", "0.5:0.2", variations, ISO_CONFIG))

    # Anisotropic-PN
    exp += ["aniso-pn"]
    variations = [
        {"name": "eq", "options": '-s eq_clamped_ex'},
        {"name": "pn",
            "options": '-s pn_ex'},
        {"name": "pn_warp_PB",
            "options": '-s pn_warp_ex -w "P" -t "B"'},
        {"name": "pn_warp_PTB",
            "options": '-s pn_warp_ex -w "PT" -t "B"'},
        {"name": "pn_warp_TPB",
            "options": '-s pn_warp_ex -w "TP" -t "B"'},
        {"name": "pn_phase_taylor_o6",
            "options": '-s pn_phase_taylor_ex -o 6'},
        {"name": "pn_best",
            "options": '-s pn_best_ex'},
    ]
    infos.append(run_variations("results/point-normal_g_9",
                "scene/point-normal/point_normal.xml", "0.5:0.2:0.9", variations, ANISO_CONFIG))

    # Print informations
    for e, ins in zip(exp, infos):
        print("=====")
        print(e)
        for i in ins:
            del i["details"]
            print(i)

def buddha_experiment():
    CONFIG = {
        'spp' : SPP(1),
        'options' :  "ats hvs_lights",
        'scale' : '0.25'
    }
    variations = [
        {"name": "splitting-pn", "options": '-s pn_ex -k 0.04 -z'},
        {"name": "splitting-pn-best", "options": '-s pn_best_ex -k 0.04 -z'},
        {"name": "splitting-equiangular-clamped", "options": '-s eq_clamped_ex -k 0.04 -z'},
    ]

    infos = run_variations("results/buddha",
                "scene/point-normal/buddha/scene.pbrt", "0.7:0.2", variations, CONFIG)

def retro_experiment():
    CONFIG = {
        'spp' : EQTIME(300, 1),
        'options' :  "ats",
        'scale' : '0.5'
    }

    variations = [
        {"name": "splitting-pn", "options": '-s pn_ex -k 0.04 -z'},
        {"name": "splitting-pn-best", "options": '-s pn_best_ex -k 0.04 -z'},
        {"name": "splitting-equiangular-clamped", "options": '-s eq_clamped_ex -k 0.04 -z'},
    ]

    infos = run_variations("results/retrowave",
                "scene/point-normal/retrowave/scene.pbrt", "0.1:0.1:0.8", variations, CONFIG, delete=False)

canonical_experiment()
buddha_experiment()
retro_experiment()