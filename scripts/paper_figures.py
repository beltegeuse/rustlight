# Classical python imports
import optparse
import xml.etree.ElementTree as ET
import os
import math
import logging
import copy

# For plotting informations
import matplotlib.pyplot as plt
from matplotlib import cm

try:
    import Image
    import ImageDraw
except ImportError:
    from PIL import Image
    from PIL import ImageDraw
import numpy as np
import pyexr

logger = logging.getLogger(__name__)


def open_img(path):
    ext = path.split(".")[-1]
    if ext == "exr":
        img = pyexr.read(path)
        return img
    elif ext == "hdr":
        import cv2
        fp = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        fp = cv2.cvtColor(fp, cv2.COLOR_BGR2RGB)
        return np.array(fp, dtype=np.float32)
    else:
        raise Exception("{} extension is not supported ({})".format(ext, path))

def saveNPImage(imgPath,pRef,output,scale=1.0):

    pixelsHDR = open_img(imgPath)

    # --- Change data to compute norm
    pRef = lum(pRef)
    pixelsHDR = lum(pixelsHDR)

    pixelsHDR[(pRef+pixelsHDR) > 0] = 2.0*(pixelsHDR - pRef)/(pixelsHDR+pRef)
    pixelsHDR[(pRef+pixelsHDR) == 0] = 0.0 # TODO: Check this

    logger.debug("NP Image: Min",min(pixelsHDR),"Max",max(pixelsHDR))

    im = Image.new("RGB", (width,height))
    pixConv = lambda x,v: (int(v*255),0,0) if x < 0.0 else (0,int(v*255),0) if x > 0.0 else (0,0,0)

    # FIXME
    buf = [pixConv(p,pow(min(abs(p)/scale,1.0),1/2.2)) for p in pixelsHDR[:,:]]
    im.putdata(buf)
    im.save(output, quality=100, subsampling=0)
    im.close()

def saveNPImageRef(imgPath,imgRefPath,output,scale=1.0):
    """
    The difference from the previous function is here
    we read the reference again from scratch
    """

    pRef = open_img(imgRefPath)
    return saveNPImage(imgPath, pRef, output, scale)

def saveFig(w,h,data,output,cmapData=None, minData=None, maxData=None):

    # --- Save image
    fig = plt.figure(figsize=((w/100)+1, (h/100)+1), dpi=100)
    cax = plt.figimage(data, vmin=minData, vmax=maxData, cmap=cmapData)
    fig.savefig(output)#, bbox_inches=extent)
    plt.close()

    # Load and resave image
    im = Image.open(output)
    (widthN, heightN) = im.size
    logger.info("Detected size: ",widthN,heightN, "targeted", w, h)
    im2 = im.crop(((widthN-w),
                   (heightN-h),
                   w,h))
    im.close()
    im2.save(output, quality=100, subsampling=0)


def lum(p):
    return 0.21268*p[:,:,0] + 0.7152*p[:,:,1] + 0.0722*p[:,:,2]

def readColor(t):
    tA = [int(v) for v in t.split(",")]
    return (tA[0],tA[1],tA[2])

class MetricOp:
    def __init__(self):
        self.ref = ""
        self.img = ""
        self.exposure = 0
        self.mask = ""

    def readXML(self, n, config):
        self.ref = n.attrib["ref"]
        self.img = n.attrib["img"]
        self.exposure = config["exposure"]

    def show(self, wk, config):
        # --- Load reference
        pixelsHDR = open_img(os.path.join(wk,config["inputDir"],self.img))
        pRef = open_img(os.path.join(wk,config["inputDir"],self.ref))
        if(self.exposure != 0.0):
            mult = math.pow(2, self.exposure)
            pixelsHDR *= mult
            pRef *= mult
            
        # --- Change data to compute norm
        ref = lum(pRef)
        test = lum(pixelsHDR)

        diff = np.array(ref - test)
        eps = 1e-2

        metrics = {}
        metrics["l1"] = np.abs(diff)
        metrics["l2"] = diff * diff
        metrics["mrse"] = diff * diff / (ref * ref + eps)
        metrics["mape"] = np.abs(diff) / (ref + eps)
        metrics["smape"] = np.abs(diff) / (ref + test)
        metrics["smape"][ref == 0] = 0.0

        print("Metric for "+self.img+" ( with "+self.ref+")")
        for (n, v) in metrics.items():
            print(n, ":", np.sum(v)/(pRef.shape[0]*pRef.shape[1]))

class BoxOp:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.sX = 0
        self.sY = 0
        self.border = 0
        self.color = (0,0,0)

    def readXML(self, n):
        self.x = int(n.attrib["x"])
        self.y = int(n.attrib["y"])
        self.sX = int(n.attrib["sX"])
        self.sY = int(n.attrib["sY"])
        self.border = int(n.attrib["border"])
        if(self.border != 0):
            self.color = readColor(n.attrib["color"])

    def apply(self, im, w, h):
        x0 = self.x
        y0 = self.y
        x1 = x0 + self.sX
        y1 = y0 + self.sY

        im2 = im.copy()
        im2 = im2.crop((x0, y0, x1, y1))
        draw = ImageDraw.Draw(im)
        draw.rectangle((x0-self.border,
                        y0-self.border,
                        x1+self.border-1,
                        y1+self.border-1), fill=self.color)
        im.paste(im2, (x0, y0))

        return im

class CropOp(BoxOp):
    def __init__(self):
        BoxOp.__init__(self)

    def apply(self, im, w, h):
        x0 = self.x
        y0 = self.y
        x1 = x0 + self.sX
        y1 = y0 + self.sY

        sX = self.sX + self.border*2
        sY = self.sY + self.border*2

        im2 = im.copy()
        im2 = im2.crop((x0-self.border,
                        y0-self.border,
                        x1+self.border,
                        y1+self.border))

        if(self.border != 0):
            im3 = im2.copy()
            im3 = im3.crop((self.border, self.border,
                            sX - self.border, sY - self.border))

            draw = ImageDraw.Draw(im2)
            draw.rectangle((0,
                            0,
                            sX,
                            sY), fill=self.color)
            im2.paste(im3, (self.border, self.border))

        return im2

class ImageOp(object):
    def __init__(self):
        self.expo = 0
        self.input = ""
        self.output = ""
        self.actions = []
        self.actions_load = None
        self.gamma = 1.0
        self.high_res = False

        # === internal attrib
        self.im = None
        self.pixelsHDR = []


    def readXML(self, n, config):
        # === Read all basic informations
        self.expo = config["exposure"]
        self.input = n.attrib["input"]
        if "output" in n.attrib: 
            self.output = n.attrib["output"]
        else:
            # Rename input with jpg
            self.output = self.input.replace(".exr", ".jpg")
        self.gamma = config["gamma"]

        # === Read all actions
        for bXML in n.iter('Box'):
            b = BoxOp()
            b.readXML(bXML)
            self.actions.append(b)
        for bXML in n.iter('Crop'):
            b = CropOp()
            b.readXML(bXML)
            self.actions.append(b)
            
    def getImg(self,wk):
        return os.path.join(wk, config["inputDir"], self.input)

    def loadHDR(self, wk):
        # --- Load HDR
        self.pixelsHDR = open_img(self.getImg(wk))
        print("Reading HDR: ", self.getImg(wk))

        if(self.expo != 0.0):
            self.pixelsHDR *= pow(2, self.expo)
        
    def generate(self, wk, config):
        # Load HDR file
        self.loadHDR(wk)
        if self.actions_load:
            self.pixelsHDR = self.actions_load.apply_hdr(wk, self, config)

        # Load LDR file
        if self.actions_load:
            self.im = self.actions_load.apply_ldr(self)
        else:
            if self.gamma != 1.0:
                self.pixelsHDR = np.power(self.pixelsHDR, 1.0 / self.gamma)
            pInt = np.clip(self.pixelsHDR * 255, 0, 255).astype(np.uint8)
            self.im = Image.fromarray(pInt)

        for action in self.actions:
            self.im = action.apply(self.im, self.pixelsHDR.shape[0], self.pixelsHDR.shape[1])

        # --- At the end, save it
        #print("Save "+self.output)
        logger.info("Save "+self.output)
        if self.high_res:
            self.im = self.im.resize((self.im.width * 4, self.im.height * 4), resample=1)
        #print("Saving: ", os.path.join(wk,config["exportDir"],self.output))
        self.im.convert("RGB").save(os.path.join(wk,config["exportDir"],self.output), quality=100, subsampling=0)


class ImageFalseColorOp(object):
    def __init__(self):

        # Magic numbers
        self.minV = 0.0
        self.maxV = 1.0
        self.cmap = cm.get_cmap("viridis")
    
    def readXML(self, n):
        if "min" in n.attrib:
            self.minV = float(n.attrib["min"])
        if "max" in n.attrib:
            self.maxV = float(n.attrib["max"])
    
        if "colormap" in n.attrib:
            self.cmap = cm.get_cmap(n.attrib["colormap"])

    def apply_hdr(self, wk, image, config):
        return image.pixelsHDR

    def apply_ldr(self, image):
        logger.debug("Complex load")
        
        data = np.mean(image.pixelsHDR, axis=2)
        
        # # --- Save the figure
        data = np.clip((data - self.minV) / (self.maxV - self.minV), 0, 1)
        data = self.cmap(data)
        data = np.clip(data * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(data)

class ImageFalseColorDiffOp(ImageFalseColorOp):
    def __init__(self):
        ImageFalseColorOp.__init__(self)
        self.ref = ""
        self.metric = None

    def readXML(self, n):
        ImageFalseColorOp.readXML(self, n)
        self.ref = n.attrib["ref"]
        if "metric" in n.attrib:
            self.metric = n.attrib["metric"]
        else:
            raise "'metric' need to be specified"
    def apply_hdr(self, wk, image, config):
        # --- Load reference
        ref = open_img(os.path.join(wk,config["inputDir"],self.ref))
        if(image.gamma != 1.0):
            raise Exception("Not possible to change gamma when computing false color image")
        
        if(image.expo != 0.0):
            ref *= pow(2, image.expo)

        test = image.pixelsHDR
        diff = np.array(ref - test)

        eps=1e-2
        diff = np.array(ref - test)
        if (self.metric == 'l1'):      # Absolute error
            error = np.abs(diff)
        elif (self.metric == 'l2'):    # Squared error
            error = diff * diff
        elif (self.metric == 'mrse'):  # Relative squared error
            error = diff * diff / (ref * ref + eps)
        elif (self.metric == 'mape'):  # Relative absolute error
            error = np.abs(diff) / (ref + eps)
        elif (self.metric == 'smape'):  # Symmetric absolute error
            error = 2 * np.abs(diff) / (ref + test + eps)
        else:
            raise "Unknow metric: %s".format(self.metric)
        
        print(" - Err {}  = {:.6f}".format(self.metric, np.mean(error)))

        return error

def readXMLComparisons(file):
    tree = ET.parse(file)
    root = tree.getroot()

    config = {
        "exportDir": ".",
        "inputDir" : ".",
        "exposure" : 0,
        "gamma"    : 2.2,
    }

    for c in root.iter("Config"):
        if "exportDir" in c.attrib:
            config["exportDir"] = c.attrib["exportDir"]
        if "inputDir" in c.attrib:
            config["inputDir"] = c.attrib["inputDir"]
        if "exposure" in c.attrib:
            config["exposure"] = float(c.attrib["exposure"])
        if "gamma" in c.attrib:
            config["gamma"] = float(c.attrib["gamma"])
    
    # Create directory if needed
    if not os.path.exists(config["exportDir"]):
        os.makedirs(config["exportDir"], exist_ok=True)

    images = []
    for imNode in root.iter('Image'):
        im = ImageOp()
        im.readXML(imNode, config)
        images.append(im)

    error_images = []
    for imNode in root.iter('ImageFalseColorDiff'):
        false_im = ImageFalseColorDiffOp()
        false_im.readXML(imNode)
        for im in images:
            im = copy.deepcopy(im)
            # Change the gamma
            im.gamma = 1.0
            im.output = im.output.replace(".jpg", f"_{false_im.metric}.jpg")
            im.actions_load = copy.deepcopy(false_im)
            error_images += [im]
    images += error_images

    crop_images = []
    for (i, cropNode) in enumerate(root.iter('Crop')):
        crop = CropOp()
        crop.readXML(cropNode)
        for im in images:
            im = copy.deepcopy(im)
            im.high_res = True
            im.output = im.output.replace(".jpg", f"_b{i+1}.jpg")
            im.actions.append(crop)
            crop_images.append(im)
    
    # Apply box on top of the image
    boxes = []
    for (i, cropNode) in enumerate(root.iter('Box')):
        crop = BoxOp()
        crop.readXML(cropNode)
        boxes += [crop]
    
    if len(boxes) > 0:
        for im in images:
            im = copy.deepcopy(im)
            im.output = im.output.replace(".png", f"_box.png")
            im.actions += boxes
            crop_images.append(im)

    images += crop_images

    displays = []
    for disNode in root.iter('DisplayMetric'):
        dis = MetricOp()
        dis.readXML(disNode)
        displays.append(dis)

    return images,displays,config

if __name__ == "__main__":
    # --- Read all params
    parser = optparse.OptionParser()
    parser.add_option('-i','--input', help='input')
    (opts, args) = parser.parse_args()

    inXML = opts.input
    wk = os.path.dirname(inXML)

    print(wk)
    images,displays,config = readXMLComparisons(inXML)

    while len(images) != 0:
        images[-1].generate(wk, config)
        images.pop() # Remove last one

    for display in displays:
        display.show(wk)



    if os.path.exists(os.path.join(config["inputDir"], "out.log")):
        log = open(os.path.join(config["inputDir"], "out.log")).readlines()
        isequaltime = False
        for l in log:
            if "INFO rustlight::integrators::equal_time - Number spp:" in l:
                isequaltime = True

        lines = []
        for l in log:
            if "Save final image:" in l:
                lines.append(os.path.basename(l.split(" ")[-1].replace("\n", "")))
            if "Elapsed Integrator:" in l:
                lines.append(" ".join(l.split(" ")[-2:]).replace("\n", ""))
            if "INFO rustlight::integrators::equal_time - Number spp:" in l:
                lines.append(" ".join(l.split(" ")[-1:]).replace("\n", ""))
        if isequaltime:
            for i in range(0, len(lines), 3):
                print(f" - {lines[i+2]} = {lines[i+1]} | {lines[i]} spp")
        else:
            for i in range(0, len(lines), 2):
                print(f" - {lines[i+1]} = {lines[i]}")