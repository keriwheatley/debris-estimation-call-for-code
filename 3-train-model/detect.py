import skimage.io
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances
import os
import sys
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np
import cv2
import json

class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "debris_model_cfg"
	# number of classes (background + structures)
	NUM_CLASSES = 1 + 14
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

def save_drawn_result(image, result, class_names, show_mask=True, show_bbox=True, save_dir=None, img_name=None):
    colors = color_map()
    for i in range(result['rois'].shape[0]):
        color = colors[result['class_ids'][i]].astype(np.int).tolist()
        if show_bbox:
            coordinate = result['rois'][i]
            cls = class_names[result['class_ids'][i]]
            score = result['scores'][i]
            cv2.rectangle(image, (coordinate[1], coordinate[0]), (coordinate[3], coordinate[2]), color, 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, '{}: {:.3f}'.format(cls, score), (coordinate[1], coordinate[0]), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        if show_mask:
            mask = result['masks'][:, :, i]
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.int)
            color_mask[mask] = color
            image = cv2.addWeighted(color_mask, 0.5, image.astype(np.int), 1, 0)
        cv2.imwrite(os.path.join(save_dir, img_name), image)
        return image


"""This is the function to take in a new input image for detection.
Inputs are the 'image path', 'model directory' and the 'saved model weights (.h5 file)'.
Output is the image with the detections and it returns the bounding boxes, their class ids and the confidence scores.
"""


def detect_damage(imagepath, model_directory, model_weights, save_directory):
    image_id = imagepath.split('/')[-1]
    cfg = PredictionConfig()
    cfg.display()
    model = MaskRCNN(mode='inference', config=cfg, model_dir=model_directory)
    model.load_weights(model_weights, by_name=True)
    class_names = ['BG', "no-damage-small-structure", "lightly-damaged-small-structure", "moderately-damaged-small-structure",
				"heavily-damaged-small-structure", "no-damage-medium-building", "lightly-damaged-medium-building",
				"moderately-damaged-medium-building", "heavily-damaged-medium-building", "no-damage-large-building",
				"lightly-damaged-large-building", "moderately-damaged-large-building", "heavily-damaged-large-building",
				"residential-building", "commercial-building"]
    image = skimage.io.imread(imagepath)
    result = model.detect([image], verbose=1)[0]
    #display_instances(image, result['rois'], result['masks'],result['class_ids'], class_names, result['scores'], title="Predictions")
    save_drawn_result(image, result, class_names, save_dir=save_directory, img_name="annotated_"+image_id)
    result_dict = {'boxes': result['rois'].tolist(), 'class_ids': result['class_ids'].tolist(), 'scores': result['scores'].tolist()}
    json.dump(result_dict, open(save_directory+"/"+image_id+".json", "w"), indent=6)
    return result_dict

imgpath = sys.argv[1]
modeldir = sys.argv[2]
modelweights = sys.argv[3]
savedir = sys.argv[4]
detect_damage(imgpath, modeldir, modelweights, savedir)   
