import os
import sys
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "debris_model_cfg"
	# number of classes (background + structures)
	NUM_CLASSES = 1 + 14
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
import skimage.io
def detect_damage(imagepath, model_directory, model_weights):
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
	display_instances(image, result['rois'], result['masks'],
					result['class_ids'], class_names, result['scores'], title="Predictions")
	return {'boxes': result['rois'], 'class_ids': result['class_ids'], 'scores': result['scores']}

imgpath = sys.argv[1]
modeldir = sys.argv[2]
modelweights = sys.argv[3]
detect_damage(imgpath, modeldir, modelweights)   