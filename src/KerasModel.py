# detect damages in images with mask rcnn model
# for more information checkout https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/
import os
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import mean
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

class HurricaneDataset(Dataset):
	# load the dataset definitions
	#@jit(target ="cuda")
	def load_dataset(self, dataset_dir, is_train=True):
		# define classes
		self.add_class("dataset", 1, "no-damage-small-structure")
		self.add_class("dataset", 2, "lightly-damaged-small-structure")
		self.add_class("dataset", 3, "moderately-damaged-small-structure")
		self.add_class("dataset", 4, "heavily-damaged-small-structure")
		self.add_class("dataset", 5, "no-damage-medium-building")
		self.add_class("dataset", 6, "lightly-damaged-medium-building")
		self.add_class("dataset", 7, "moderately-damaged-medium-building")
		self.add_class("dataset", 8, "heavily-damaged-medium-building")
		self.add_class("dataset", 9, "no-damage-large-building")
		self.add_class("dataset", 10, "lightly-damaged-large-building")
		self.add_class("dataset", 11, "moderately-damaged-large-building")
		self.add_class("dataset", 12, "heavily-damaged-large-building")
		self.add_class("dataset", 13, "residential-building")
		self.add_class("dataset", 14, "commercial-building")
		# define data locations
		images_dir = dataset_dir + '/Images/'
		annotations_dir = dataset_dir + '/Annotations/'
		image_count = 0
		#helps us get the number of images
		file_count = sum(len(files) for _, _, files in os.walk(images_dir))

		for filename in listdir(images_dir):
			# extract image id
			image_id = filename.split('.')[0]
			image_count += 1
			# skip all images after 80%, if we are building the train set
			if is_train and int(image_count) >= int(0.8 * file_count):
				continue
			# skip all images before 80%, if we are building the test/val set
			if not is_train and int(image_count) < int(0.8 * file_count):
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# load all bounding boxes for an image
	#@jit(target ="cuda")
	def extract_boxes(self, filename):
		# load and parse the file
		root = ElementTree.parse(filename)
		boxes, labels = list(), list()
		# extract each bounding box
		for box in root.findall('.//bndbox'):
			xmin = int(float(box.find('xmin').text))
			ymin = int(float(box.find('ymin').text))
			xmax = int(float(box.find('xmax').text))
			ymax = int(float(box.find('ymax').text))
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		for label in root.findall('object'):
			labels.append(label.find('name').text)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height, labels

	# load the masks for an image
	#@jit(target ="cuda")
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h, labels = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			label = labels[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index(label))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

"""Load the train dataset"""

train_set = HurricaneDataset()
train_set.load_dataset('train_data_small', is_train=True)
train_set.prepare()
trainlength = len(train_set.image_ids)
print('Train: %d' % trainlength)

"""Load the test dataset"""

test_set = HurricaneDataset()
test_set.load_dataset('train_data_small', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

"""Define a Configuration for the model"""

class HurricaneConfig(Config):
	# define the name of the configuration
	NAME = "debris_model_cfg"
	# number of classes (background + structures)
	NUM_CLASSES = 1 + 14
	# number of training steps per epoch
	#Should depend on training size
	STEPS_PER_EPOCH = trainlength

"""Enumerating all images in the dataset.
Just for testing sake.
"""

# for image_id in train_set.image_ids:
# 	info = train_set.image_info[image_id]
# 	print(info)

"""Load just one image and show a plot of it with the bounding boxes."""

image_id = 5
image = train_set.load_image(image_id)
print(image.shape)
mask, class_ids = train_set.load_mask(image_id)
print(mask.shape)
pyplot.imshow(image)
pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
pyplot.show()

"""Display image with masks and bounding boxes"""

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
image_id = 5
image = train_set.load_image(image_id)
mask, class_ids = train_set.load_mask(image_id)
bbox = extract_bboxes(mask)
display_instances(image, bbox, mask, class_ids, train_set.class_names)

"""Prepare config
Load weights
Train model
"""

debrisconfig = HurricaneConfig()
debrisconfig.display()
model = MaskRCNN(mode='training', config=debrisconfig, model_dir='./')
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model.train(train_set, test_set, learning_rate=debrisconfig.LEARNING_RATE, epochs=5, layers='heads')

"""Save summary in pickle and text files, Save model weights."""

import pickle
import io
if os.path.exists("model_summary.pkl") == False:
	open("model_summary.pkl", 'w').close
stream = io.StringIO()
model.keras_model.summary(print_fn=lambda x: stream.write(x + '\n'))
model_summary = stream.getvalue()
pickle.dump(model_summary, open("model_summary.pkl", 'wb'))
stream.close()
model.keras_model.save_weights("model.h5")
print("Saved model summary and weights to disk")

with open('model_summary.txt','w') as fh:
    model.keras_model.summary(print_fn=lambda x: fh.write(x + '\n'))
fh.close

"""Define the prediction configuration"""

class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "debris_model_cfg"
	# number of classes (background + structures)
	NUM_CLASSES = 1 + 14
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

"""Calculate the mean average precision (mAP) for a model on a given dataset."""

def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

"""Plot a number of images with ground truth and predictions."""

def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		pyplot.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		# get the context for drawing boxes
		pyplot.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Predicted')
		ax = pyplot.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	pyplot.show()

"""Evaluate mask rcnn model on training and test datasets."""

cfg = PredictionConfig()
model = MaskRCNN(mode='inference', config=cfg, model_dir='./')
model.load_weights('model.h5', by_name=True)
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)

# Save model to JSON
import json
model_json = model.keras_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()

plot_actual_vs_predicted(train_set, model, cfg)
plot_actual_vs_predicted(test_set, model, cfg)