# detect damages in images with mask rcnn model
# for more information checkout https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/
import os
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from numba import jit, cuda

# class that defines and loads the hurricane dataset
class HurricaneDataset(Dataset):
	# load the dataset definitions
	#@jit(target ="cuda")
	def load_dataset(self, dataset_dir, is_train=True):
		# define classes
		self.add_class("dataset", 1, "no-damage-small-structure")
		self.add_class("dataset", 2, "lightly-damaged-small-structure")
		self.add_class("dataset", 3, "moderately-damaged-small-structure")
		self.add_class("dataset", 4, "heavily-damaged-small-structure")
		self.add_class("dataset", 5, "lightly-damaged-medium-building")
		self.add_class("dataset", 6, "moderately-damaged-medium-building")
		self.add_class("dataset", 7, "heavily-damaged-medium-building")
		self.add_class("dataset", 8, "no-damage-large-building")
		self.add_class("dataset", 9, "lightly-damaged-large-building")
		self.add_class("dataset", 10, "moderately-damaged-large-building")
		self.add_class("dataset", 11, "heavily-damaged-large-building")
		self.add_class("dataset", 12, "residential-building")
		self.add_class("dataset", 13, "commercial-building")
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
		boxes = list()
		# extract each bounding box
		for box in root.findall('.//bndbox'):
			xmin = int(float(box.find('xmin').text))
			ymin = int(float(box.find('ymin').text))
			xmax = int(float(box.find('xmax').text))
			ymax = int(float(box.find('ymax').text))
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	#@jit(target ="cuda")
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			if self.class_names.index('no-damage-small-structure'):
				class_ids.append(self.class_names.index('no-damage-small-structure'))
			elif self.class_names.index('lightly-damaged-small-structure'):
				class_ids.append(self.class_names.index('lightly-damaged-small-structure'))
			elif self.class_names.index('moderately-damaged-small-structure'):
				class_ids.append(self.class_names.index('moderately-damaged-small-structure'))
			elif self.class_names.index('heavily-damaged-small-structure'):
				class_ids.append(self.class_names.index('heavily-damaged-small-structure'))
			elif self.class_names.index('lightly-damaged-medium-building'):
				class_ids.append(self.class_names.index('lightly-damaged-medium-building'))
			elif self.class_names.index('moderately-damaged-medium-building'):
				class_ids.append(self.class_names.index('moderately-damaged-medium-building'))
			elif self.class_names.index('heavily-damaged-medium-building'):
				class_ids.append(self.class_names.index('heavily-damaged-medium-building'))
			elif self.class_names.index('no-damage-large-building'):
				class_ids.append(self.class_names.index('no-damage-large-building'))
			elif self.class_names.index('lightly-damaged-large-building'):
				class_ids.append(self.class_names.index('lightly-damaged-large-building'))
			elif self.class_names.index('moderately-damaged-large-building'):
				class_ids.append(self.class_names.index('moderately-damaged-large-building'))
			elif self.class_names.index('heavily-damaged-large-building'):
				class_ids.append(self.class_names.index('heavily-damaged-large-building'))
			elif self.class_names.index('residential-building'):
				class_ids.append(self.class_names.index('residential-building'))
			elif self.class_names.index('commercial-building'):
				class_ids.append(self.class_names.index('commercial-building'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define a configuration for the model
class HurricaneConfig(Config):
	# define the name of the configuration
	NAME = "debris_model_cfg"
	# number of classes (background + structures)
	NUM_CLASSES = 1 + 13
	# number of training steps per epoch
	STEPS_PER_EPOCH = 79

# load the train dataset
train_set = HurricaneDataset()
train_set.load_dataset('train_data', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = HurricaneDataset()
test_set.load_dataset('train_data', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# prepare config
config = HurricaneConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

import datetime
import pickle

x = datetime.datetime.now()
Pkl_Filename = "keras-model-"+str(x)+".pkl"
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)

# # define the prediction configuration
# class PredictionConfig(Config):
# 	# define the name of the configuration
# 	NAME = "debris_predict_cfg"
# 	# number of classes (background + structures)
# 	NUM_CLASSES = 1 + 13
# 	# simplify GPU config
# 	GPU_COUNT = 1
# 	IMAGES_PER_GPU = 1

# # plot a number of photos with ground truth and predictions
# def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
# 	# load image and mask
# 	for i in range(n_images):
# 		# load the image and mask
# 		image = dataset.load_image(i)
# 		mask, _ = dataset.load_mask(i)
# 		# convert pixel values (e.g. center)
# 		scaled_image = mold_image(image, cfg)
# 		# convert image into one sample
# 		sample = expand_dims(scaled_image, 0)
# 		# make prediction
# 		yhat = model.detect(sample, verbose=0)[0]
# 		# define subplot
# 		pyplot.subplot(n_images, 2, i*2+1)
# 		# plot raw pixel data
# 		pyplot.imshow(image)
# 		pyplot.title('Actual')
# 		# plot masks
# 		for j in range(mask.shape[2]):
# 			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
# 		# get the context for drawing boxes
# 		pyplot.subplot(n_images, 2, i*2+2)
# 		# plot raw pixel data
# 		pyplot.imshow(image)
# 		pyplot.title('Predicted')
# 		ax = pyplot.gca()
# 		# plot each box
# 		for box in yhat['rois']:
# 			# get coordinates
# 			y1, x1, y2, x2 = box
# 			# calculate width and height of the box
# 			width, height = x2 - x1, y2 - y1
# 			# create the shape
# 			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
# 			# draw the box
# 			ax.add_patch(rect)
# 	# show the figure
# 	pyplot.show()

# # create config
# cfg = PredictionConfig()
# # define the model
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# # load model weights
# model_path = 'mask_rcnn_kangaroo_cfg_0005.h5'
# model.load_weights(model_path, by_name=True)
# # plot predictions for train dataset
# plot_actual_vs_predicted(train_set, model, cfg)
# # plot predictions for test dataset
# plot_actual_vs_predicted(test_set, model, cfg)

# # enumerate all images in the dataset, just for testing sake
# for image_id in train_set.image_ids:
# 	# load image info
# 	info = train_set.image_info[image_id]
# 	# display on the console
# 	print(info)

# #load an image
# image_id = 75
# image = train_set.load_image(image_id)
# print(image.shape)
# # load image mask
# mask, class_ids = train_set.load_mask(image_id)
# print(mask.shape)
# # plot image
# pyplot.imshow(image)
# # plot mask
# pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
# pyplot.show()

# from mrcnn.visualize import display_instances
# from mrcnn.utils import extract_bboxes
# image_id = 0
# # load the image
# image = train_set.load_image(image_id)
# # load the masks and the class ids
# mask, class_ids = train_set.load_mask(image_id)

# # print(image.shape)
# # print(mask.shape[-1])
# # print(class_ids.shape[0])
# # extract bounding boxes from the masks
# bbox = extract_bboxes(mask)
# # display image with masks and bounding boxes
# display_instances(image, bbox, mask, class_ids, train_set.class_names)