"""
This code iterates through a directory of jpgs and xml annotations to crop images into a grid of photos in the specified size. 
Each new image has the tile locations appended to the ends of the original image names.

Example:
    Original Image     "C24852789.jpg"
    New Image          "C24852789-0-5.jpg"     *This new image is located in the 1st column and 6th row of the image grid.
    
Inputs:
    source_image_dir: Directory where source images are stored (ex. 'sample_images_dir')
    source_annotation_dir: Directory where source annotations are stored (ex. 'sample_annotations_dir')
    dest_dir: Destination directory for output (ex. 'sample_cropped_dir')
    original_width: The width of images to be cropped (ex. 10328)
    original_height: The height of images to be cropped (ex. 7760)
    crop_width: The width of the resulting cropped images (ex. 1000)
    crop_height: The height of the resulting cropped images (ex. 1000)

Sample Run: python crop_images.py 'sample_images_dir' 'sample_annotations_dir' 'sample_cropped_dir' 10328 7760 1000 1000
"""

import os
from PIL import Image
import xml.etree.ElementTree as ET
import sys
import math

def crop_annotated_images(source_image_dir, source_annotation_dir, dest_dir, original_width, original_height, crop_width, crop_height):

    counter_int = 1

    for image in os.listdir(source_image_dir):
        if 'jpg' in image:

            # if counter_int%2==0:
            #     print("Processing image:", counter_int)
            print("Processing image:", counter_int)

            counter_int+=1

            try:
                basename =  os.path.splitext(image)[0]

                crop_width_index = range(0,math.floor(original_width/crop_width)) 
                crop_height_index = range(0,math.floor(original_height/crop_height))

                for width_int in crop_width_index:
                    for height_int in crop_height_index:

                        tree = ET.parse(source_annotation_dir + '/' + basename + '.xml')
                        root = tree.getroot()

                        start_x = float(crop_width * width_int)
                        end_x = float(start_x + crop_height)
                        start_y = float(crop_width * height_int)
                        end_y = float(start_y + crop_height)

    #                     print("image:", image)
    #                     print("Image size:", start_x, end_x, start_y, end_y)

                        root.find('filename').text = str(basename+'-'+str(width_int)+'-'+str(height_int)+'.jpg')

                        for item in root.iter('size'):
                            item.find('width').text = str(crop_width)
                            item.find('height').text = str(crop_height)

                        for boxes in root.findall('object'):

                            ymin, xmin, ymax, xmax = None, None, None, None
                            box = boxes.find('bndbox')
                            xmin = float(box.find('xmin').text)
                            ymin = float(box.find('ymin').text)
                            xmax = float(box.find('xmax').text)
                            ymax = float(box.find('ymax').text)
    #                         print('Original:', xmin, xmax, ymin, ymax)

                            if xmax <= start_x-75 or xmin-75 >= end_x or ymin-75 >= end_y or ymax <= start_y-75:
                                root.remove(boxes)
    #                             print("Removed.")
                            else:
                                altered_xmin = max(0, xmin - start_x)
                                altered_xmax = min(crop_width,xmax - start_x)
                                altered_ymin = max(0, ymin - start_y)
                                altered_ymax = min(crop_height,ymax - start_y)
    #                             print('Altered:', altered_xmin, altered_xmax, altered_ymin, altered_ymax)

                                box.find('xmin').text = str(altered_xmin)
                                box.find('ymin').text = str(altered_ymin)
                                box.find('xmax').text = str(altered_xmax)
                                box.find('ymax').text = str(altered_ymax)

                        result = len(root.findall('object'))
    #                     print("result", result)
                        if result > 0:
                            tree.write(open(dest_dir+'/'+basename+'-'+str(width_int)+'-'+str(height_int)+'.xml', 'wb'))
                            image_open = Image.open(source_image_dir+'/'+image) 
                            image_cropped = image_open.crop((start_x, start_y, end_x, end_y))
    #                         print("start_x, start_y, end_x, end_y", start_x, start_y, end_x, end_y)
    #                         print("Save to:", dest_dir+'/'+basename+'-'+str(width_int)+'-'+str(height_int)+'.jpg')
                            image_cropped.save(dest_dir+'/'+basename+'-'+str(width_int)+'-'+str(height_int)+'.jpg')

    #                     print()

            except Exception as inst:
                print(inst)

    print("Cropping process complete.")

if __name__ == "__main__":

    print("Reading inputs...")

    source_image_dir = sys.argv[1]
    source_annotation_dir = sys.argv[2]
    dest_dir = sys.argv[3]
    original_width = int(sys.argv[4])
    original_height = int(sys.argv[5])
    crop_width = int(sys.argv[6])
    crop_height = int(sys.argv[7])

    print("Starting cropping process...")
    crop_annotated_images(source_image_dir, source_annotation_dir, dest_dir, original_width, original_height, crop_width, crop_height)
