"""
This code iterates through a directory of jpgs and xml annotations to crop images into a grid of photos in the specified size.
"""

crop_annotated_images(source_image_dir='images_dir', source_annotation_dir='annotations_dir', 
                      dest_dir='cropped_dir', original_width=10328, original_height=7760, crop_width=crop_width, crop_height=1000)

def crop_annotated_images(source_image_dir, source_annotation_dir, dest_dir, original_width, original_height, crop_width, crop_height):

    import os
    from PIL import Image
    import xml.etree.ElementTree as ET
    import sys
    import math

    cropped_dir = dest_dir
    annotations_dir = source_annotation_dir
    images_dir = source_image_dir

    crop_width = crop_width
    crop_height = crop_height

    for image in os.listdir(images_dir):
        if 'jpg' in image:

            try:
                basename =  os.path.splitext(image)[0]

                original_image_width = original_width
                original_image_height = original_height

                crop_width_index = range(0,math.floor(original_image_width/crop_width)) 
                crop_height_index = range(0,math.floor(original_image_height/crop_height))

                for width_int in crop_width_index:
                    for height_int in crop_height_index:

                        tree = ET.parse(annotations_dir + '/' + basename + '.xml')
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

                        counter_int = 1
                        for boxes in root.findall('object'):
    #                         print(counter_int)
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

                            counter_int+=1

                        result = len(root.findall('object'))
    #                     print("result", result)
                        if result > 0:
                            tree.write(open(cropped_dir+'/'+basename+'-'+str(width_int)+'-'+str(height_int)+'.xml', 'wb'))
                            image_open = Image.open(images_dir+'/'+image) 
                            image_cropped = image_open.crop((start_x, start_y, end_x, end_y))
    #                         print("start_x, start_y, end_x, end_y", start_x, start_y, end_x, end_y)
    #                         print("Save to:", cropped_dir+'/'+basename+'-'+str(width_int)+'-'+str(height_int)+'.jpg')
                            image_cropped.save(cropped_dir+'/'+basename+'-'+str(width_int)+'-'+str(height_int)+'.jpg')

    #                     print()

            except Exception as inst:
                print(inst)
