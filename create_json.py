import json
import numpy as np
import os

global count
count = 1

for j in range(2):
    if j == 0:
        path_label = './COCO/label_t/'
        path_save = './COCO/annotations/instances_train2017.json'
    else:
        path_label = './COCO/label_v/'
        path_save = './COCO/annotations/instances_val2017.json'
    
    file_list = os.listdir(path_label)
    dataset = {'categories': [], 'annotations': [], 'images': []}
    # label name
    with open('custom.txt') as f:
        content = f.read()

    for class_i, class_name in enumerate(content.split(sep='\n')):
        if class_name != '':
            dataset['categories'].append({'id':class_i + 1,'name':class_name,'supercategory':'RoadSign'})

    for file in file_list:
        dataset['images'].append({'filename':file[:6]+'.jpg', 'id':int(file[:6]), 'width':1920, 'height':1200})

    width = 1920 # 사진 너비
    height = 1200 # 사진 높이
    for file in file_list:
        with open(path_label + file) as f:
            annos = f.readlines()
            image_id = int(file[:6])
            for i, anno in enumerate(annos):
                parts = anno.strip().split()
                cls_id = int(parts[0]) + 1 # class_id는 1부터 시작
                x_min = (np.round(float(parts[1]), 3) - (1/2 * np.round(float(parts[3]), 3))) * width
                x_max = (np.round(float(parts[1]), 3) + (1/2 * np.round(float(parts[3]), 3))) * width 
                y_min = (np.round(float(parts[2]), 3) - (1/2 * np.round(float(parts[4]), 3))) * height
                y_max = (np.round(float(parts[2]), 3) + (1/2 * np.round(float(parts[4]), 3))) * height
                width_b = max(0, x_max - x_min)
                height_b = max(0, y_max - y_min)
                dataset['annotations'].append({
                    'area':width_b * height_b,
                    'bbox':[x_min, y_min, width_b, height_b],
                    'category_id':cls_id,
                    'id':count,
                    'image_id':image_id,
                    'iscrowd':0,
                    'segmentation':[[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]
                })
                count += 1

    json_name = path_save
    with open(json_name, 'w') as f:
        json.dump(dataset, f, indent=4)