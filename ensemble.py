import json
import os
from collections import defaultdict

model_list = os.listdir('./submission/ensemble')
model_list

model_json = defaultdict(list)
for model in model_list:
    with open('./submission/ensemble/' + model) as f:
        model_json[model[:-5]] = json.load(f)

def compare_point(result, compare, frame_id, overlap = 2):
    skip_loop = []
    Object_id = 0
    for k in range(len(compare)):
        if k in skip_loop:
            continue
        count = 1
        width = compare[k]['x2'] - compare[k]['x1']
        height = compare[k]['y2'] - compare[k]['y1']

        for g in range(len(compare)):
            if k+g+1 > len(compare) - 1:
                break

            if k+g+1 in skip_loop:
                continue

            if (compare[k]['x1'] - (1/3 * width) <= compare[k+g+1]['x1'] <= compare[k]['x1'] + (1/3 * width)) and \
               (compare[k]['x2'] - (1/3 * width) <= compare[k+g+1]['x2'] <= compare[k]['x2'] + (1/3 * width)) and \
               (compare[k]['y1'] - (1/3 * height) <= compare[k+g+1]['y1'] <= compare[k]['y1'] + (1/3 * height)) and \
               (compare[k]['y2'] - (1/3 * height) <= compare[k+g+1]['y2'] <= compare[k]['y2'] + (1/3 * height)):
                count += 1
                skip_loop.append(k+g+1)

        if count >= overlap:
            #print('count loop')
            result['labels'].append({
                'sensor_id':'front_camera',
                'level1_category':'moving_object',
                'level2_category':'vehicle',
                'frame_id':frame_id,
                'Object_id':Object_id,
                'detection_type':'2d_bounding_box',
                '2d_bounding_box':{
                    'x1':compare[k]['x1'],
                    'y1':compare[k]['y1'],
                    'x2':compare[k]['x2'],
                    'y2':compare[k]['y2']
                }
            })
            Object_id += 1

    return result

image_amount = 99
result = {'labels':[]}
for i in range(image_amount):
    compare = []
    for model in model_list:
        iterate = len(model_json[model[:-5]]['labels'])
        for j in range(iterate):
            label = model_json[model[:-5]]['labels']
            if int(label[j]['frame_id'][:-4]) == i:
                compare.append(label[j]['2d_bounding_box'])
    dataset = compare_point(result, compare, '{:06d}'.format(i) + '.png', 2)

json_name = 'HAD_A4_UC2_S1.json'
with open(json_name, 'w') as f:
    json.dump(dataset, f, indent = 4)