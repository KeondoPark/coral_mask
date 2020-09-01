import json
import os

files = sorted(os.listdir('../2NN_CPU_8bit_detections'))

res = []
for file in files:
    f = open('../2NN_CPU_8bit_detections/{}'.format(file), "r")
    lines = f.readlines()
    for line in lines:
        data = {}
        ls = list(line.split(' '))
        data["image_id"] = int(file[0:5])
        if ls[0] == 'mask':
            data["category_id"] = 0
        else:
            data["category_id"] = 1
        # bbox width, height adjust
        data["bbox"] = [int(float(ls[2])), int(float(ls[3])), int(float(ls[4]) - float(ls[2])), int(float(ls[5]) - float(ls[3]))]
        data["score"] = float(ls[1])
        res.append(data)

with open('2NN_CPU_8bit_res.txt', 'w') as outfile:
    json.dump(res, outfile)