"""An evaluation module that retrieves test accuracies for quantized 1NN model.
Test Folder = for_evaluation(test_set)
Run evaluation model:
python3 Evaluation_1NN_GPU.py \
"""

# Import necessary modules/packages
import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
#import pyttsx3
import tflite_runtime.interpreter as tflite
import xml.etree.ElementTree as ET
import shutil
import time

shutil.rmtree("./mAP/groundtruths")
shutil.rmtree("./mAP/1NN_Coral_8bit_detections")
os.makedirs("./mAP/groundtruths")
os.makedirs("./mAP/1NN_Coral_8bit_detections")

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(len(scores)) if not np.isnan(class_ids[i])]#if scores[i] >= score_threshold]

# 박스 친거만 이미지에서 크롭하기
def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

def main():
    #efault_model_dir = './all_models'
    
    # Set model
    default_model = './1NN/quantized/one_nn11_edgetpu.tflite' # Coral ver
    # default_model = './1NN/quantized/one_nn11.tflite' # GPU ver
    default_labels = 'face_labels.txt' 

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default = default_model)                
    
    parser.add_argument('--labels', help='label file path',
                        default = default_labels)

    #parser.add_argument('--top_k', type=int, default=5,
    #                    help='number of categories with highest score to display')
    #parser.add_argument('--threshold', type=float, default=0.1,
    #                    help='classifier score threshold')
    args = parser.parse_args()

    # Load 1NN
    interpreter = tflite.Interpreter(model_path = args.model, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()

    # Load labels
    labels = load_labels(args.labels)
    # Load Test Data - ground truth, image
    test_dir = 'for_evaluation(test_set)/xml'
    test_img_dir = 'for_evaluation(test_set)/image'
    filenames = os.listdir(test_dir)
    full_filenames = []
    for filename in filenames:
        full_filename = os.path.join(test_dir, filename)
        full_filenames.append(full_filename)
    
    total_maskdetection_time = 0
    mask_detection_count = 0

    for filename in full_filenames:
        print(f'---------------------------', filename, '---------------------------')
        # get filenum
        filenum = filename[-9:-4]
        # filenum = filename.split('/')[2].split('.')[0]

        # set root from xml
        tree = ET.parse(filename)
        root = tree.getroot()

        # find img directory
        image_filename = root.find('filename').text
        image_path = os.path.join(test_img_dir, image_filename)

        # Load Image, get height and width
        cv2_im = cv2.imread(image_path,1)
        height, width, channels = cv2_im.shape

        # Get ground truths
        all = root.findall('object')
        ground_truths  = []
        for object in all:
            # get name, bndbox for labels and bbox
            name = object.find('name')
            bndbox = object.find('bndbox')

            # set test label to name.text (mask or nomask)
            test_label = name.text
            bbox = []
            for element in bndbox:
                bbox.append(int(element.text))
            xmin, ymin, xmax, ymax = bbox
            top_left, bottom_right = (xmin, ymax), (xmax, ymin)
            color = (0, 0, 255)
            thickness = 2
            #cv2.rectangle(cv2_im, top_left, bottom_right, color, thickness)
            test_bbox = [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height]

            ground_truths.append([test_label, test_bbox])
        
        print('ground_truths: ', ground_truths)

        for ground_truth in ground_truths:
            with open("./mAP/groundtruths/{}.txt".format(filenum), "a+") as file:
                file.write(str(ground_truth[0]) + ' ')
                for item in ground_truth[1]:
                    file.write("%s " % item)
                file.write("\n")

        # Evaluation of object detection
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        common.set_input(interpreter, pil_im)

        # Latency calculation
        mask_start_time = time.time()
        interpreter.invoke()
        mask_end_time = time.time()
        total_maskdetection_time += mask_end_time - mask_start_time
        mask_detection_count += 1
        
        objs = get_output(interpreter) # score_threshold=args.threshold, top_k=args.top_k)
        print('detection result:', objs)

        for i in range(len(objs)):
            if objs[i].id != 0 and objs[i].id != 1:
                continue
            if objs[i].score > 1:
                continue
            obj_bbox = list(objs[i].bbox)
            if any(edge > 1 for edge in obj_bbox):
                continue
            xmin, ymin, xmax, ymax = obj_bbox
            xmin, ymin, xmax, ymax = int(xmin*width), int(ymin*height), int(xmax*width), int(ymax*height)
            print(xmin, ymin, xmax, ymax)
            top_left, bottom_right = (xmin, ymax), (xmax, ymin)
            color = (255, 0, 0)
            thickness = 2
            #cv2.rectangle(cv2_im, top_left, bottom_right, color, thickness)

            if objs[i].id == 0:
                label = "nomask"
            elif objs[i].id == 1:
                label = "mask"
            score = objs[i].score
            print(obj_bbox, label, score)

            with open("./mAP/1NN_Coral_8bit_detections/{}.txt".format(filenum), "a+") as file:
                file.write(label + ' ')
                file.write(str(score) + ' ')
                for item in obj_bbox:
                    file.write("%s " % item)
                file.write("\n")

        window_name = 'image'
        # cv2.imshow(window_name, cv2_im)
        # cv2.waitKey()

    avg_mask = total_maskdetection_time/mask_detection_count
    print('Average Total Inference Time: ', avg_mask)

if __name__ == '__main__':
    main()