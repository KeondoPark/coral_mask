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
import pathlib

shutil.rmtree("./mAP/groundtruths")
shutil.rmtree("./mAP/1NN_CPU_8bit_detections")
os.makedirs("./mAP/groundtruths")
os.makedirs("./mAP/1NN_CPU_8bit_detections")

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
    # default_model = './1NN/quantized/one_nn11_edgetpu.tflite' # Coral ver
    default_model = './1NN/quantized/one_nn_batch_8.tflite' # GPU ver
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
    interpreter = tflite.Interpreter(model_path = args.model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    # Load labels
    labels = load_labels(args.labels)
    # Load Test Data - ground truth, image
    test_dir = 'for_evaluation(test_set)/test_xml'
    test_img_dir = 'for_evaluation(test_set)/test_image'
    filenames = os.listdir(test_dir)
    full_filenames = []
    for filename in filenames:
        full_filename = os.path.join(test_dir, filename)
        full_filenames.append(full_filename)
    
    total_maskdetection_time = 0
    mask_detection_count = 0

    images = []

    for index, file in enumerate(pathlib.Path('for_evaluation(test_set)/image').iterdir()):
        
        # read and resize the image
        cv2_im = cv2.imread(r"{}".format(file.resolve()))
        #cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        #pil_im = Image.fromarray(cv2_im_rgb)

        pil_im = cv2.resize(cv2_im, (320, 320))
        
        images.append(pil_im)
        
        # resize the input tensor
        if index == 0:
            new_images = np.array(images, dtype=np.uint8)
            print(new_images.shape)
            interpreter.resize_tensor_input(input_details[0]['index'],[len(images), 320, 320, 3])
            print(interpreter.get_input_details())
            interpreter.resize_tensor_input(output_details[0]['index'], [len(images), 1])
            print(interpreter.get_output_details())
            interpreter.allocate_tensors()
            #common.set_input(interpreter, images)
            interpreter.set_tensor(input_details[0]['index'], images)
            # run the inference
            interpreter.invoke()
            # output_details[0]['index'] = the index which provides the input
            output_data = interpreter.get_tensor(output_details[0]['index'])
            # clear the list
            images.clear()

if __name__ == '__main__':
    main()