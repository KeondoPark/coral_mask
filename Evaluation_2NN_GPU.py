"""An evaluation module that retrieves test accuracies for quantized 2NN model.
Test Images: test_img, Test XML: test_xml
Run evaluation model:
python3 detect.py \
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

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
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

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

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

# This part will run when module is invoked
def main():
    default_model_dir = './all_models'
    
    # Set face detection model
    # default_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite' # Coral ver
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite' # GPU ver
    default_labels = 'coco_labels.txt'  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default = default_model)                
    
    # Set mask classification model
    default_model2 = 'mask_detector_quant.tflite' # GPU ver
    #default_model2 = 'mask_detector_quant_edgetpu.tflite' #Coral ver
    parser.add_argument('--model2', help='.tflite model path',
                        default=default_model2)
    
    parser.add_argument('--labels', help='label file path',
                        default = default_labels)

    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()
    
    # Load 1NN
    interpreter = tflite.Interpreter(model_path = args.model)
    interpreter.allocate_tensors()
    
    # Load 2NN
    interpreter2 = tflite.Interpreter(model_path = args.model2)
    interpreter2.allocate_tensors()

    # Load labels
    labels = load_labels(args.labels)
    
    # Load Test Data - ground truth, image
    test_dir = 'test_xml'
    test_img_dir = 'test_img'
    filenames = os.listdir(test_dir)
    full_filenames = []
    for filename in filenames:
        full_filename = os.path.join(test_dir, filename)
        full_filenames.append(full_filename)
    
    for filename in full_filenames:
        tree = ET.parse(filename)
        root = tree.getroot()
        for child in root:
            test_label = ''
            bbox = []
            if child.tag == 'filename':
                image_filename = child.text
                print('------------------------------', image_filename, '----------------------------------------------------------')
            if child.tag == 'size':
                width_height = []
                for subchild in child:
                    width_height.append(int(subchild.text))
            if child.tag == 'object':
                for subchild in child:
                    if subchild.tag == 'name':
                        test_label = subchild.text
                    if subchild.tag == 'bndbox':
                        for subsubchild in subchild:
                            bbox.append(int(subsubchild.text))
        # final_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]                    
        final_bbox = [bbox[0]/width_height[0], bbox[1]/width_height[1], bbox[2]/width_height[0], bbox[3]/width_height[1]]
        print(final_bbox)
        image_path = os.path.join(test_img_dir, image_filename)
        
        cv2_im = cv2.imread(image_path,1)

        # Evaluation of object detection
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        common.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
        print(objs)
        filenum_xml = filename.split('\\')[1]
        filenum = filenum_xml.split('.')[0]
        labenc = objs[0].id
        if labenc == 0:
            lab = "mask"
        else:
            lab = "nomask"

        print(filenum)
        with open("./mAP/input/ground-truth/{}.txt".format(filenum), "w") as file:   
            file.write(str(test_label) + ' ')
            for item in final_bbox:
                file.write("%s " % item)
        print(test_label)
        with open("./mAP/input/detection-results/{}.txt".format(filenum), "w") as file:
            file.write(lab + ' ')
            file.write(str(objs[0].score) + ' ')
            for item in objs[0].bbox:
                file.write("%s " % item)

        height, width, channels = cv2_im.shape
        
        mask_data = []

        for i in range(len(objs)):
            x0, y0, x1, y1 = list(objs[i].bbox)
            x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)                        
            pil_im2 = Image.fromarray(cv2_im_rgb[y0:y1, x0:x1])            
            common.set_input2(interpreter2, pil_im2)
            output_data = common.output_tensor2(interpreter2)
            interpreter2.invoke()
            if objs[i].id == 0:                   
                mask_data.append(('mask_percentage', output_data))
            
        print(mask_data)
        
        print('-------------------------------next file----------------------------------------------------------')

if __name__ == '__main__':
    main()