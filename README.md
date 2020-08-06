# Mask detection with Coral
This python codes run on Google Coral dev board and detect whether any person wears a mask or not
from video stream. The detection result is shown on connected monitor.
The codes are modified from Google's default sample face detection codes, using opencv:
https://github.com/google-coral/examples-camera/tree/master/opencv

## Changes made from Google's default sample
1. Mask detection neural network is added

    We trained mask detection neural network and added another inference for this neural network. I will call this "MaskNet" from now on.
    "Interpreter2" in the codes runs the inference for this neural network.
    Basically, after the face detection neural network('mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite') is inferenced, MaskNet is run on the resulting faces.
    This neural network outputs binary classification result: Mask or NoMask.
    
    If you want to see the NN training codes see "train_mask_detector" folder.
    
2. Voice output is added

    We added voice output so that Coral can warn if there is anybody not wearing a mask.
    We used PyGame module for this.
    
3. Some funcions are added in common.py

    We added some functions for the inference of MaskNet in common.py.


## How to run on laptop
There are some parts in the code you need to change if you want to run this code on your laptop
(WHere there is no EdgeTPU).
Find below code blocks in the main() function, and modify as below.

```
#### In order to run on Laptop, tflite file before edgetpu compile should be used ###
#default_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'  #
default_model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'    

#default_model2 = 'mask_detector_quant_edgetpu.tflite'
default_model2 = 'mask_detector_quant.tflite'    
#####################################################################################    
```
    
And 
    
```
### Some functions in common.make_interpreter needs Edge TPU ########################
### Simply use tflite.Interpreter method on laptop
#interpreter = common.make_interpreter(args.model)
interpreter = tflite.Interpreter(model_path = args.model)
interpreter.allocate_tensors()

#interpreter2 = common.make_interpreter(args.model2)
interpreter2 = tflite.Interpreter(model_path = args.model2)
interpreter2.allocate_tensors()
#####################################################################################  
```




Below is original description from Google.

# OpenCV camera examples with Coral

This folder contains example code using [OpenCV](https://github.com/opencv/opencv) to obtain
camera images and perform object detection on the Edge TPU.

This code works on Linux using a webcam, Raspberry Pi with the Pi Camera, and on the Coral Dev
Board using the Coral Camera or a webcam. For the first two, you also need a Coral
USB/PCIe/M.2 Accelerator.


## Set up your device

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)). You can check which version is installed
    using the ```pip3 show tflite_runtime``` command.

2.  Clone this Git repo onto your computer or Dev Board:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/google-coral/examples-camera --depth 1
    ```

3.  Download the models:

    ```
    cd examples-camera

    sh download_models.sh
    ```

4.  Install the OpenCV libraries:

    ```
    cd opencv

    bash install_requirements.sh
    ```


## Run the detection demo (SSD models)

```
python3 detect.py
```

By default, this uses the ```mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite``` model.

You can change the model and the labels file using flags ```--model``` and ```--labels```.


# Quantization
You got to have .tflite file to quantize and the environment of Jupyter Terminal is needed(if other method found, please tell us so that we can share!). Our reference [here](https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_classification_ptq_tf1.ipynb#scrollTo=joxrIB0I3cdi).  



1. Go to Directory that .tflite is in.  
```
cd [your_dir_that_model_exists]

# example #
cd coral_mask
```

2. Run Below command **line by line** .It is the process of installing 'edgetpu-compiler'

```
! curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

! echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

! sudo apt-get update

! sudo apt-get install edgetpu-compiler	
```

3. And Finally Quantize the model

```
! edgetpu_compiler [your_model_name].tflite

# example #
! edgetpu_compiler mask_detector_quant_v2.tflite
```

4. In the same dir, name is the same but '_edgetpu' added will be seen. Here we can see file named 'mask_detector_quant_v2 **_edgetpu**.tflite'

5. Deploy the ~_edgetpu.tflite file together to your Coral device, so that Coral can utilize the quantized tflite model.


# Coral Run
Run command is below

```
edgetpu_detect_server \
--model ~/coral_mask/1NN/mask_model_export_tflite6/one_nn_edgetpu.tflite \
--labels ~/coral_mask/1NN/mask_model_export_tflite6/one_nn_label.txt
```