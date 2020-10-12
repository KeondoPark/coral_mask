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

## Changes(2020.10.12)
Training and Test images are removed.
Old tflite models are deleted, they were used for test purpose.

---------------------------------------------------------------------------------------------

*Below is original description from Google.*

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

## Bazel



> one_nn11.tflite
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/mask_model_export_tflite11/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/one_nn11.tflite \
--input_shapes=1,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops

```


below failed
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/mask_model_export_tflite11/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/one_nn11_2.tflite \
--input_shapes=None,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops

```

error message
```
Failed to parse input_shape: None,320,320,3
zsh: abort
```

below successful - this produced one_nn11_2.tflite
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/mask_model_export_tflite11/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/one_nn11_2_.tflite \
--input_shapes=8,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops
```

compile success, error message, though
```
2020-09-01 22:16:24.135063: I tensorflow/lite/toco/import_tensorflow.cc:663] Converting unsupported operation: TFLite_Detection_PostProcess
2020-09-01 22:16:24.151858: F tensorflow/lite/toco/tooling_util.cc:1675] Check failed: input_array_dims[i] == input_array_proto.shape().dims(i) (1 vs. 8)
```


below successful - this produced one_nn11_3.tflite ('allow_custom_ops' discarded)
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/mask_model_export_tflite11/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/one_nn11_3_.tflite \
--input_shapes=8,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false
```

compile success, error message, though
```
2020-09-01 22:16:24.135063: I tensorflow/lite/toco/import_tensorflow.cc:663] Converting unsupported operation: TFLite_Detection_PostProcess
2020-09-01 22:16:24.151858: F tensorflow/lite/toco/tooling_util.cc:1675] Check failed: input_array_dims[i] == input_array_proto.shape().dims(i) (1 vs. 8)
```

below successful - this produced one_nn11_4.tflite ('allow_custom_ops' kept, 'change_concat_input_ranges'='true')
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/mask_model_export_tflite11/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/one_nn11_4_.tflite \
--input_shapes=8,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=true \
--allow_custom_ops
```

compile success, error message, though
```
2020-09-01 22:16:24.135063: I tensorflow/lite/toco/import_tensorflow.cc:663] Converting unsupported operation: TFLite_Detection_PostProcess
2020-09-01 22:16:24.151858: F tensorflow/lite/toco/tooling_util.cc:1675] Check failed: input_array_dims[i] == input_array_proto.shape().dims(i) (1 vs. 8)
```



> one_nn_batch_8.tflite
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/mask_model_export_quant_batch8/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/one_nn_batch_8.tflite \
--input_shapes=8,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops

```



> one_nn_det_100.tflite
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/mask_model_export_quant_det100/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/one_nn_det_100.tflite \
--input_shapes=1,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops

```

> one_nn_det_100_2.tflite
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/mask_model_export_quant_det100_2/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/one_nn_det_100_2.tflite \
--input_shapes=1,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops
```

> one_nn_det_100_3.tflite
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/mask_model_export_quant_det100_3/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/one_nn_det_100_3.tflite \
--input_shapes=1,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops
```

> CAUTION! THIS IS NOT 1NN MODEL. THIS IS THE SECOND NN OF 2NN MODEL.
> two_nn_nomask.tflite
```
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/woolee/mldl_project/github/coral_mask/1NN/nomask_model_export_quant_200903/tflite_graph.pb \
--output_file=/Users/woolee/mldl_project/github/coral_mask/1NN/quantized/two_nn_nomask.tflite \
--input_shapes=1,320,320,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops
```


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
--model ~/coral_mask/1NN/quantized/one_nn8_edgetpu.tflite \
--labels ~/coral_mask/1NN/quantized/one_nn_label.txt
```
