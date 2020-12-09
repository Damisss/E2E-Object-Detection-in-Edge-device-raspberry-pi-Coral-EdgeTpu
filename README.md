# E2E-Object-Detection-in-Edge-device-raspberry-pi-Coral-EdgeTpu
<img src="/result.gif" width="400" height="400"/>

This tutorial attempts to depict the implementation of  an end to end custom object detection for edge device (Google Coral's TPU usb Accelerator). We will perform quantize aware training using TensorFlow Object Detection API. Remark the colab notebook shows the training workflow. However,  training were runing on Cloud TPUs (on AI Platform Jobs). GCP file provides the instruction for allowing TPU in project to access cloud storage bucket.

- Please follow  [Pyimagesearch's tutorial](https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator) to setup the rasberry pi for google coral's TPU usb Accelerator.

Inference:

To run real time inference: 
  - cd to deployment folder:
    - python detect_realtime.py. 
  - Note both --input and  --confidence are optional.  If there is no input video the system will grab frame from device camera. Also confidence is set to 0.5, which can be changed.

References

- https://www.pyimagesearch.com/raspberry-pi-for-computer-vision/
- https://coral.ai/docs/
- https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator/
