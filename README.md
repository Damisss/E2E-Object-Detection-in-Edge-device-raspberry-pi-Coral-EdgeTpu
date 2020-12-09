# E2E-Object-Detection-in-Edge-device-raspberry-pi-Coral-EdgeTpu
<img src="/result.gif" width="400" height="400"/>

In this tutorial atempts to depict how to implement an end to end custom object detection for edge device (Google Coral's TPU usb Accelerator). We will train a custom model using TensorFlow Object Detection API on TPU and and cloud jobs and model deployment in Coral Edge Tpu. GCP file contains the instruction for providing cloud storage bucket access to the TPU in project.

- Please follow  [Pyimagesearch's tutorial](https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator) to setup the rasberry pi for google coral's TPU usb Accelerator.

Inference:

To run real time inference: 
  - cd deployment python detect_realtime.py. 
  - Note both --input and  --confidence are optional.  If there is no input video the system will grab frame from device camera. Also confidence is set to 0.5, which can be change.

References

https://www.pyimagesearch.com/raspberry-pi-for-computer-vision
https://coral.ai/docs/
https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator
