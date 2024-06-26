# Chapter 02 - 2D Object Detection with YOLO

<img src="./docs/02-2d-object-det/2d-obj-det-yolo-cover.png" width="520" style="border:0px solid #FFFFFF; padding:1px; margin:1px">

## Overview

- This section is about **2D object detection** using **YOLO (You Only Look Once)**, which is a deep learning model that can detect and classify objects in images.
- The course uses the **Lyft 3D Object Detection Challenge** dataset, which contains camera and LIDAR data from a self-driving car.
- For the purpose of this course only Camera sets are used. 

## Requirements

- Kaggle Account
- [YOLOv3 - Ultralytics](https://github.com/ultralytics/yolov3)
- [YOLOv5 - Ultralytics (Optional)](https://github.com/ultralytics/yolov5)
- [YOLOv8 - Ultralytics (Optional)](https://github.com/ultralytics/ultralytics)

## The Road Segmentation Problem

![](https://miro.medium.com/v2/resize:fit:850/1*alePWgdMrQeS7-WBVd0MVA.jpeg)

(Src: [The AI Technology](https://medium.com/the-ai-technology/segmentation-vs-object-detection-vs-classification-things-you-need-to-know-b86b05f5167b)

- Given a 2D image's input, we are required to estimate the `location` defined by a `bounding box` and the `class` of all objects in the scene. 
- Usually, we choose classes that are relevant to our application.
- For self-driving cars, we usually are most interested in object classes that are dynamic, that is ones that move through the scene.
- These include vehicles in their subclasses, pedestrians, and cyclists.
- Drawbacks:
  - Extent of obj is not fully observed
    - occlusion: background obj covered by foreground obj
    - Truncation:Obj out of img boundaries 
  - scale
    - object size gets smaller as the object moves farther ways
  - Illumination changes:
    - too bright
    - too dark  

## Solutions/Methodologie: Deep Learning Algorithms - dynamic and fast computation, high scalability and accuracy

![](https://github.com/afondiel/Self-Driving-Cars-Specialization/raw/main/Course3-Visual-Perception-for-Self-Driving-Cars/resources/w4/img/l1-obj-det-math1.png)
(Src: [Visual Perception - UoT - Cousera](https://github.com/afondiel/Self-Driving-Cars-Specialization/raw/main/Course3-Visual-Perception-for-Self-Driving-Cars/resources/w4/img/l1-obj-det-math1.png))

Given an input image x , we want to find the `function` (algorithm) that produces an output vector that includes the coordinates of the top-left of the box, x_min and y_min, and the coordinates of the lower right corner of the box, x_max and y_max, and a class score Sclass1 to Sclassk.

- `YOLO (You Only Look Once)` is one of the ConvNet capable of producing this result.


## 2D Object Detection: Key Concepts

- The goal of **2D Object Detection** is to locate and classify objects in an image using bounding boxes, which are rectangular regions that enclose the objects.
- A **bounding box** is defined by four parameters: the center coordinates (x, y), the width (w), and the height (h) of the box.
- A **class label** is a categorical variable that indicates the type of the object, such as car, person, dog, etc.
- A **confidence score** is a numerical value that reflects how confident the model is about the detection, ranging from 0 to 1.
- A **threshold** is a value that determines whether a detection is valid or not, based on the confidence score. For example, if the threshold is 0.5, only detections with confidence scores above 0.5 are considered valid.
- A **non-maximum suppression (NMS)** is a technique that eliminates redundant and overlapping bounding boxes, keeping only the ones with the highest confidence scores.

## YOLO (You Only Look Once) in Action: How it Works?

<img src="./docs/02-2d-object-det/2d-obj-det-yolo-concept.png" width="520" style="border:0px solid #FFFFFF; padding:1px; margin:1px">

- **YOLO** is a state-of-the-art, real-time object detection algorithm that uses a single convolutional neural network (CNN) to process an image in one pass.
- YOLO v3 divides the input image into a 19x19 grid (cells) and predicts a vector for each cell that contains the probability, position, size, and class of the object in that cell.
- YOLO v3 uses **anchor boxes** to handle multiple objects in the same cell. Each anchor box has a predefined shape and predicts a specific object.
- YOLO v3 uses **Intersection over Union (IOU)** and **non-maximum suppression (NMS)** to eliminate overlapping and redundant bounding boxes and select the best ones.

![](https://storage.googleapis.com/kaggle-media/competitions/rsna/IoU.jpg)
(Src: [Lyft Dataset Kaggle](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/overview))

- YOLO applies a series of convolutional and pooling layers to the input image, followed by a fully connected layer that outputs a `tensor` of shape `(grid size x grid size x (bounding boxes x (4 + 1 + classes)))`.
- The output `tensor` can be interpreted as a set of predictions for each cell in the grid, where each prediction consists of 4 coordinates, 1 confidence score, and a vector of class probabilities.
- YOLO applies a `softmax function` to the `class probabilities` and a `sigmoid` function to the `confidence scores` and `the coordinates`.
- YOLO applies a threshold to filter out low-confidence detections and performs NMS to remove duplicate detections.

### YOLO benchmarks

- YOLO is fast and accurate, achieving state-of-the-art performance on various benchmarks and real-world applications.
- YOLO has several versions, such as YOLOv2, YOLOv3, YOLOv4, and YOLOv5, each introducing new features and improvements to enhance the speed, accuracy, and robustness of the algorithm.
- The latest version (update: 2024) of YOLO is YOLOv8 , developed by [Ultralytics](https://docs.ultralytics.com/), which supports a `full range of vision AI tasks`, such as:
  - Detection
  - Segmentation
  - Pose estimation
  - Classification ...

## Lab: Resources & Notebooks

**Resources**

- [Kaggle Competition Dataset](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles)
- [Dataset Visualization Notebook - EDA](https://www.kaggle.com/code/sakshaymahna/lyft-3d-object-detection-eda) 
- [YOLO Paper](https://arxiv.org/pdf/1506.02640.pdf)

**YOLOv3 Model Implementation**

- [YOLO3 (Detection, Training, and Evaluation) - @experiencor](https://github.com/experiencor/keras-yolo3)
- [How to Perform Object Detection With YOLOv3 in Keras - Jason Brownlee - 2019 (old but gold)](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)

**2D Object Detection Notebook Implementation**

|Notebook|Colab|Kaggle|
|--|--|--|
|[Go to notebook](./lab/notebooks/02-2d-object-det/2d-object-detection-yolov3-keras.ipynb)| [![Open notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/afondiel/Self-Driving-Cars-Perception-and-Deep-Learning-Free-Course-freeCodeCamp/blob/main/lab/notebooks/02-2d-object-det/2d-object-detection-yolov3-keras.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/thepostitguy/2d-object-detection-yolov3-keras/edit)|


**Additional resources**

- [Playlist on Fundamentals of Object Detection](https://www.youtube.com/playlist?list=PL_IHmaMAvkVxdDOBRg2CbcJBq9SY7ZUvs) 
- [Blog on YOLO](https://www.webscale.com/engineering-education/)

## Retrospective

![](https://www.kaggleusercontent.com/kf/79252954/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..yWbyL6PXinV6qLAKuatBQA.OUQXGS5x_b2mbq89ozldEcxMwWgpSbTokVCijfBDmRdSTrRjKp9JQ1FBaSLN4EoYwS3PiQlnNP3OkIueJBRtPNAQAmg1XL3ju5DRurbBPsp7mABWx1zQuyIRCtrswS7Xx4sdhW5QoVCsy_FCnR-NItZ5lJJ0JhFhLTNrAKsMwdxbGZBSVBPuCMDGbAN-v4xD2At8ZvUylnChUpXftTLOeFyMSMbjAy18tDP-liRKX_LIz90YWgcrU3NSynhdanuVLr8BiKFsIY4er8fCCYUn_Uy2vmgDRtGMf1fsQiJPn7cjcE-0938ur9AA5btHqaDjEZVKx7o3VyZyQqclHkezeZvILQgemW15nvpR9YgRKgJaSMdIRQNSNRyB3ftTlOi-_UO7aEbl_Zv4FBES_rsJ82y7-_5X-SLWwbsrMQX1GstUHM0fXY86r2zgyBrT0bSP_YiROQhAApfoGEWC_zJnUIBOifGiU4RlgepwpuSpc9gaqbr2iOTbsgnZJvl7Qoz3rQ9fRXu7l5ZHJWQBhy8MoLkS1ASziLk20FjdGqnly09Hz5IE02SZfBjGfynkGRBVjgn5idAvnxbTlq8ainrk9TsK-Yl7gEyxQwoWk83MD4G291E7pAa_QVdSY6jQuRConTxomWS9Dd-UrjErRkVi00VMgfV1rKcgDceXAfDc3EJwtIV2ZOyo_eOS5-W_2EOs.QoTSvC3pepXFA6zXVb1lcQ/__results___files/__results___16_5.png)


Alright, so these were the results that we got on running the Yolo v3 algorithm on the left and we already have very good algorithms like YOLO that do our tasks.

However, there is some research that is still required on **[3D Object Detection](https://www.quora.com/How-different-is-3D-object-detection-from-2D-object-detection-for-a-researcher)**, which was also the motivation behind why Lyft conducted this competition as  well.

## Contributing

Taks:
- Test the same scenario w/ YOLOv5 available on Kaggle and compare the against the YOLOv3
- Test with [YOLOv8](https://www.kaggle.com/models/keras/yolov8) (last till now: March 24) and compare w/ YOLOv5, and YOLOv5  


## References

- [Kaggle Competition Dataset](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles)
- [Visualization Notebook - EDA](https://www.kaggle.com/code/sakshaymahna/lyft-3d-object-detection-eda) 
- [YOLO Notebook Implementation](https://www.kaggle.com/code/sakshaymahna/yolov3-keras-2d-object-detection) 
- [Playlist on Fundamentals of Object Detection](https://www.youtube.com/playlist?list=PL_IHmaMAvkVxdDOBRg2CbcJBq9SY7ZUvs) 
- [Blog on YOLO](https://www.webscale.com/engineering-education/)
- [YOLO Paper](https://arxiv.org/pdf/1506.02640.pdf)

Additional resources:
- [Course-3 - W4 - MODULE 4: 2D Object Detection, Course3 Visual Perception for Self-Driving Cars - U of T - Cousera](https://github.com/afondiel/Self-Driving-Cars-Specialization/blob/main/Course3-Visual-Perception-for-Self-Driving-Cars/course3-w4-notes.md)
- [Object Detection using Deep Learning - OpenCV Team, COCO Dataset and MobileNet - Google](https://github.com/afondiel/Computer-Vision-OpenCV-Python-Free-Course-Udemy/tree/main/14_Object_Detection_using_Deep_Learning)
- (1) [YOLO Object Detection Explained: A Beginner's Guide | DataCamp.](https://www.datacamp.com/blog/yolo-object-detection-explained) 
- (2) [YOLO Algorithm for Object Detection Explained [+Examples] - Medium.](https://www.v7labs.com/blog/yolo-object-detection) 
- (3) [Home - Ultralytics YOLOv8 Docs.](https://docs.ultralytics.com/) 
  
