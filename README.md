# Track football payers with YOlOv8 and ByteTrack <!-- omit from toc -->


Football automated analytics is hot topics in the intersection between AI and sports. In this project, we build a tool for detection and tracking football players, referees and ball in videos. For this we use [YOLOv8](https://github.com/ultralytics/ultralytics) (the latest version of the popular and fast object detector) for detecting the players in each frame of the video, and [ByteTrack](https://github.com/ifzhang/ByteTrack) a multi object detection model released in 2022.

For the data, we use videos from the [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data) competition on Kaggle for the demo. For training YOLOv8, we sue the [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) dataset from Roboflow


## Table of content <!-- omit from toc -->
- [DEMO](#demo)
- [YOLOv8 training](#yolov8-training)
  - [YOLO8m results](#yolo8m-results)
  - [YOLO8l results](#yolo8l-results)
- [Tracking the players with ByteTrack](#tracking-the-players-with-bytetrack)
- [YOlOv8 explained](#yolov8-explained)
  - [The Backbone](#the-backbone)
  - [The Neck](#the-neck)
  - [The head](#the-head)
  - [The loss](#the-loss)
- [ByteTrack explained](#bytetrack-explained)

## DEMO

[![]()](https://youtu.be/7bFXjk8wYs0)


## YOLOv8 training

The first part of the project is to train YOLOv8 on detecting players in images: [Training notebook](train_yolov8_football_players.ipynb)

The model has trouble detecting the ball due to its small size. One of the solutions is to increase the network resolution to 1280x1280. However, it requires resources beyond my reach.

Here are the results on the validation dataset:

### YOLO8m results

| class      | Number of images | Number of instances | Precision | Recall | mAP50 | mAP50-95 |
|------------|------------------|---------------------|-----------|--------|---------|----------|
| all        | 38               | 905                 | 0.945     | 0.755  | 0.832   | 0.585    |
| ball       | 38               | 35                  | 1         | 0.206  | 0.427   | 0.164    |
| goalkeeper | 38               | 27                  | 0.888     | 0.963  | 0.972   | 0.742    |
| player     | 38               | 754                 | 0.953     | 0.964  | 0.986   | 0.796    |
| referee    | 38               | 89                  | 0.938     | 0.888  | 0.942   | 0.637    |

### YOLO8l results

| class      | Number of images | Number of instances | Precision | Recall | mAP50 | mAP50-95 |
|------------|------------------|---------------------|-----------|--------|---------|----------|
| all        | 38               | 905                 | 0.975     | 0.754  | 0.859   | 0.613    |
| ball       | 38               | 35                  | 1         | 0.215  | 0.51    | 0.206    |
| goalkeeper | 38               | 27                  | 0.961     | 0.92   | 0.981   | 0.753    |
| player     | 38               | 754                 | 0.981     | 0.958  | 0.983   | 0.814    |
| referee    | 38               | 89                  | 0.956     | 0.921  | 0.963   | 0.679    |



<!-- <div align="center">

| <img width="100%" src="resources/yolov8l-loss.png"> | 
|:--:| 
| *YOLOv8 training losses plots* |
</div> -->

## Tracking the players with ByteTrack

The second part is running yolo inference on each frame of the video and then track the detections with ByteTrack: [Tracking notebook]()


## YOlOv8 explained

YOlOv8 is a single-stage object detector, meaning one network is responsible for predicting the bounding boxes and classifying them. The YOLO series of algorithms are known for their low inference time.  
The network is built of three sections: the backbone, the neck and the head. In figure bellow, we see the full details of the network.

<div align="center">

| <img width="100%" src="https://user-images.githubusercontent.com/27466624/211974251-8de633c8-090c-47c9-ba52-4941dc9e3a48.jpg"> | 
|:--:| 
| *YOLOv8 architecture* |
| *(Source: [ open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov8))* |
</div>

### The Backbone
The backbone network extract the important features from the images at different levels. It is composed of series of ``ConvBlock`` and ``CSPLayer_2``. The CSPLayer is made of residuals blocks whose filters are concatenated to form rich features.

### The Neck
The neck is a feature pyramid network. This family of networks take as input the features of the backbone at low resolutions (the bottom-up pathway) and reconstruct them by up-scaling and applying convolution blocks between the layers. Lateral connection are added to ease the training (they function as residual connection) and overcome the lsot information due to the down-scaling and up-scaling performed.

<div align="center">

| <img width="100%" src="https://miro.medium.com/max/640/1*aMRoAN7CtD1gdzTaZIT5gA.webp"> | 
|:--:| 
| *FPN architecture* |
| *(Source: [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf))* |
</div>

### The head

The head network applies convolutions to the each output of the neck layers. Its output is prediction of the bounding box coordinates, width and height, the probability and the object class.

### The loss 
The loss function is as follows:

$$
\begin{gathered}
loss = \lambda_1 L_{box} + \lambda_2 L_{cls} + \lambda_3 L_{dfl} \\

\end{gathered}
$$

The $L_{cls}$ is a Cross Entropy loss.

The $L_{box}$ is CIoU loss, it aims to:

* Increase the overlapping area of the ground truth box and the predicted box.
* Minimize their central point distance.
* Maintain the consistency of the boxes aspect ratio.


The CIoU loss function can be defined as
$$
\mathcal{L}_{C I o U}=1-I o U+\frac{\rho^2\left(b, b^{g t}\right)}{c^2}+\alpha v .
$$
where the trade-off parameter $\alpha$ is defined as
$$
\alpha=\frac{v}{(1-I o U)+v}
$$

and $v$ measures the consistency of a aspect ratio,

$$
v=\frac{4}{\pi}\left(\arctan \frac{w^{g t}}{h^{g t}}-\arctan \frac{w}{h}\right)^2 .
$$

The $L_{dfl}$ is distributional focal loss.


## ByteTrack explained