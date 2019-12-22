# Machine Learning All in One

For people who don't have time but want to have a quick overview of the development in machine learning 

# Theory and Basic Models

Regression

Decision Tree

Random Forest

Bayesian

Enssemble Algorithms

Support Vector Machine

### Deep Learning

CNN

Feature Pyramid Networks

Graph Convolutional Networks

Fully Convolutional Networks

[Capsule Neural Network](https://github.com/Sarasra/models/tree/master/research/capsules)

Autoencoder

Variational Autoencoder

Generative Adversarial Network (GAN)

VAE-GAN

[Graph Networks](https://arxiv.org/abs/1806.01261)

RNN

LSTM

Temporal Convolutional Network

Attention

Transformer

# ML Training

## Basic Methods

### Loss Function

### Optimization

### Regularization

Batch Normalization

### Tools

[gradient-checkpointing](https://github.com/cybertronai/gradient-checkpointing)

## Supervised Learning

## Semi-Supervised Learning (mix labeled and unlabeled data)

## Self-Supervised Learning (Unsupervised Learning)

Self-supervised learning means training the models without labeled data. AutoEncoder, GAN are examples in CV, while pre-trained model like Bert is example in NLP.

[UGATIT](https://github.com/taki0112/UGATIT?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)


## Transfer Learning

Transfer learning dominates the machine learning today. CV mostly benefit from supervised learning and transfer the features learned to similar tasks. NLP on the other hands benefit more from self-supervised learning. These pre-trained models show incredible performances in many tasks, such as Bert, GPT-2.

## Federated Learning

Federated learning can train the model without upload the data to a single machine and keeps the data at local machine.

# Data pre-processing

## Data Labeling

[Snorkel](https://www.snorkel.org/)

[Snorkel DryBell](https://arxiv.org/abs/1812.00417)

Papers and Blogs:

[Annotating Object Instances with a Polygon-RNN](http://www.cs.toronto.edu/polyrnn/poly_cvpr17/)

[Fluid Annotation: An Exploratory Machine Learning–Powered Interface for Faster Image Annotation](https://ai.googleblog.com/2018/10/fluid-annotation-exploratory-machine.html)

[Demo](https://fluidann.appspot.com/)

[Semi-Automatic Labeling for Deep Learning in Robotics](https://arxiv.org/abs/1908.01862)

## Data Augmentation

[Augmentor](https://github.com/mdbloice/Augmentor?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)

[albumentations](https://github.com/albumentations-team/albumentations)

[DALI](https://github.com/NVIDIA/DALI)

## Data Privacy

[Differential Privacy](https://github.com/google/differential-privacy)

[Tensorflow Differential Privacy](https://medium.com/tensorflow/introducing-tensorflow-privacy-learning-with-differential-privacy-for-training-data-b143c5e801b6)

## Tools

Spark

Neo4j

# Explainable ML

[InterpretML](https://github.com/interpretml/interpret)

[Google TCAV](https://github.com/tensorflow/tcav)

Papers:

[Explaining Explanations: An Overview of Interpretability of Machine Learning](https://arxiv.org/abs/1806.00069)

[TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing](https://arxiv.org/abs/1807.10875)

## Visualization

[What If Tool](https://pair-code.github.io/what-if-tool/)

[LIME](https://github.com/marcotcr/lime)

Image Generator

Image heat map:

[Saliency maps](https://www.kaggle.com/ernie55ernie/mnist-with-keras-visualization-and-saliency-map)

Kernel Map Visualization

Intermiediate Layer Output Visualization 

[Embedding Projector](https://towardsdatascience.com/visualizing-bias-in-data-using-embedding-projector-649bc65e7487)

[TensorWatch](https://github.com/microsoft/tensorwatch)

# Image and Video Processing

## Image/Video Analysis

[DeepVideoAnalytics](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics)

## 3D Rendering

[TensorFlow Graphics](https://github.com/tensorflow/graphics)

## Image/Video Generation

[deepfakes_faceswap](https://github.com/deepfakes/faceswap)

[style2paints](https://github.com/lllyasviel/style2paints)

[CycleGAN](https://junyanz.github.io/CycleGAN/)

[pix2pix](https://phillipi.github.io/pix2pix/)

[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

[Super Resolution](https://towardsdatascience.com/deep-learning-based-super-resolution-without-using-a-gan-11c9bb5b6cd5)

[Super Resolution2](https://github.com/open-mmlab/mmsr)

[Deep-Exemplar-based-Colorization](https://github.com/msracver/Deep-Exemplar-based-Colorization)

[Deep image prior](https://dmitryulyanov.github.io/deep_image_prior)

[video-object-removal](https://github.com/zllrunning/video-object-removal)

Papers:

[Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1807.10875)

## Image/Video Compression

[generative-compression](https://github.com/Justin-Tan/generative-compression)

Papers:

# Computer Vision

## Image Classification

VGG

ResNet

DenseNet

Morphnet

MobileNet

EfficientNet

Papers:

[Active Generative Adversarial Network for Image Classification](https://arxiv.org/abs/1906.07133)


## Object Detection

SSD

Faster R-CNN

[YOLO3](https://pjreddie.com/darknet/yolo/)

[DIoU-darknet](https://github.com/Zzh-tju/DIoU-darknet)

[Google object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)

[SNIPER](https://github.com/mahyarnajibi/SNIPER)

[NCRF](https://github.com/baidu-research/NCRF)

## Semantic Image Segmentation

[DeepLab](http://liangchiehchen.com/projects/DeepLab.html)

[DeepLab V3](https://github.com/tensorflow/models/tree/master/research/deeplab)

[body-pix](https://github.com/tensorflow/tfjs-models/tree/master/body-pix)

[robosat](https://github.com/mapbox/robosat)

## Instance Segmentation

[MMDetection](https://github.com/open-mmlab/mmdetection)

[Mask R-CNN](https://github.com/matterport/Mask_RCNN)

[ShapeMask](https://cloud.google.com/blog/products/ai-machine-learning/high-performance-large-scale-instance-segmentation-with-cloud-tpus) 

[yolact](https://github.com/dbolya/yolact)

## Image and Video Depth Estimation

[Densepose](https://github.com/facebookresearch/DensePose)

Papers:

[Deep Convolutional Neural Fields for Depth Estimation from a Single Image](https://arxiv.org/abs/1411.6387)

[Moving Camera, Moving People: A Deep Learning Approach to Depth Prediction](https://ai.googleblog.com/2019/05/moving-camera-moving-people-deep.html)

## Pose Detection

[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## All in One

[Detectron2](https://github.com/facebookresearch/detectron2)

Papers:

[Panoptic Segmentation](https://arxiv.org/abs/1801.00868)

## Object tracking

## Action Recognition


## Facial Recognition

[OpenFace](https://cmusatyalab.github.io/openface/)

[Face Recognition](https://github.com/ageitgey/face_recognition)

## SLAM/3D Reconstruction

Papers:

https://arxiv.org/abs/1712.07122

Resources:

https://zhuanlan.zhihu.com/p/74085115

https://zhuanlan.zhihu.com/p/64720052

https://niessnerlab.org/projects/avetisyan2019scan2cad.html

https://zhuanlan.zhihu.com/p/60954106

https://medium.com/vitalify-asia/create-3d-model-from-a-single-2d-image-in-pytorch-917aca00bb07


# Natural Language Processing

### Pre-trained Language Models

Bert

UniLM

XLNet

GPT-2

T5

### Framework and Tools

[Bert as Service](https://github.com/hanxiao/bert-as-service)

[Huggingface](https://github.com/huggingface)

[spaCy](https://spacy.io)

[nlp-architect](https://github.com/NervanaSystems/nlp-architect)

### Applications

Part-of-speech Tagging, Dependecy Parsing, Named Entity Recognition

Text Classification, Text Generation

Question Answering

Machine Translation

Automatic Summarization

Textual Entailment

Dialog System:

[DeepPavlov](https://github.com/deepmipt/DeepPavlov)

[ParlAI](https://github.com/facebookresearch/ParlAI)

## Knowledge Graph

Papers:

[Knowledge Graphs in Natural Language Processing @ACL 2019](https://medium.com/@mgalkin/knowledge-graphs-in-natural-language-processing-acl-2019-7a14eb20fce8)

## Speaker Diarization

## Speech Recognition (ASR)

[Alibaba-MIT-Speech](https://github.com/alibaba/Alibaba-MIT-Speech)

[DeepSpeech](https://github.com/mozilla/DeepSpeech)

[Wav2letter++](https://github.com/facebookresearch/wav2letter)

[Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)

On-device wake word detection:

[porcupine](https://github.com/Picovoice/porcupine)

## Speech Synthesis (TTS)

[WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) 

[FastSpeech](https://arxiv.org/abs/1905.09263)

## Voice Conversion

[deep-voice-conversion](https://github.com/andabi/deep-voice-conversion)

# Integration of CV and NLP

## Image and Video Content Understanding, Reasoning and Analysis

Recognize Text in image or Video

[text-to-video-generation](https://antonia.space/text-to-video-generation)

Papers and Blogs:

[Pythia v0.1: the Winning Entry to the VQA Challenge 2018](https://arxiv.org/abs/1807.09956)

[Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods](https://arxiv.org/abs/1907.09358)

[Hello, World: Building an AI that understands the world through video](https://medium.com/twentybn/watch-and-learn-building-an-ai-that-understands-the-world-through-video-9e2796400176)

# Training Tracking and Deployment

Tensorflow in production (TF Estimator, TF Serving)

Mobile device with tensorflow (tensorflow Lite, tensorflow.js)

[Polyaxon](https://polyaxon.com/)

[MLflow](https://mlflow.org/)

[Mace](https://github.com/XiaoMi/mace)

[SOD](https://github.com/symisc/sod)

Blogs:

[A scalable Keras + deep learning REST API](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/)

# Tools and Frameworks

[Tensorflow Models](https://github.com/tensorflow/models)

[TF-Slim](https://github.com/google-research/tf-slim)

[MMdnn](https://github.com/Microsoft/MMdnn)

[tfpyth](https://github.com/BlackHC/tfpyth)

# Hardware Acceleration

[GLOW](https://github.com/pytorch/glow)

# Other Applications of Machine Learning

[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

Emotion Recognition/Generation, Sentiment Analysis

Art and Music Generation:

[LibROSA](https://librosa.github.io/librosa/)

[Magenta](https://magenta.tensorflow.org/)

Fraud Detection

Maleware Detection

Fake Documents Detection

[Defect Detection](https://devblogs.nvidia.com/automatic-defect-inspection-using-the-nvidia-end-to-end-deep-learning-platform/)

Cybersecurity

[Auto Code Generator](https://github.com/tonybeltramelli/pix2code)

Code IntelliSense (code-completion)

[python_autocomplete](https://github.com/vpj/python_autocomplete)

[Finding and fixing software bugs](https://engineering.fb.com/developer-tools/finding-and-fixing-software-bugs-automatically-with-sapfix-and-sapienz/)

[Deep universal probabilistic programming](https://github.com/pyro-ppl/pyro)

Construction

[Virtual Assistant](https://github.com/DragonComputer/Dragonfire)

[An implementation of a deep learning recommendation model (DLRM)](https://github.com/facebookresearch/dlrm?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)

## Medical:

[NiftyNet](https://github.com/NifTK/NiftyNet)

[Med3D: Transfer Learning for 3D Medical Image Analysis](https://github.com/Tencent/MedicalNet)

[deepvariant](https://github.com/google/deepvariant)

# Model Protect

Papers:

[How to Prove Your Model Belongs to You: A Blind-Watermark based Framework to Protect Intellectual Property of DNN](https://arxiv.org/abs/1903.01743)

# Network Compression

[Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

# Neural Architecture Search (NAS)

[AutoKeras](https://autokeras.com/)

Papers:

[Auto DeepLab](https://arxiv.org/abs/1901.02985)

[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)

# Anomaly Detection

# ML Attack

[Breaking neural networks with adversarial attacks](https://towardsdatascience.com/breaking-neural-networks-with-adversarial-attacks-f4290a9a45aa)


# Reinforcement Learning

Traffic Control

## Imitation Learning (supervised learning)

Papers:

[Global overview of Imitation Learning](https://arxiv.org/abs/1801.06503)

# Robotics

Papers:

[End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702)

[Synthetically Trained Neural Networks for Learning Human-Readable Plans from Real-World Demonstrations](https://arxiv.org/abs/1805.07054)

# Self Driving Car

Blogs:

[End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

[Pixel-Perfect Perception: How AI Helps Autonomous Vehicles See Outside the Box](https://blogs.nvidia.com/blog/2019/10/23/drive-labs-panoptic-segmentation/)

[How to build a self-driving car in one month](https://getpocket.com/redirect?url=https%3A%2F%2Fmedium.com%2F%40maxdeutsch%2Fhow-to-build-a-self-driving-car-in-one-month-d52df48f5b07)

## Platform

Carla

[AirSim](https://github.com/Microsoft/AirSim)

Apollo

# Academic Focus

## Life Long Learning

## Few Shot Learning / One Shot Learning / Zero Shot Learning

### Metric-Based

Siamese Network

### Meta Learning (Learning to learn)

### Model-Based

### Optimization-Based

# Tips

https://buzzorange.com/techorange/2019/03/14/4-tips-of-shell/

# Reference

https://kknews.cc/code/34aja5o.html

# [Learning Resources](learning_resources.md)

[TensorFlow Hub](https://github.com/tensorflow/hub)

[tensor2tensor](https://github.com/tensorflow/tensor2tensor?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)

# Dataset Resources
