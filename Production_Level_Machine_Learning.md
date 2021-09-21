# Production Level Machine Learning (MLOps)

Machine learning models development is only little part of the whole ML system.

![MLOps](https://www.kdnuggets.com/wp-content/uploads/Fig1-Bose-mlops-why-required-what-is.jpg)

Let's try an example from from data collecting, annotation, to model developing, training, tracking to deployment and monitoring.


1. Data preprocessing 
2. Model Training and Tracking using mlflow
3. Model Evalution and explaination
4. Model Development and Monitoring



## Resources

### Data Storage

* Data Lake
* Data Warehouse

### Orchestration

[TFX]()

[Kubeflow](https://www.kubeflow.org/)

[Metaflow](https://metaflow.org/)

### Distributed Training

* [Polyaxon](https://polyaxon.com/)

* OpenPAI

* [Horovod]()

* [Ray]()

### Training Management and Tracking

[MLflow](https://mlflow.org/)

[Mace](https://github.com/XiaoMi/mace)

[SOD](https://github.com/symisc/sod)

[MMdnn](https://github.com/Microsoft/MMdnn)


### Model Inspection

Please refer to [Responsible AI](Responsible_AI.md)

## Model Format for Deployment

* Saved Model

* protobuf

* ONNX

[tfpyth](https://github.com/BlackHC/tfpyth)


## Deployment Tools

* Cortex
* Graphpipe
* TF serving
* KF serving
* Mleap
* Clipper

### Edge Deployment

[Tensorflow Lite]()

### MCU

[Tensorflow Lite Micro]()

### Web Deployment

[Tensorflow.js]()

## Hardware Acceleration

[TensorRT]()

[GLOW](https://github.com/pytorch/glow)


## AI Platform

* Open Source
   * [Ludwig](https://github.com/ludwig-ai/ludwig):A tool box lets you train and test deep learning models without writing code. It supports distributed training with Horovod and integrates with mlflow.
   * [H2O]():
   * [ElasticSearch Machine Learning]():
   * [DataRobot]()
   * [Analytics Zoo](https://github.com/intel-analytics/analytics-zoo):Big data AI platform for scaling end-to-end AI to distributed Big Data.
   * [igel](https://github.com/nidhaloff/igel)
* Commercial
   * [Amazon SageMaker]()
   * [SalesForce Einstein Discovery]()
   * [Databricks]()
   * [Huawei ModelArts]()
   * [OpenPanel](https://github.com/onepanelio/onepanel):End-to-end computer vision platform

## Reference
* A Guide to Production Level Deep Learning [`github`](https://github.com/alirezadir/Production-Level-Deep-Learning)
* [Uber Michelangelo](https://eng.uber.com/michelangelo-machine-learning-platform/)
