# Production Level Machine Learning (MLOps)

Machine learning models development is only little part of the whole ML system.

![MLOps](https://www.kdnuggets.com/wp-content/uploads/Fig1-Bose-mlops-why-required-what-is.jpg)


1. Data Acquisition or Collecting:
   - ELK
   - Kafka

2. Data Store:
   - File Storage:
     - Amazon S3
     - GCS
   - Data Lake:
     - HDFS
     - Elastic Search
   - Data Warehouse:
     - SnowFlake
     - Google BigQuery
   - Database
     - SQL   
     - noSQL

3. ETL Process:
   - Spark
   - ETL Orchestration:
     - Airflow
     
4. Data Annotation/Labeling:
   - Label Studio      

5. Data Validation:   
   - Tensorflow Data Validation

6. Data Transform:
   - Tensorflow Transform

7. Data Preparation/Version Control:
   - Split
   - DVC

8. Model Training/Tracking/Evaluation:
   - Tracking
     - Mlflow 
   - Inspection
     - Shap
     - Tensorflow Model Analysis

9. Model Deployment: 
   - Seldon
   - KFServing

10. Model Monitoring:
   * Data Quality
     * Data Drift
     * Outlier detection
   * Metrics 
   * Concept Drift
   * Concerted adversaries
   * Operational related index
   - Elasticsearch APM
   - Prometheous

X. Pipeline Orchestration:
   Steps above can be built by pipeline orchestration tool
   - Kubeflow
   - Airflow
  



## Model IP Protection

How to Prove Your Model Belongs to You: A Blind-Watermark based Framework to Protect Intellectual Property of DNN [`arXiv`](https://arxiv.org/abs/1903.01743)


##  Adversarial Attack

[Breaking neural networks with adversarial attacks](https://towardsdatascience.com/breaking-neural-networks-with-adversarial-attacks-f4290a9a45aa)



## Resources

### Data Storage

* Data Lake 
  * Amazon S3
  * Google Cloud Storage (GCS)
  * Hadoop Distributed File System (HDFS)
  * MongoDB
* Data Warehouse
  * BigQuery 
  * Snowflake

### Toolkit

TFX [`link`](https://www.tensorflow.org/tfx)

### Orchestration

Kubeflow [`link`](https://www.kubeflow.org/)

Metaflow [`link`](https://metaflow.org/)

Airflow [`link`](https://airflow.apache.org/)

### Distributed Training

* Polyaxon [`link`](https://polyaxon.com/)

* OpenPAI [`link`](https://openpai.readthedocs.io/en/latest/#:~:text=OpenPAI%20is%20an%20open%2Dsource,User%20Manual%20and%20Admin%20Manual.)

* Horovod [`github`](https://github.com/horovod/horovod)

* Ray [`link`](https://www.ray.io/)

### Training Management and Tracking

MLflow [`link`](https://mlflow.org/)

Mace [`github`](https://github.com/XiaoMi/mace)

SOD [`github`](https://github.com/symisc/sod)

MMdnn [`github`](https://github.com/Microsoft/MMdnn)


### Model Inspection

Please refer to [Responsible AI](Responsible_AI.md)

## Model Format for Deployment

* Tensorflow [`link`](https://www.tensorflow.org/guide/keras/save_and_serialize)
  * TensorFlow SavedModel format (or in the older Keras H5 format)
  * Saving the architecture / configuration only
  * Saving the weights values only

* Pytorch
  * Torchscript 
  * ONNX

tfpyth [`github`](https://github.com/BlackHC/tfpyth)


## Deployment Tools

* Cortex
* Graphpipe
* TF serving
* KF serving
* Mleap
* Clipper
* Torchserve

### Edge Deployment

Tensorflow Lite [`link`](https://www.tensorflow.org/lite)

### MCU

Tensorflow Lite Micro [`link`]()

### Web Deployment

Tensorflow.js [`link`]()

## Hardware Acceleration

TensorRT [`link`]()

GLOW [`github`](https://github.com/pytorch/glow)


## AI Platform

* Open Source
   * Ludwig [`github`](https://github.com/ludwig-ai/ludwig):A tool box lets you train and test deep learning models without writing code. It supports distributed training with Horovod and integrates with mlflow.
   * H2O  [`github`](https://github.com/h2oai/h2o-3):
   * ElasticSearch Machine Learning [`link`]():
   * DataRobot [`link`]()
   * Analytics Zoo [`github`](https://github.com/intel-analytics/analytics-zoo):Big data AI platform for scaling end-to-end AI to distributed Big Data.
   * igel [`github`](https://github.com/nidhaloff/igel)
* Commercial
   * [Amazon SageMaker]()
   * [Google AI Platform]()
   * [SalesForce Einstein Discovery]()
   * [Databricks]()
   * [Huawei ModelArts]()
   * OpenPanel [`github`](https://github.com/onepanelio/onepanel):End-to-end computer vision platform

## Reference
* A Guide to Production Level Deep Learning [`github`](https://github.com/alirezadir/Production-Level-Deep-Learning)
* [Meet Michelangelo: Uber’s Machine Learning Platform](https://eng.uber.com/michelangelo-machine-learning-platform/)
