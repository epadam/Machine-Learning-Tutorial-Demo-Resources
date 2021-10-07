# Production Level Machine Learning (MLOps)

Machine learning models development is only little part of the whole ML system.

![MLOps](https://www.kdnuggets.com/wp-content/uploads/Fig1-Bose-mlops-why-required-what-is.jpg)

Let's follow the steps beflow to build a machine learning service on GCP.

1. Data Acquisition and ETL Process (Fetch data from internet )

  - Kafka, Spark

2. Data Validation/Data Annotation and Store.



3. Data preprocessing/


4. model training and tracking
   * Early Stopping
   * checked point
  
We can fetch the data in bigQuery and do some preprocessing.

Let's orchastrate a pipeline for machine learning pipeline

Here the most convenient way is Jupyter Notebook, you can also use H2O

You can also use autoML to automatically build the model, but it would take some time.

We can use MLflow or Tensorboard to evaluate different model or hyperparameters.

5. Model Inspection and Explanation

Here we can use Manifold, SHAP for explanation to avoid bias and misbehavior

It would be nice if there is an interactive report that you can show the EDA and explanbility

6. Model Deployment
 

Let's convert our model to the format we want for serving
(Depending on the scenario, you might need to give explanation for every single inference like medical usage or)

You may need some optimization to reduce the cost of inference, 


7. Model Performance Monitoring

What can be monitored from our model? 

   * Data Quality
     * Data Drift
     * Outlier detection
   * Metrics 
   * Concept Drift
   * Concerted adversaries
   
* Operational related index

Let's use Elasticsearch APM to monitor our model


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

### Orchestration

[TFX](https://www.tensorflow.org/tfx)

[Kubeflow](https://www.kubeflow.org/)

[Metaflow](https://metaflow.org/)

### Distributed Training

* [Polyaxon](https://polyaxon.com/)

* [OpenPAI](https://openpai.readthedocs.io/en/latest/#:~:text=OpenPAI%20is%20an%20open%2Dsource,User%20Manual%20and%20Admin%20Manual.)

* [Horovod](https://github.com/horovod/horovod)

* [Ray](https://www.ray.io/)

### Training Management and Tracking

[MLflow](https://mlflow.org/)

[Mace](https://github.com/XiaoMi/mace)

[SOD](https://github.com/symisc/sod)

[MMdnn](https://github.com/Microsoft/MMdnn)


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

[tfpyth](https://github.com/BlackHC/tfpyth)


## Deployment Tools

* Cortex
* Graphpipe
* TF serving
* KF serving
* Mleap
* Clipper
* Torchserve

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
   * [Google AI Platform]()
   * [SalesForce Einstein Discovery]()
   * [Databricks]()
   * [Huawei ModelArts]()
   * [OpenPanel](https://github.com/onepanelio/onepanel):End-to-end computer vision platform

## Reference
* A Guide to Production Level Deep Learning [`github`](https://github.com/alirezadir/Production-Level-Deep-Learning)
* [Meet Michelangelo: Uber’s Machine Learning Platform](https://eng.uber.com/michelangelo-machine-learning-platform/)
