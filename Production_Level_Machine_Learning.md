# Production Level Machine Learning (MLOps)

Machine learning models development is only little part of the whole ML system.

![MLOps](https://www.kdnuggets.com/wp-content/uploads/Fig1-Bose-mlops-why-required-what-is.jpg)

Let's follow the steps beflow to build a machine learning service on GCP.

1. Data Acquisition and ETL Process (Fetch data from internet )

Kafka, Spark

2. Data Validation/Data Annotation and Store.

Let's use labeling tools 

3. Data preprocessing/model training and tracking

Here the most convinient way is Jupyter Notebook

We can use MLflow to evaluate our model.

4. Model Evalution and explaination

Here we can use Manifold, SHAP for explaniation

5. Model Development

Let's convert our model to the format we want for serving

6. Model performance monitoring

What can be monitored from our model?

Let's use Elasticsearch APM to monitor our model




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
   * [Google AI Platform]()
   * [SalesForce Einstein Discovery]()
   * [Databricks]()
   * [Huawei ModelArts]()
   * [OpenPanel](https://github.com/onepanelio/onepanel):End-to-end computer vision platform

## Reference
* A Guide to Production Level Deep Learning [`github`](https://github.com/alirezadir/Production-Level-Deep-Learning)
* [Meet Michelangelo: Uber’s Machine Learning Platform](https://eng.uber.com/michelangelo-machine-learning-platform/)
