# Responsible AI

Check the Interactive Streamlit [Demo]()

## Explainability

### For All Data Type (image, text, tabular data)

* Perturbation Based
   * LIME [`github`](https://github.com/marcotcr/lime)
  

* Gradient Based (for Nueral Network)

   * Saliency Map: Gradient with respect to output
      * Tutorial [`link`](https://www.kaggle.com/ernie55ernie/mnist-with-keras-visualization-and-saliency-map)
      * Keras-vis [`github`](https://raghakot.github.io/keras-vis/)

   * Integrated Gradients: Avoid saturation of activation 
      * Tutorial [`link`](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)
  
   * Grad-CAM: Gradient with respect to last Conv layer
      * Pytorch Version [`github`](https://github.com/jacobgil/pytorch-grad-cam)
   * Activation Maximization
      * Keras Tutorial [`link`](https://keras.io/examples/vision/visualizing_what_convnets_learn/)

  
* Relevance Score Based
    
   * Layerwise Relevance Propagation [`tutorial`](https://towardsdatascience.com/indepth-layer-wise-relevance-propagation-340f95deb1ea)

   * DeepLift [`github`](https://github.com/kundajelab/deeplift)

### For Tabular Data Only

#### Methods

* Feature Importance
   * Permutation Importance

* Partial Dependence Plots (PDP): When the features are corelated, it can be misleading.

* Individual Conditional Expectation (ICE)

* Accumulated Local Effects (ALE)

* Counterfact Analysis

* Error Analysis

#### Tools

* What If Tool [`github`](https://pair-code.github.io/what-if-tool/)

* InterpretML [`github`](https://github.com/interpretml/interpret)

* TCAV [`github`](https://github.com/tensorflow/tcav)

* SHAP [`github`](https://github.com/slundberg/shap)

* microsoft/responsible-ai-widgets[`github`](https://github.com/microsoft/responsible-ai-widgets/)

### For Natural Language Processing Only

* LIT [`github`](https://github.com/PAIR-code/lit)


### Research Papers

* Explaining Explanations: An Overview of Interpretability of Machine Learning [`arXiv`](https://arxiv.org/abs/1806.00069)

* TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing [`arXiv`](https://arxiv.org/abs/1807.10875)

* A Unified Approach to Interpreting Model Predictions [`arXiv`](https://arxiv.org/abs/1705.07874)


## Fairness

* Trusted-AI/AIF360 [`github`](https://github.com/Trusted-AI/AIF360)

## Privacy

## Security

## Other Resources

* Captum [`github`](https://captum.ai/)

* EthicalML/xai [`github`](https://github.com/EthicalML/xai)

* Trusted-AI/AIX360 [`github`](https://github.com/Trusted-AI/AIX360)

* [https://github.com/pbiecek/xai_resources](https://github.com/pbiecek/xai_resources)

* Evidently[`github`](https://github.com/evidentlyai/evidently)

## Reference

* https://maelfabien.github.io/machinelearning/Explorium_2/#limitations-of-linear-regression

* https://christophm.github.io/interpretable-ml-book/
