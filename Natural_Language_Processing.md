# Dive into Natural Language Processing

You can play with a [Streamlit demo]() to see what can NLP do. You can find the code [here]().

And check all the [notebooks]() in this guide.

If you are interested in all tasks of NLP, [PapersWithCode](https://paperswithcode.com/area/natural-language-processing) is definitely the good place.

This is a guide that helps you understand the NLP world, from basic knowledge to building a real application.
If you are already a python programmer, I suggest you also read through the guide from [AllenNLP](https://guide.allennlp.org/) and [spaCy](https://course.spacy.io/en). Then check [spaCy 101](https://spacy.io/usage/spacy-101) to see what the a NLP library offers to build a NLP application. Most libraries offer from text preprocessing, cleansing to model training.

Enjoy!!
<ul>
    <li><a href="#language-models">Language Models</a></li>
    <li><a href="#nlp-tasks">NLP Tasks</a></li>
    <li><a href="#Resouces-for-NLP">More Resources for NLP</a></li>
    <li><a href="#Simple-project">How to develop a production level NLP service</a></li>
    <li><a href="#Advanced-research-topics-in-NLP">Advanced Research Topics in NLP</a></li>
   
</ul>



# Language Models

Language models take the text as input and generate a prediction, a word, a sentence or even an article. 

- [Tokenization](#Tokenization)
- [Statistical Language Models](#Statistical-Language-Models)
  - [Bag of Words](#bag-of-words)
  - [N-grams](#N-grams)
  - [TF-IDF](#TF-IDF)
  - [Glove](#glove)
  - [FastText](#FastText)
- [Neural Language Models](#Neural-Language-Models)
  - [Word2Vec](#Word2Vec) 
  - [RNN/GRU/LSTM](#RNN/GRU/LSTM)
  - [ELMo](#Elmo)
  - [Transformer](#Transformer)
    - [Attention](#Attention)
    - [Encoder](#Encoder)
    - [Decoder](#Decoder)


## Tokenization

Vocab:

* BPE
* WordPiece

SentencePiece [`github`](https://github.com/google/sentencepiece)



## Statistical Language Models

### Bag of Words
* Calculate how many times a word is showing in a documents.


### N-grams
* N-token sequence of words

### TF-IDF 
* Term Frequency x Inverse Document Frequency 

### Glove

* Create word vectors that capture meaning in vector space. It takes advantage of global count statistics instead of only local information.


### FastText

* N-gram + Softmax

## Neural Language Models

### Word2Vec

* CBOW
* Skip-Gram

<img src="https://miro.medium.com/max/700/1*cuOmGT7NevP9oJFJfVpRKA.png" alt="Exploiting Similarities among Languages for Machine Translation" width="600"/>


### RNN/GRU/LSTM

* RNN 
<img src="http://blog.peddy.ai/assets/2019-05-26-Recurrent-Neural-Networks/rnn_rnn_unrolled.png" alt="Architeccture of RNN" width="500"/>

<img src="https://pica.zhimg.com/80/v2-b45f69904d546edde41d9539e4c5548c_720w.jpg?source=1940ef5c" alt="Detail Structure of RNN" width="500"/>

As you can see, if input is a sentence, each word is turned into a vector and fed into a neural netork.

Check the RNN source cod [here]().

Calculate number of parameters of RNN:

* GRU

<img src="https://www.researchgate.net/publication/331848495/figure/fig3/AS:738004381466626@1552965364778/Gated-Recurrent-Unit-GRU.ppm" alt="Architeccture of GRU" width="400"/>


* LSTM

<img src="https://i.stack.imgur.com/RHNrZ.jpg" alt="Architeccture of LSTM" width="400"/>


The downside of RNN/GRN/LSTM is the first input element can't see the information after it. Also, they can't be computed in parallel, so it is replaced by transformer in many NLP tasks.

### ELMo

It uses bidirectional LSTM in the seq2seq model.

### Transformer 

Transformer is proposed in 2017 by Google in the paper "Attention is all you need". 

Below is the architecture of the transformer:

<img src="https://www.researchgate.net/publication/342045332/figure/fig2/AS:900500283215874@1591707406300/Transformer-Model-Architecture-Transformer-Architecture-26-is-parallelized-for-seq2seq.png" alt="Transformer Model Architecture" width="600"/>

* code of transformer:[`notebook`]()

* How to train a transformer?

* Improved transformer models
   * [Transformer-XL]():
   * [UniLM]():
   * [Bart]():
   * [Megatron-LM]() 8.3B parameters 
   * [T5]():
   * [DeBERTa](https://github.com/microsoft/DeBERTa): 1.5B parameters
   * [Meena](): Open domain chatbot by Google based on Transformer.
   * [Blender](): Open domain chatbot by Facebook based on Transformer.
   * [BlenderBot2](): 2nd generation chatbot by Facebook based on Transformer. It can search information on the internet and reply up to date information.

Transformer architecture has been used for many domains such as biology, computer vision and business data analysis.


### Encoder

Transformer is a seq2seq model which can do NLU and NLG tasks. Only Encoder itself can mostly do NLU jobs.

The most famous encoder is Bert proposed by Google. You can check the original paper here. Although it is not the first one introducing pre-train and finetune paradime

<img src="https://miro.medium.com/max/1200/1*p4LFBwyHtCw_Qq9paDampA.png" alt="Bert Pretraining and Fine-Tuning" width="600"/>

* Bert from scratch! [notebook]()

* How is Bert pre-trained? MLM and Next Sentence Prediction [notebook](). 

* Improved version of Bert:
   * [XLNet]():
   * [ELECTRA]():
   * [MT-DNN](): Multi-tasking model
   * [RoBERTa]():
   * [BigBird]():
   * [ERNIE]():

* How to make Bert smaller? [`notebook`]()

   * Quantization:
      * [TinyBert]():
      * [Q-Bert]():

   * Knowledge Distillation:
      * [mobileBert]():
      * [Albert]():
      * [DistillBert]():


* Pretrained Bert models with domain specific text, which can perform better :
   * [BioBert]():
   * [SciBert]():
   * [ClinicalBERT]():
]


### Decoder

* [GPT/GPT-2/GPT-3]()
* [Turing-LG](): 17B

Decoder itself is better at generation tasks. The most famous one is GPT-2. [notebook](GPT-2)

* Control text generation:
   * Beam Search
   * random

* Bias Control of NLG model:
   * [Towards Controllable Biases in Language Generation](https://arxiv.org/abs/2005.00268)


More Resources:

* [Chinese GPT-2](https://github.com/imcaspar/gpt2-ml) 


# NLP Tasks

- [Sequence Tagging](#Sequence-Tagging)
   - [Part-of-speech Tagging](#Part-of-speech-Tagging)
   - [Named Entity Recognition](#Named-Entity-Recognition)
   - [Dependency Parsing](#Dependency-Parsing)
   - [Relation Extraction](#Relation-Extraction)
   - [Grammer correction](#Grammer-correction)
- [Sentence/Paragraph/Document Level Tasks](#Sentence/Paragrah/Document-Level-Tasks)
  - [Classification Tasks](#Classification-tasks)
    - [Sentiment Analysis](#Sentiment-Analysis)
    - [Ducument Classification](#Ducument-Classification)
    - [Textual Entailment/Natural Language Inference](#Textual-Entailment/Natural-Language-Inference)
    - [Sentence Segmentation](#Sentence-Segmentation)
    - [Paraphrase indentification](#Paraphrase-indentification)
    - [Reading Comprehension](#reading-comprehension)
    - [Extractive Summarization](#Extractive-Summarization)
    - [Text Matching/Semantic Similarity](#Text-Matching/Semantic-Similarity)
      - [Information Retrieval](#information-retrieval)
      - [Question Anwsering](#question-answering)
   - [Natural Language Generation](#natural-language-generation)
     - [Machine Translation](#Machine-Translation)
     - [Abstractive Summarization](#Abstractive-Summarization)
     - [Paraphrase Generation](#Paraphrase-Generation)
     - [Code-Generation](#Code-Generation)

 

The goal of all NLP tasks is to understand the text(each word, relation between words, sentiment or the topic) and generate the text we want. The tasks below try to understand the text from word level, sentence level and document level. 

## Sequence Tagging

### Part-of-speech Tagging 

POS can be used for grammer checking. There are statitical and deep learning solutions

Let's use this [Penn Treebank Dataset](https://deepai.org/dataset/penn-treebank)

* [Spacy and_RNN for ](notebooks/nlp/part-of-speech-tagging/Spacy_POS_RNN.ipynb)


### Grammer correction

* [LSTM for Cola](notebooks/nlp/grammer-classification/Cola_LSTM.ipynb)


### Named Entity Recognition 

NER is very useful for chatbot or building or querying knowledge graph.

* [RNN for_NER](notebooks/nlp/name-entity-recognition/NER_RNN.ipynb)

* [Bert for_NER_kaggle](notebooks/nlp/name-entity-recognition/ner-bert-kaggle.ipynb)

More dataset please check below:

* [https://github.com/juand-r/entity-recognition-datasets](https://github.com/juand-r/entity-recognition-datasets)

### Dependency Parsing

* [RNN for ](notebooks/nlp/DP_RNN.ipynb)


### Relation Extraction

* [RNN for ](notebooks/nlp/RE_RNN.ipynb)



## Sentence/Paragraph/Document Level Tasks

## Classification tasks

### Sentiment Analysis

Sentiment analysis is especially useful in customer services, understanding the feedback of the product or services.
There are acutially two types, one is sentiment ranking, the other one is sentence semantic similarity. For emantic similarity, please check the text matching section.

Here we use 5-class Stanford Sentiment Treebank (SST-5) dataset to test the performance of different models.

* [RNN for SST-5](notebooks/nlp/sentiment-classification/SST-5_RNN.ipynb)

* [Bert for SST-5](notebooks/nlp/sentiment-classification/SST-5_Bert.ipynb)

* [Train multiple datasets and evaluate the performance]()

Also check other people's work for sentiment analysis

* [https://github.com/prrao87/fine-grained-sentiment](https://github.com/prrao87/fine-grained-sentiment)

* [https://github.com/gopalanvinay/thesis-vinay-gopalan](https://github.com/gopalanvinay/thesis-vinay-gopalan)

SOTA:

* [RoBERTa-large+Self-Explaining](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)



For more sentiment dataset please check:

* IMDB Movie Review


### Ducument Classification

* Clustering (PCA)

* TF-IDF

* FastText

* Latent Dirichlet Allocation (LDA)

* [RNN for ](notebooks/nlp/Doc_RNN.ipynb)

* [Comparision of document classification](notebooks/nlp/document-classification/ag-news-bert.ipynb)

* [T-Bert] 

* [TechDOfication: Technical Domain Identification]()

* Comparison between TF-IDF, FastText and Bert for Document Classification

### Toxic Comment Classification

* [RNN for](notebooks/nlp/Toxic_RNN.ipynb)

### Textual Entailment/Natural Language Inference

* [RNN for ](notebooks/nlp/natural-language-inference/NLI_RNN.ipynb)

### Sentence Segmentation

### Reading Comprehension

Squad 2.0

* [RNN for SQuAD](notebooks/nlp/reading-comprehension/Squad_RNN.ipynb)

* [GPT-2 for SQuAD2.0](https://medium.com/analytics-vidhya/an-ai-that-does-your-homework-e5fa40c43d17)

### Extractive Summarization


* [Bert for extractive Summarization](notebooks/nlp/summarization/Bert_extractive_summarization.ipynb)


### Text Matching/Semantic Similarity

There are mainly two ways:
  * Input both text into the model.
  * Generate embedding for both text and calculate cosine for both.

#### Paraphrase Indentification


* [RNN for ](notebooks/nlp/paraphrase-identification/QQP_RNN.ipynb)

* [Bert for ](notebooks/nlp/paraphrase-identification/QQP_Bert.ipynb)

* Dual BERT

* QQP Classification with Siamese Bert (Sentenc Transformer)

* Dataset:
  * PAWS: Paraphrase Adversaries from Word Scrambling [`github`](https://github.com/google-research-datasets/paws)

#### Information Retrieval/Semantic Search/Question Answering

Information Retrieval is gradually relying on semantic understanding these days, where user can search information in a question format not just keywords.

There are actually many types of question answering:

* Classification Question Answering:
  One model to trained with all data

  * [Bert for FAQ](notebooks/nlp/question-answering/Bert_FAQ.ipynb)
   

* Close/Open Domain Question Answering:
   
  * Retrieve
    * TF-IDF
    * BM25 
    * Text Embedding: Use [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) to encode the documents once and can be used for similarity comparison later.

  * Ranking or Answer Finding:
     * Use Bert/Sentence-Transformer to re-rank the retrieving documents.
     * QA Bert to find exact answer in the document. 
  * Answer Generation:
     * RAG
     
  
  * Examples:
    * [Bert for covid](notebooks/nlp/question-answering/covid_Bert.ipynb)
    * [Bert for openQA](notebooks/nlp/question-answering/openQA_Bert.ipynb)

Tools:

* [Facebook RAG](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/): Trandformer Style Retrieval Augmented Generation. Models are available on HuggingFace.

* [Hystack](https://github.com/deepset-ai/haystack): Retriever(Elasticsearch, SQL, in memory, FAISS) and Reader(Bert, RobertA, etct) based search system. Check the Demo. 


One application of these pretrained models are as follows:

[Covid-19 Semantic Browser](https://github.com/gsarti/covid-papers-browser)
Semantic search of Covid-19 & SARS-CoV-2 Scientific Papers



## Natural Language Generation

### Utterance Generation

News writing, article gneration

* [GPT-2 for utterance](notebooks/nlp/natural-language-generation/GPT-2_writing.ipynb)

* [CTRL]()

* Reference:
   * [How to Fine-Tune GPT-2 for Text Generation](https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272)

### Machine Translation

* [T5 for translation](notebooks/nlp/natural-language-generation/translation/T5_translation.ipynb)

more research can be found here
* Meta-Learning for Low-Resource Neural Machine Translation [`arXiv`](https://arxiv.org/abs/1808.08437)

* OpenKiwi [`github`](https://github.com/Unbabel/OpenKiwi)

### Abstractive Summarization

* [T5 for_news_summary](notebooks/nlp/natural-language-generation/summarization/T5_for_news.ipynb)


### Paraphrase Generation

* [T5 for paraphrase_generation_fine-tune with QQP](notebooks/nlp/natural-language-generation/paraphrase-generation/T5_small_qqp_paraphrase_generation.ipynb)

## Data to Text Generation

* [WebNLG Challenge 2020]()

### Code Generation

Microsoft and OpenAI uses GPT-3 to launch their Codex service. Honestly I think the more codes embedded in models, the prediction and suggestion will become better.

* [Awesome Machine Learning On Source Code](https://github.com/src-d/awesome-machine-learning-on-source-code)


## More Resourcesfor NLP

### NLP Benchmark

* [Glue](https://gluebenchmark.com/)

* [SuperGlue](https://super.gluebenchmark.com/)

* [XTREME](https://sites.research.google/xtreme)

* [SentEval](https://github.com/facebookresearch/SentEval)

You can check [http://nlpprogress.com/](http://nlpprogress.com/) for more benchmarks and dataset in NLP.

### Data Augmentation

* Snorkel [`github`](https://github.com/snorkel-team/snorkel)
* nlpaug [`github`](https://github.com/makcedward/nlpaug)

### Text Processing

* [Torchtext](): Include data processing utilities and popular datasets 

### Multi functional including Text Preprocessing, Model Training)

* [Pytext](): Deep-learning based NLP modeling framework, including training and export models to Caffe2 format

* [spaCy](https://spacy.io)

* [AllenNLP]()

* [GlounNLP]()

* [Flair]()

* [FARM]()

* [Fast.ai]()

* [nlp-architect](https://github.com/NervanaSystems/nlp-architect)

* [NeMO](): NeMo is designed for conversational AI, but also offers tools for NLP training and exporting

* [Deepavlov](): It offers many pretrained NLP models and you can also fine-tune your own model



[You can check the comparison between libraries](https://luckytoilet.wordpress.com/2018/12/29/deep-learning-for-nlp-spacy-vs-pytorch-vs-allennlp/)

You can choose the library based on which framework they are using(tensorflow, pytorch or both), running speed, support of distributed training or special processing functions.

### Adversarial Attack Test

* [TextAttack](https://github.com/QData/TextAttack)

### pre-trained models

* [Huggingface](https://github.com/huggingface)
* [Tensorflow Hub]()

### DataSet Viewer

* [https://huggingface.co/datasets/viewer/](https://huggingface.co/datasets/viewer/)

### Data and Model Inspection/Visualization

* [LIT](https://github.com/PAIR-code/lit) 

* [Transformers Interpret](https://github.com/cdpierse/transformers-interpret)

* [BertViz](https://github.com/jessevig/bertviz)

### For Production

* [Bert as Service](https://github.com/hanxiao/bert-as-service)

### More Learning Resources

* [awesome-nlp](https://github.com/keon/awesome-nlp#research-summaries-and-trends)

* [Stanford CS224N: Natural Language Processing with Deep Learning Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)

## Challenges

* ICON 2020: 17th International Conference on Natural Language Processing[`link`](https://www.iitp.ac.in/~ai-nlp-ml/icon2020/shared_tasks.html)
* SemEval-2020 International Workshop on Semantic Evaluation [`link`](https://alt.qcri.org/semeval2020/index.php?id=tasks)

## Advanced Research Topics in NLP

### GAN for NLP

### Meta Learning for NLP

### Few Shot Learning for NLP

### Reinforcement Learning for NLP 

#### Imitation Learning for NLP

