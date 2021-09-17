# Dive into Natural Language Processing

If you are in a hurry:

You can play with a Streamlit demo to see what can NLP do.

* [NLP interactive tasks overview]() deployed on Heroku, you can find the source [here]().

Here are the [links]() to all the notebooks in this guide.

You can also check [Conversational AI](applications/chat_bot.md) and [Knowledge Graph](Knowledge_Graph.md) in another guide. 

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
  - [N-grams](#N-grams)
  - [Bag of Words](#bag-of-words)
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
* 
WordPiece

[SentencePiece](https://github.com/google/sentencepiece)



## Statistical Language Models

### N-gram

### Bag of Words

### Glove
Create word vectors that capture meaning in vector space
Takes advantage of global count statistics instead of only local information

The source code can be found here.

### FastText

N-gram + Softmax

## Neural Language Models

### Word2Vec

CBOW & Skip-Gram

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

It uses bidirectional LSTM in the seq2seq model

### Transformer 

Transformer is proposed in 2017 by google in the paper "Attention is all you need". 

Below is the architecture of the transformer:

<img src="https://www.researchgate.net/publication/342045332/figure/fig2/AS:900500283215874@1591707406300/Transformer-Model-Architecture-Transformer-Architecture-26-is-parallelized-for-seq2seq.png" alt="Transformer Model Architecture" width="600"/>

Check the code of transformer!

How to train a transformer?


There are many improved transformer models
* [Transformer-XL]():
* [UniLM]():
* [XLNet]():
* [ELECTRA]():
* [Bart]():
* [MegatronLM]() 8.3b (use GPU parallel training, Doesn't Pytorch or tf support this?) 
* [Microsoft MT-DNN](): Multi-tasking model
* [Megatron-LM from Nvidia]():
* [T5]():
* [DeBERTa](https://github.com/microsoft/DeBERTa): 1.5B parameters
* [Turing-LG](): 17b
* [Meena](): Open domain chatbot by Google based on Transformer.
* [Blender](): Open domain chatbot by Facebook based on Transformer.
* [BlenderBot2](): Open domain chatbot by Facebook based on Transformer. It can search information on the internet and reply up to date information.

### Encoder

Transformer is a seq2seq model which can do NLU and NLG tasks. Only Encoder itself can mostly do NLU jobs.

* Bert

<img src="https://miro.medium.com/max/1200/1*p4LFBwyHtCw_Qq9paDampA.png" alt="Bert Pretraining and Fine-Tuning" width="600"/>






The most famous encoder is Bert proposed by Google. You can check the original paper here. Although it is not the first one introducing pre-train and finetune paradime

Now let's build our own Bert from scratch! [notebook]()

How is Bert pre-trained? [notebook](). There are two tasks

How to fine-tune Bert for downstream tasks? [notebook]()

How to make Bert smaller? [notebook]()

There are many examples here:
* [mobileBert]():
* [TinyBert]():
* [DistillBert]():
* [Albert]():


There are also some pretrained Bert models with domain specific text, which can perform better :
* [BioBert]():
* [SciBert]():
* [ClinicalBERT]():




Also improved version of Bert:
* [M-BERT (multi lingual)]():
* [RoBERTa]():
* [BigBird]():
* [ERNIE]():


Now it's your turn:

Train a Bert model with one of the following dataset.

### Decoder

* GPT/GPT-2/GPT-3

Decoder itself is better at generation tasks. The most famous one is GPT-2.

Control text generation:

* Beam Search

* random

Let's fine-tune a GPT-2 model. [notebook]()


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

* [Spacy and_RNN for ](notebooks/nlp/Spacy_POS_RNN.ipynb)


### Grammer correction

* [LSTM for Cola](notebooks/nlp/Cola_LSTM.ipynb)


### Named Entity Recognition 

NER is very useful for chatbot or building or querying knowledge graph.

* [RNN for_NER](notebooks/nlp/NER_RNN.ipynb)

* [Bert for_](notebooks/nlp/NER_Bert.ipynb)


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

* [RNN for SST-5](notebooks/nlp/SST-5_RNN.ipynb)

* [Bert for SST-5]()

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

* [T-Bert] 

* Comparison between TF-IDF, FastText and Bert for Document Classification

### Toxic Comment Classification

* [RNN for](notebooks/nlp/Toxic_RNN.ipynb)

### Textual Entailment/Natural Language Inference

* [RNN for ](notebooks/nlp/NLI_RNN.ipynb)

### Sentence Segmentation

### Reading Comprehension

Squad 2.0

* [RNN for SQuAD](notebooks/nlp/Squad_RNN.ipynb)

* [GPT-2 for SQuAD2.0](https://medium.com/analytics-vidhya/an-ai-that-does-your-homework-e5fa40c43d17)

### Extractive Summarization


* [Bert for ](notebooks/nlp/Bert_extractive_summarization.ipynb)


### Text Matching/Semantic Similarity

There are mainly two ways:
  * Input both text into the model.
  * Generate embedding for both text and calculate cosine for both.

#### Paraphrase Indentification


* [RNN for ](notebooks/nlp/QQP_RNN.ipynb)

* [Bert for ](notebooks/nlp/QQP_Bert.ipynb)

* Dual BERT

* QQP Classification with Siamese Bert (Sentenc Transformer)

#### Information Retrieval/Semantic Search

The future of information retrieval will evolve to more like question answering system

* retriever (BM25, TF-IDF)

* Ranking 

Embeding the documents and then use Cosine to calculate the similarity

* [Sentence Transformer]()

* [Facebook RAG]()

* [Hystack]()


One application of these pretrained models are as follows:

[Covid-19 Semantic Browser](https://github.com/gsarti/covid-papers-browser)
Semantic search of Covid-19 & SARS-CoV-2 Scientific Papers


#### Question Answering

There are actually many types of question answering

* Classification Question Answering
  One model to trained with all data

  * [Bert for FAQ](notebooks/nlp/Bert_FAQ.ipynb)


* Open Domain 

   You may need Knowledge Graph, please check this [guide]() for more information.

   * [Bert for openQA](notebooks/nlp/openQA_Bert.ipynb)


* Close Domain 

It is similar to Information Retrieval but use natural language for processing

-- Covid-19 question answering

* [Bert for covid](notebooks/nlp/covid_Bert.ipynb)


## Natural Language Generation

### Utterance Generation

News writing, article gneration

* [GPT-2 for utterance](notebooks/nlp/GPT-2_writing.ipynb)

* [CTRL]()

### Machine Translation

* [T5 for translation](notebooks/nlp/T5_translation.ipynb)

more research can be found here
* Meta-Learning for Low-Resource Neural Machine Translation [`arXiv`](https://arxiv.org/abs/1808.08437)

* OpenKiwi [`github`](https://github.com/Unbabel/OpenKiwi)

### Abstractive Summarization

* [T5 for_news_summary](notebooks/nlp/T5_for_news.ipynb)


### Paraphrase Generation

* [T5 for paraphrase_generation_fine-tune with QQP](notebooks/nlp/T5_small_qqp_paraphrase_generation.ipynb)

## Data to Text Generation

* [WebNLG Challenge 2020]()

### Code Generation

Microsoft and OpenAI uses GPT-3 to launch their Codex service. Honestly I think the more codes embedded in models, the prediction and suggestion will become better.

* [Awesome Machine Learning On Source Code](https://github.com/src-d/awesome-machine-learning-on-source-code)


## How to develop a production level NLP services

Let's build a technical document service
1. Data Labeling and Preprocessing (let's use label studio and snorkel)
2. Model Training and tracking (Kubeflow, mlflow)
3. Model evaluation, Inspection (LIT, tensorboard)
4. Model Deployment and Monitoring (convert the model to torchscript, and use torchserve for the serving)

keras version

tensorflow version

pytorch version

## More Resourcesfor NLP

### NLP Benchmark

* [Glue](https://gluebenchmark.com/)

* [SuperGlue](https://super.gluebenchmark.com/)

* [XTREME](https://sites.research.google/xtreme)

* [SentEval](https://github.com/facebookresearch/SentEval)

You can check [http://nlpprogress.com/](http://nlpprogress.com/) for more benchmarks and dataset in NLP.

### Data Augmentation

* [Snorkel]()

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

<table border="0" align="center">
<tr>
    <td style="padding:15px;"><a href="#tokenization"> </a></td>
    <td style="padding:15px;"><a href="#word-embeddings---word2vec">AllenNLP</td>
    <td style="padding:15px;"><a href="#word-embeddings---glove">SpaCy</a></td>
    <td style="padding:15px;"><a href="#word-embeddings---elmo">Farm</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#rnn-lstm-gru">Distributed Training</a></td>
    <td style="padding:15px;"><a href="#packing-padded-sequences">X</a></td>
    <td style="padding:15px;"><a href="#attention-mechanism---luong">X</a></td>
    <td style="padding:15px;"><a href="#attention-mechanism---bahdanau">Attention Mechanism - Bahdanau</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#pointer-network">Speed</a></td>
    <td style="padding:15px;"><a href="#transformer">Transformer</a></td>
    <td style="padding:15px;"><a href="#gpt-2">GPT-2</a></td>
    <td style="padding:15px;"><a href="#bert">BERT</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#topic-modelling-using-lda">Preprocessing</a></td>
    <td style="padding:15px;"><a href="#principal-component-analysispca">Principal Component Analysis (PCA)</a></td>
    <td style="padding:15px;"><a href="#naive-bayes-algorithm">Naive Bayes</a></td>
    <td style="padding:15px;"><a href="#data-augmentation-in-nlp">Data Augmentation</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#sentence-embeddings">Sentence Embeddings</a></td>
</tr>

</table>

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


## Advanced Research Topics in NLP

### GAN for NLP

### Meta Learning for NLP

### Few Shot Learning for NLP

### Reinforcement Learning for NLP 

#### Imitation Learning for NLP

