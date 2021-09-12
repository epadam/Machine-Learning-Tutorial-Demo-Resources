# Dive into Natural Language Processing

If you are in a hurry:

You can play with these demo apps to see what can NLP do.

* [NLP interactive tasks overview]()

Here are the [links]() to all the notebooks in this guide.

If you are interested in all tasks of NLP, [papers with code](https://paperswithcode.com/area/natural-language-processing) is definitely the good place.


You can also check [Conversational AI](applications/chat_bot.md) and [Knowledge Graph](Knowledge_Graph.md) in another guide. 

If you have time, check the theories and the tasks in detail below: 

This is a tutorial helps you understand the NLP world, from basic knowledge to building a real application, also gives resources to become an advanced researcher. Please notice that audio processing is not included in this tutorial.

If you are already a python programmer, I would suggest you also read through the guide from [AllenNLP](https://guide.allennlp.org/) and [spaCy](https://course.spacy.io/en). First go through it fast to get the general idea of NLP. Then check [spaCy 101](https://spacy.io/usage/spacy-101) to see what the a NLP library offers to build a NLP application. Most libraries offer text preprocessing, cleansing and model training. To see the comparison between frameworks and tools, please check [here](). It is better to focus on one resource first and you will realize it would be also very easy to use other tools. 

Enjoy!!
<ul>
    <li><a href="#language-models">Language Models</a></li>
    <li><a href="#nlp-tasks">NLP Tasks</a></li>
    <li><a href="#Resouces-for-NLP">More Resources for NLP</a></li>
    <li><a href="#Other-research-topics-in-NLP">Other Research Topics in NLP</a></li>
   
</ul>




# Language Models

Language models take the text as input and generate a prediction, a word, a sentence or even an article. 

- [Tokenization](#Tokenization)
- [Statistical Language Models](#Statistical-Language-Models)
  - [Glove](#glove)
  - [N-grams](#N-grams)
  - [FastText](#FastText)
- [Neural Language Models](#Neural-Language-Models)
  - [Word2Vec](#Word2Vec) 
  - [RNN/GRN/LSTM](#RNN/GRN/LSTM)
  - [ELMo](#Elmo)
  - [Transformer](#Transformer)
    - [Attention](#Attention)
    - [Encoder](#Encoder)
    - [Decoder](#Decoder)
  - [Multitask Models](#multitask-models)

## Tokenization

You can check some tools here:

[SentencePiece](https://github.com/google/sentencepiece)

## Statistical Language Models

### Glove
Create word vectors that capture meaning in vector space
Takes advantage of global count statistics instead of only local information

The source code can be found here.

### N-grams

### FastText

## Neural Language Models

### Word2Vec

CBOW & Skip-Gram

![Exploiting Similarities among Languages for Machine Translation](https://miro.medium.com/max/700/1*cuOmGT7NevP9oJFJfVpRKA.png)

### RNN/GRN/LSTM

* RNN 
Check the RNN source cod in a [notebook]()

* GRN

* LSTM

### ELMo

It uses bidirectional LSTM in the seq2seq model

### Transformer 

Transformer is proposed in 2017 by google in the paper "Attention is all you need". 

Below is the architecture of the transformer:
![Transformer Model Architecture](https://www.researchgate.net/publication/342045332/figure/fig2/AS:900500283215874@1591707406300/Transformer-Model-Architecture-Transformer-Architecture-26-is-parallelized-for-seq2seq.png)

Check the code of transformer!

How to train a transformer?


There are many improved transformer models
* [Transformer-XL]():
* [XLNet]():
* [T5]():
* [UniLM]():
* [Turing-LG](): 17b
* [Bart]():
* [MegatronLM]() 8.3b (use GPU parallel training, Doesn't Pytorch or tf support this?) 
* [ELECTRA]():
* [Microsoft MT-DNN]():
* [Megatron-LM from Nvidia]():
* [DeBERTa](https://github.com/microsoft/DeBERTa): 1.5B parameters
* [Meena](): Open domain chatbot by Google based on Transformer.
* [Blender](): Open domain chatbot by Facebook based on Transformer.
* [BlenderBot2](): Open domain chatbot by Facebook based on Transformer. It can search information on the internet and reply up to date information.

### Encoder

Transformer is a seq2seq model which can do NLU and NLG tasks. Only Encoder itself can mostly do NLU jobs.

* Bert

![Bert Pretraining and Fine-Tuning](https://miro.medium.com/max/1200/1*p4LFBwyHtCw_Qq9paDampA.png)





The most famous encoder is Bert proposed by Google. You can check the original paper here. Although it is not the first one introducing pre-train and finetune paradime

Now let's build our own Bert from scratch! [notebook]()

How is Bert pre-trained? [notebook](). There are two tasks

How to fine-tune Bert for downstream tasks? [notebook]()

How to make Bert smaller? [notebook]()

There are many examples here:
* [mobileBert]():
* [TinyBert]():
* [DistillBert]():


There are also some pretrained Bert models with domain specific text, which can perform better :
* [BioBert]():
* [SciBert]():
* [ClinicalBERT]():

One application of these pretrained models are as follows:

[Covid-19 Semantic Browser](https://github.com/gsarti/covid-papers-browser)
Semantic search of Covid-19 & SARS-CoV-2 Scientific Papers


Also improved version of Bert:
* [M-BERT (multi lingual)]():
* [RoBERTa]():
* [BigBird]():
* [ERINE]():
* [Albert]():

Now it's your turn:

Train a Bert model with one of the following dataset.

### Decoder

* GPT/GPT-2/GPT-3

Decoder itself is better at generation tasks. The most famous one is GPT-2.

Let's fine-tune a GPT-2 model. [notebook]()



### Multi-Task models

# NLP Tasks

- [Sequence Tagging](#Sequence-Tagging)
   - [Part-of-speech Tagging](#Part-of-speech-Tagging)
   - [Named Entity Recognition](#Named-Entity-Recognition)
   - [Dependency Parsing](#Dependency-Parsing)
   - [Relation Extraction](#Relation-Extraction)
   - [Grammer correction](#Grammer-correction)
- [Sentence/Paragraph/Document Level Tasks](#Sentence/Paragrah/Document-Level-Tasks)
  - [Classification Tasks](#Classification-tasks)
    - [Ducument Classification](#Ducument-Classification)
    - [Textual Entailment/Natural Language Inference](#Textual-Entailment/Natural-Language-Inference)
    - [Sentiment Analysis](#Sentiment-Analysis)
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

And practice in this notebook  

### Grammer correction

notebook 

### Named Entity Recognition 
notebook 

### Dependency Parsing

notebook 

### Relation Extraction

notebook  


## Sentence/Paragraph/Document Level Tasks

## Classification tasks

### Ducument Classification

* TF-IDF

* FastText

* Latent Dirichlet Allocation (LDA)

* Clustering (PCA)

Toxic Comment Classification
notebook  

### Textual Entailment/Natural Language Inference

notebook 


### Sentiment Analysis

notebook  

### Sentence Segmentation

### Paraphrase indentification

QQP Classification with Siamese Network

notebook 

### Reading Comprehension

notebook  

### Extractive Summarization

notebook 


### Text Matching/Semantic Similarity

Siamese Bert 

Dual BERT


#### Information Retrieval/Semantic Search

The future of information retrieval will evolve to more like question answering system

Query

retriever

Ranking

Embeding the documents and then use Cosine to calculate the similarity

Facebook RAG

Hystack

Sentence Transformer



#### Question Answering

There are actually many types of question answering

* simple question answering

One model to trained with all data

* Open Domain 

You may need Knowledge Graph, please check this [guide]() for more information.

* Close Domain 

It is similar to Information Retrieval but use natural language for processing

-- Covid-19 question answering



## Natural Language Generation

Utterance Generation (which can be seen as a data augmentation)

News writing, article gneration

### Machine Translation

* Meta-Learning for Low-Resource Neural Machine Translation [`arXiv`](https://arxiv.org/abs/1808.08437)

* OpenKiwi [`github`](https://github.com/Unbabel/OpenKiwi)

### Abstractive Summarization

notebook  


### Paraphrase Generation


### Code Generation

* [Awesome Machine Learning On Source Code](https://github.com/src-d/awesome-machine-learning-on-source-code)


## More Resourcesfor NLP

### NLP Benchmark

[Glue]()

[SuperGlue]()

[XTREME]()

[Turing Machine]()

You can check [http://nlpprogress.com/](http://nlpprogress.com/) for more benchmarks and dataset in NLP.

### Data Augmentation

[snorkel]()

### Multi functional including Text Preprocessing, Model Training)

[spaCy](https://spacy.io)

[AllenNLP]()

[GlounNLP]()

[FARM]()

[Fast.ai]()

[nlp-architect](https://github.com/NervanaSystems/nlp-architect)

[Check the comparison between libraries](https://luckytoilet.wordpress.com/2018/12/29/deep-learning-for-nlp-spacy-vs-pytorch-vs-allennlp/)

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

[Huggingface](https://github.com/huggingface)

### Data and Model Inspection

[LIT](https://github.com/PAIR-code/lit) 

Check the tutorial see how to use this for the model we just trained

### For Productions

* [Bert as Service](https://github.com/hanxiao/bert-as-service)

### More Learning Resources

* [awesome-nlp](https://github.com/keon/awesome-nlp#research-summaries-and-trends)

* [Stanford CS224N: Natural Language Processing with Deep Learning Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)


## Other Research Topics in NLP

### GAN for NLP

### Meta Learning for NLP

### Few Shot Learning for NLP

### Reinforcement Learning for NLP 

#### Imitation Learning for NLP

