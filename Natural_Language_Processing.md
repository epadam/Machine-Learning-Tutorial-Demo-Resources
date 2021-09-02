# Natural Language Processing

This is a tutorial helps you understand the NLP world, from basic knowledge to building a real application, also gives resources to become an advanced researcher. Please notice that audio processing is not included in this tutorial.

If you are already a python programmer, I would suggest you also read through the guide from [AllenNLP](https://guide.allennlp.org/) and [spaCy](https://course.spacy.io/en). First go through it fast to get the general idea of NLP. Then check [spaCy 101](https://spacy.io/usage/spacy-101) to see what the a NLP library offers to build a NLP application. Most libraries offer text preprocessing, cleansing and model training.

I would also give links of other learning resources and libraries. But it is better to focus on one resource first and you will realize it would be also very easy to use other tools. 

You can either check the language models section first or NLP tasks section. 

## Language Models (Word Embedding)

Language models take the text as input and generate a prediction, a word, a sentence or even an article. 

## Statistical Language Models

## Glove

## N-grams

## FastText


## Neural Language Models

### Data Processing and Augmentation

Neural Network requires many data for training. Therefore, just like image task, data augmentation is very helpful.

#### snorkel

### Word2Vec

CBOW & Skip-Gram

### RNN

Check deep learning.md

Use case

### GRN

Check deep learning.md

Use case

### LSTM, ELMo

Check deep learning.md

Use case

### Transformer 

Model Pretraining (Self supervised)

Use case
Transformer is not only used in NLP
but also used in chemistry and biology research

#### Encoder

Check deep learning.md, Model Pretraining (Self supervised)

Variation of Bert ()
SciBert, BioBert

* Bert

Build your own Bert from scratch

* XLNet
* Megatron-LM

#### Decoder

GPT/GPT-2/GPT-3

Use case

### Multi-Task models





## NLP Tasks 

The goal of all NLP tasks is to understand the text(each word, relation between words, sentiment or the topic) and generate the text we want. The tasks below try to understand the text from word level, sentence level and document level. 

### Word Level Tasks

#### Part-of-speech Tagging (POS)

notebook  Demo

Open Dataset

#### Named Entity Recognition (NER)

notebook  Demo

Open Dataset

#### Dependecy Parsing

notebook  Demo

Open Dataset

### Relation Extraction

notebook  Demo

Open Dataset

### Sentence/Paragrah Level Tasks

#### Text Classification

Clustering (PCA)

Toxic Comment Classification
notebook  Demo

Open Dataset

#### Grammer correction

notebook  Demo

Open Dataset

#### Textual Entailment/Inference

notebook  Demo

Open Dataset

#### Text Matching

#### Paraphrase

QQP Classification with Siamese Network

notebook  Demo

#### Sentence Segmentation

#### Sentiment Analysis

notebook  Demo

Open Dataset

#### Question Answering

notebook  Demo

Open Dataset

#### Reading Comprehension

notebook  Demo

Open Dataset

#### Machine Translation

* Meta-Learning for Low-Resource Neural Machine Translation [`arXiv`](https://arxiv.org/abs/1808.08437)

* OpenKiwi [`github`](https://github.com/Unbabel/OpenKiwi)

notebook  Demo

Open Dataset

#### Automatic Summarization


##### Extractive Summarization

notebook  Demo

Open Dataset

##### Abstractive Summarization

notebook  Demo

Open Dataset

Please check this post to see how to use summerizer using different models

#### Information Retrieval

Query

Ranking

Recall


notebook  Demo

Open Dataset

#### Natural Language Generation

Utterance Generation (which can be seen as a data augmentation)

News writing, article gneration


### Document Level Tasks

### Ducument Classification

#### TF-IDF

FastText

atent Dirichlet Allocation (LDA)

Cosine similar


## Other research topics to improve performance of NLP

### GAN for NLP

### Meta Learning for NLP

### Few Shot Learning for NLP

### Reinforcement Learning for NLP 

#### Imitation Learning for NLP

### Knowledge Graph for Chatbot

* Knowledge Graphs in Natural Language Processing @ACL 2019 [`arXiv`](https://medium.com/@mgalkin/knowledge-graphs-in-natural-language-processing-acl-2019-7a14eb20fce8)

### NLP Benchmark

Glue

SuperGlue

## Deployment of NLP models


## Conversational AI

Since this is an huge topic, please check this [tutorial](applications/chat_bot.md) 


## Open Source Tools for NLP

### Multi functional (Preprocessing, Training)

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

### Other Resources

* [Bert as Service](https://github.com/hanxiao/bert-as-service)

* [awesome-nlp](https://github.com/keon/awesome-nlp#research-summaries-and-trends)

* [Stanford CS224N: Natural Language Processing with Deep Learning Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)

### Other Applications using NLP

* [Awesome Machine Learning On Source Code](https://github.com/src-d/awesome-machine-learning-on-source-code)




