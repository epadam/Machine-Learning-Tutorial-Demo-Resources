# Natural Language Processing

This is a tutorial helps you understand the NLP world, from basic knowledge to building a real application, also gives resources to become an advanced researcher.

If you are already a python programmer, I would suggest you also read through the guide from [AllenNLP](https://guide.allennlp.org/) and [spaCy](https://course.spacy.io/en). First go through it fast to get the general idea of NLP. Then check [spaCy 101](https://spacy.io/usage/spacy-101) to see what the a NLP library offers to build a NLP application.

I would also give links of other learning resources and libraries. But it is better to focus on one resource first and you will realize it would be also very easy to use other tools. 

You can check the 

## Language Models (Word Embedding)



## Statistical Language Models

## Glove

## N-grams

## FastText

## Neural Language Models

### Data Processing and Augmentation for Neural Language Models Training

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

### Transformer, T5

Check deep learning.md

Use case
Transformer is not only used in NLP
but also used in chemistry and biology research

### Bert, XLNet

Check deep learning.md, Model Pretraining (Self supervised)

Variation of Bert ()
SciBert, BioBert

Use case

### GPT/GPT-2/GPT-3

Model Pretraining (Self supervised)

Use case

### Multi-Task models

## Tasks 

### Word Level Tasks

#### Part-of-speech Tagging 

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

#### Paraphrase

QQP Classification with Siamese Network

notebook  Demo

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

notebook  Demo

Open Dataset

#### Natural Language Generation

Utterance Generation (which can be seen as a data augmentation)

News writing, article gneration


### Document Level Tasks

### Ducument Classification

#### TF-IDF

atent Dirichlet Allocation (LDA)

Cosine similar


### NLP Benchmark

SuperGlun


### Dialog System (Chatbot)

Chatbot is probably the most challenging application in NLP. It includes multiple NLP tasks, mostly NLU and NLG related tasks.
It ranges from simple Q&A chatbot, task specific cahtbot to open domain chatbot. Google Assitant and Alexa are multi-task chatbot.

##### NLU

##### Dialog State Tracking

##### Dialog Policy

##### NLG for Chatobt

notebook  Demo

Check also some other examples here

小冰

##### Open Source Tools

* [DeepPavlov](http://deeppavlov.ai/)

* ParlAI [`github`](https://github.com/facebookresearch/ParlAI)

* [Rasa](https://rasa.com/)

* Nemo

### Related Research

#### Open Domain Chatbot

* [Comparison of Transfer-Learning Approaches for Response Selection in Multi-Turn Conversations](http://workshop.colips.org/dstc7/papers/17.pdf)

* [Towards a Conversational Agent that Can Chat About…Anything](https://ai.googleblog.com/2020/01/towards-conversational-agent-that-can.html)

#### Chatbot with Knowledge Graph

Knowledge grpah gives the power of chatbot. Please check here 



### GAN for NLP

### Meta Learning for NLP

### Reinforcement Learning for NLP 

#### Imitation Learning for NLP



Knowledge Graphs in Natural Language Processing @ACL 2019 [`arXiv`](https://medium.com/@mgalkin/knowledge-graphs-in-natural-language-processing-acl-2019-7a14eb20fce8)


## Open Source Tools

### Multi functional (Preprocessing, Training)

[spaCy](https://spacy.io)

[AllenNLP]

[nlp-architect](https://github.com/NervanaSystems/nlp-architect)

[Check the comparison between libraries](https://luckytoilet.wordpress.com/2018/12/29/deep-learning-for-nlp-spacy-vs-pytorch-vs-allennlp/)


### Data Augmentation

Snorkel


### pre-trained models

[Huggingface](https://github.com/huggingface)

### Model training

[FARM]

### Data and Model Inspection

[LIT](https://github.com/PAIR-code/lit) Check the tutorial

### Deployment

[Bert as Service](https://github.com/hanxiao/bert-as-service)




## Resources

* [awesome-nlp](https://github.com/keon/awesome-nlp#research-summaries-and-trends)

* [Stanford CS224N: Natural Language Processing with Deep Learning Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)

* [Awesome Machine Learning On Source Code](https://github.com/src-d/awesome-machine-learning-on-source-code)



## Audiio

### Speech Recognition (ASR)

Alibaba-MIT-Speech [`github`](https://github.com/alibaba/Alibaba-MIT-Speech)

DeepSpeech [`github`](https://github.com/mozilla/DeepSpeech)

Wav2letter++ [`github`](https://github.com/facebookresearch/wav2letter)

Real-Time-Voice-Cloning [`github`](https://github.com/CorentinJ/Real-Time-Voice-Cloning?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)

#### On-device wake word detection:

porcupine [`github`](https://github.com/Picovoice/porcupine)

### Speaker Diarization

Joint Speech Recognition and Speaker Diarization via Sequence Transduction [`arXiv`](https://arxiv.org/abs/1907.05337) 

### Voice Conversion

deep-voice-conversion [`github`](https://github.com/andabi/deep-voice-conversion)

## Speech Synthesis (TTS)

[WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) 

FastSpeech: Fast, Robust and Controllable Text to Speech [`arXiv`](https://arxiv.org/abs/1905.09263)






