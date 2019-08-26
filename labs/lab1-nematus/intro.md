# Machine Translation Marathon 2019

## Lab 1 -  Nematus

[Nematus](https://github.com/EdinburghNLP/nematus) is a toolkit for machine translation.
It supports the popular RNN and Transformer architectures and includes advanced features for training and inference.
Nematus is written in Python3 and uses the TensorFlow toolkit.

In this lab you will learn how to translate using pre-trained models and how to train your own models.

### Software Installation

The [setup section](setup.md) covers the installation of software required for the tutorial.
If you are using the Google Cloud VM provided by the MTM organisers, you can skip this section because the software is already installed.

### Part 1 - Translate Using a Pre-trained Model

In [Part 1](part1.md) we will translate a test set using a pre-trained Nematus model.
We will evaluate translation quality using the popular BLEU metric.

### Part 2 -  Train a Low-Resource RNN Model from Scratch

In [Part 2](part2.md) we will train a RNN using a configuation suitable for a low-resource setting.
Having completed this part of the tutorial, it will be straightforward to begin training other types of models (such as the popular Transformer - see [here](https://github.com/EdinburghNLP/wmt17-transformer-scripts/tree/master/training) for a sample Nematus configuration based on Vaswani et al's (2017) Transformer Base model) .

### Part 3 - Fine-Tune an Existing Transformer Model

In [Part 3](part3.md) we will fine-tune an existing model using new data.
The aim is to adapt the model to improve performance on a specific task.
