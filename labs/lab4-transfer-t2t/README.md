# MTM19 tutorial

This tutorial will show you how to use the Tensor2Tensor and how to apply Transfer Learning to low-resource languages. It should be easy to follow for everyone, even people that never trained Machine Translation models.

## Transfer Learning for Low-Resource Languages

The idea of transfer learning is that whenever you have low-resource language pair that has not enough training data, we can pretrain model on ANY other language pair and use the pretrained model as a starting point for the low-resource training.

The tutorial is based on paper:

> Tom Kocmi and Ondřej Bojar. 2018. Trivial Transfer Learning for Low-Resource
Neural Machine Translation. *In Proceedings of the 3rd Conference on Machine
Translation (WMT): Research Papers*, Brussels, Belgium.

## Local Machines

For those using local machines:

> user: .\mtm2019

> pass: machineXYZ (XYZ are last 3 characters of your machine name)

Virtual machine:

> user: mtm

> pass: mtm19

## Virtual Environment Installation

First, we need to prepare virtual environment:

```
virtualenv --python=/usr/bin/python env-gpu
source env-gpu/bin/activate
pip install tensor2tensor[tensorflow_gpu] sacrebleu
```

Before running any of the following commands, do not forget that we need to have the environment sourced.

# Transfer Learning

We are using the following naming convention: The *parent* is the model trained on the high-resource language pair. The *child* is the low-resource model created by the fine-tuning parent with child training data.

In this tutorial, we use English-to-Czech as a parent model and English-to-Estonian as a child model.

This transfer learning tutorial has the following steps:

1. Obtain parallel data
2. Prepare vocabulary (shared between parent and child)
3. Train parent model (English-to-Czech)
4. Prepare data for the child (English-to-Estonian)
5. Transfer parent model parameters
6. Train child model
7. Evaluate child model

## Obtain parallel data

Training data for the tutorial are from [WMT 2019](http://www.statmt.org/wmt19/translation-task.html).

To save time, download prepared data from here:

```
wget http://ufallab.ms.mff.cuni.cz/~kocmanek/mtm19/data.tar.gz
tar -xvzf data.tar.gz
```

We are not going to train the parent model in the lab. Thus the parent training data (CzEng 1.7) are reduced to only 200k sentences.

The child data are 50k sentences randomly selected from English-Estonian corpora.

The development and test sets are from WMT 2019 News Shared Task.

### T2T folders explained

Tensor2tensor uses 4 different folders. `t2t_datagen` is used for a temporary storage of data. `t2t_data` is used for preprocessed training corpora and vocabularies. `t2t_train` is used to store trained models. Lastly, `codebase` is used for user-specific definition of problems. The folders can be named differently.

## Preparation of vocabulary

Wordpieces (or BPE) can segment any text. We can use parent vocabulary to segment child training data. However, it is better to use vocabulary that contains language-specific subwords. In order to have the vocabulary, that contains parent and child subwords, we need to prepare shared vocabulary in advance of parent model training.

### Task 1

Take *equal amount* of parallel data from each training corpora and combine them into one file called `mixed.txt`. Transfer learning is going to work even when the vocabulary is not balanced. However, you get worse performance.

Then run:

```
python generate_vocab.py
```

This creates vocabulary `vocab.cseten.wp` in `t2t_data` with 32k subwords. The vocabulary contains subwords from all languages of parent and child (Czech, English, Estonian). Note that we could improve the performance of the child model by making the vocabulary smaller as noted by Sennrich and Zhang (2019).

Check the wordpieces vocabulary:

```
less t2t_data/vocab.cseten.wp
```

## Train parent model (English-to-Czech)

Now, as you prepared vocabulary, you can preprocess dataset with the wordpieces and start parent training. The training would take couple of days/weeks. Thus we skip this step and download already trained parent model:

```
wget http://ufallab.ms.mff.cuni.cz/~kocmanek/mtm19/parent.tar.gz
tar -xvzf parent.tar.gz
cp t2t_train/parent/vocab.cseten.wp t2t_data/vocab.cseten.wp
```

This parent model has been trained for 1 700 000 steps (approximately 25 days on a single GPU).

Note that we need to override generated vocabulary from the previous step with the one used for training the parent model. In order to make sure that the vocabulary has the same subwords as the trained parent model.

This pipeline can be used for other real-life training. Therefore `./train_parent.sh` command can be used to train the parent English-to-Czech model (please beware that parent training data you have at the moment are downsampled, for actual training use whole CzEng 1.7 corpora).

## Test performance of parent model

We have parent model trained; we can try to translate English testset:

```
MODEL=parent FILE=t2t_datagen/test-newstest2018-encs.src ./translate_file.sh
```

The translation does not use a beam search. Thus it should translate the file in few minutes. After that, see the output of the translation in:

```
less output.translation
```

We evaluate the translation by sacreBLEU. This tool automatically downloads WMT testset and evaluate the translation on a correct testset:

```
cat output.translation | sacrebleu -t wmt18 -l en-cs --score-only
```


## Prepare data for child model (English-to-Estonian)

Before training the child model, Tensor2Tensor needs to preprocess training data. For this, we are going to specify our training problem and preprocess data.
We define our problem in directory `codebase`.

### Task 2

Open file `codebase/child.py`. We have a template of the problem that extends standard TranslateProblem of T2T. We need to specify where the training and development data are and also specify the vocabulary.

The template does not work, yet. We need to first modify rows 8 and 9 by providing the correct name of the child training corpora. T2T will look for the corpora in folder `t2t_datagen` where temporary files are stored. 

Furthermore, provide the correct name of vocabulary we generated earlier on row 16.

The last step is to register the problem for tensor2tensor by adding the following line into `codebase/__init__.py`.

```
import . from child
```

We preprocess the corpus by the following command:

```
t2t-datagen --data_dir=t2t_data --t2t_usr_dir=codebase --tmp_dir=t2t_datagen --problem=child_problem
```

It will preprocess the corpus and store it in `t2t_data`. Notice, that tensor2tensor renamed the problem from *childProblem* to *child_problem*.

## Transfer parent model

At this point, we have English-to-Czech model and preprocessed child data. We need to make the last step: transfer parameters from parent to child.

We use the functionality of tensor2tensor framework, which automatically continues the training from the last checkpoint in the training directory. Therefore we need to provide the tensor2tensor with the checkpoint to the parent model. 

### Task 3

Copy last checkpoint from `t2t_train/parent` into `t2t_train/child`. You need to copy all files except for events\*

If the child folder is empty, T2T will train the model from scratch (baseline can be trained by `./train_baseline_model.sh`).

## Train child model

The last step is to train the child model. It is done by running the following script. Before running, look inside to know what it does.

The script calls `t2t-trainer` with default parameters. We use *transformer_big_single_gpu*, other parameters are set exactly as for the parent model. You can increase the batch size if you have access to GPU with bigger memory.

For the child model training, we need GPU.

```
./train_child_model.sh
```

The training is going to take several hours/days to train the model. The script stops after 50k steps. If you use this pipeline for real-life training, you need to watch the performance on the development set in order to prevent overfitting, which often happens on low-resource language pairs.

## Evaluate child

After the training, the final model can be tested by the following command:

```
MODEL=child FILE=t2t_datagen/test-newstest2018-enet.src ./translate_file.sh
```

You can change the script's beam size to a higher value (for example 4) and obtain better output.

The BLEU can be computed by:

```
cat output.translation | sacrebleu -t wmt18 -l en-et --score-only
```

After only 4000 steps (several hours of training) we obtain a score of 12.5 BLEU, and after 30 000 steps, you should get around a score of 16.4 BLEU.

To summarize, we trained English-to-Estonian neural machine translation with only 50k training sentences. If trained from scratch without the transfer learning and hyperparameter tuning, we would obtain roughly 5 BLEU points. 


