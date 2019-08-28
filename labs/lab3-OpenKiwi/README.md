![OpenKiwi Logo](https://github.com/Unbabel/OpenKiwi/blob/master/docs/_static/img/openkiwi-logo-horizontal.svg)

--------------------------------------------------------------------------------


# KiwiCutter

Repo for the MT Marathon OpenKiwi Tutorial. 

You can find the slides that accompany this tutorial [here](https://docs.google.com/presentation/d/1oGFDcnaSVDjfnlbL1robDk3-ZWYnWu9ANGZCVs_HMg0/edit?usp=sharing)

Quality estimation (QE) is one of the missing pieces of machine translation: its goal is to evaluate a translation system’s quality without access to reference translations.

OpenKiwi, is a Pytorch-based open-source framework that implements the best QE systems from WMT 2015-18 shared tasks, making it easy to experiment with these models under the same framework. 

Using OpenKiwi and a stacked combination of these models we have achieved state-of-the-art results on word-level QE on the WMT 2018 English-German dataset.

Furthermore, we built on top of this framework to win the WMT 2019 shared task on quality estimation. You can check our approach [here](https://arxiv.org/pdf/1907.10352.pdf)

## Overview of the Tutorial

We are going to split the tutorial in two parts:
* Interactive usage of Kiwi using a Jupyter notebook
* Practical exercises to learn how to develop and make modifications on Kiwi

You can find the notebook in this repo and the description of the exercises under the `exercise` folder.
