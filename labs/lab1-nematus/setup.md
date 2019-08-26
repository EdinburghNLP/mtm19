# Setup - Software Installation

This tutorial requires the following software.
If you are using the Google Cloud VM then this software is already present and can be found in the home directory - you can skip this section.

In most cases, we download the software and run it from it directly from the source directory, rather than actually installing.
You don't have to do the same, just keep in mind that the tutorial assumes they are in the home directory and you will need to adjust some paths to run the commands.

We assume that you already have working versions of Python (version 3.5 or later) and TensorFlow (version 1.12 or later).

## Nematus

We will use the latest master branch of Nematus.

```
git clone https://github.com/EdinburghNLP/nematus
```

This doesn't require any further build or installation steps - we run the scripts directly from our clone of the repository.


## SentencePiece

We use [SentencePiece](https://github.com/google/sentencepiece) for preprocessing text.
Instructions for installation are provided [here](https://github.com/google/sentencepiece/blob/master/README.md#installation).
For the VMs, we built SentencePiece from source but skipped the installation step, instead running the binaries directly from the build directory.

We did:

  ```
  $ git clone https://github.com/google/sentencepiece
  $ cd sentencepiece
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make -j
 ```


## SacreBLEU

We use [SacreBLEU](https://github.com/mjpost/sacreBLEU) for evaluating BLEU scores.
SacreBLEU depends on the portalocker module, which can be installed using the following command:

```
 pip3 install portalocker 
```

If you cannot (or do not want to) install the module site-wide, we recommend installing it in a virtual environment and using that environment for the tutorial.

Like the other toolkits, we run SacreBLEU scripts directly from the source directory and so the only `installation' step is to clone the repository:

```
git clone https://github.com/mjpost/sacreBLEU
```

## Moses Scripts

We make use of the WMT19 test sets, which are provided in SGML format.
The Moses toolkit provides a script to extract plain text from SGML, which we will use in the tutorial.
Since we only need this one script, we will use use the lightweight subset of Moses scripts that was released for use with Marian:

```
git clone http://github.com/marian-nmt/moses-scripts
```

