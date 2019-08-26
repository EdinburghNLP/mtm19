# Part 1 - Translate Using a Pre-trained Model

We'll begin by downloading a pre-trained Nematus model and using it to translate test sets from the [WMT news translation task](http://statmt.org/wmt19/translation-task.html).
The model in question is multilingual and is capable of translating between 100 different languages (albeit poorly for many of the 9,900 translation directions).
We will evaluate translation quality using the standard BLEU metric.

1. In your home directory, create a new directory for this part of the tutorial:

   ```
   $ cd
   $ mkdir lab1.1
   $ cd lab1.1
   ```

1. Download the pre-trained Nematus model and supporting files:
   ```
   $ wget 'http://data.statmt.org/mtm19-nematus-tutorial/multilingual-model.tar.gz'
   $ tar xzf multilingual-model.tar.gz
   ```

   The resulting ```multilingual-model``` directory should contain the following files:

    * ```sentencepiece.model``` is used for pre- and postprocessing text (more on that shortly)
    * ```vocab.json``` specifies the shared source and target vocabulary of the model.
    * ```model-990000.index```, ```model-990000.meta```, and ```model-990000.data-00000-of-00001``` together constitute a TensorFlow checkpoint.
      The ```990000``` part of the name indicates that model was saved after 990,000 training updates.
    * ```model-990000.json``` describes the Nematus configuration used during training.
    * ```model-990000.progress.json``` describes the state of training at the point when this checkpoint was saved to disk.


1. Download and unpack the WMT19 test sets:

   ```
   $ wget 'http://data.statmt.org/wmt19/translation-task/test.tgz'
   $ tar xzf test.tgz
   ```

   See [here](http://statmt.org/wmt19/translation-task.html) for more information about the WMT19 news translation task.

1. Extract plain text from one of the SGML test sets.

   We will use the German-English test set in the commands below.
   The multilingual model is capable of translating all of the WMT19 test sets except for Kazakh-English, so feel free to choose a different one.

   ```
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < sgm/newstest2019-deen-src.de.sgm > test.de
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < sgm/newstest2019-deen-ref.en.sgm > test.en
   ```

   ```test.de``` and ```test.en``` contain one sentence per line.
   The sentences in ```test.en``` are translations of the sentences in ```test.de``` made by a human translator.
   ```test.de``` will be the input to the translation system (once it has been preprocessed) and ```test.en``` will be the reference against which the system's translations will be compared.

1. Encode the text in ```test.de``` as subword units using the sentencepiece model

   ```
   $ ~/sentencepiece/build/src/spm_encode \
     --model multilingual-model/sentencepiece.model \
     --output_format id \
     < test.de \
     > test.de.ids
   ```

   [SentencePiece](https://github.com/google/sentencepiece) is a toolkit for preprocessing text for machine learning applications such as MT.
   It uses an unsupervised segmentation model to learn from data how to split text into subword units.
   For example, it might learn to split the word `stegosaurus` into ```[_ste][go][saurus]``` (the ```_``` in the first segment represents a space).

   The SentencePiece model here uses byte-pair encoding (BPE) with a vocabulary size of 64,000 tokens.
   It was learned from the training data and then used to preprocess that data prior to training the Nematus model.

   If you want to see how the SentencePiece model segments text, then try the above command without the ```--output_format id``` option (and redirect the output to a different file as we need ```test.de.ids``` to contain the id version).

1. Add a tag to the start of every source sentence to indicate the desired target language:

   ```
   $ sed 's/^/<2en> /' test.de.ids > test.de.ids.2en
   ```

   This model follows the approach to multilingual NMT proposed by [Johnson et al (2017)](https://transacl.org/ojs/index.php/tacl/article/view/1081): each input sentence (during training and inference) specifies the target language using a special token at the start of the sentence.
   Our tokens take the form ```<2xx>```, where ```xx``` is a two-letter [ISO 639-1](https://en.wikipedia.org/wiki/ISO_639-1) language code.

   Our preprocessed test set is now in the format required by the translation model.


1. Translate the preprocessed source file:

   ```
   $ cd multilingual-model
   $ CUDA_VISIBLE_DEVICES=0 ~/nematus/nematus/translate.py \
     -m model-990000 \
     -i ../test.de.ids.2en \
     -o ../test.output.ids
   ```

   We are using GPU 0 for translation - if you are using a machine with multiple GPUs then you can use a different device.

   Nematus (and TensorFlow) will produce a lot of output.
   After a short while, you should see some messages indicating translation progress:

   ```
   ...
   I0824 16:30:07.039991 140674440693504 inference.py:173] Translated 80 sents
   I0824 16:30:09.077802 140674440693504 inference.py:173] Translated 160 sents
   I0824 16:30:11.472807 140674440693504 inference.py:173] Translated 240 sents
   I0824 16:30:14.188024 140674440693504 inference.py:173] Translated 320 sents
   I0824 16:30:17.482284 140674440693504 inference.py:173] Translated 400 sents
   I0824 16:30:20.727234 140674440693504 inference.py:173] Translated 480 sents
   I0824 16:30:24.229470 140674440693504 inference.py:173] Translated 560 sents
   I0824 16:30:28.296962 140674440693504 inference.py:173] Translated 640 sents
   I0824 16:30:32.794754 140674440693504 inference.py:173] Translated 720 sents
   I0824 16:30:37.915420 140674440693504 inference.py:173] Translated 800 sents
   I0824 16:30:44.228654 140674440693504 inference.py:173] Translated 880 sents
   I0824 16:30:55.949366 140674440693504 inference.py:173] Translated 960 sents
   I0824 16:31:19.986465 140674440693504 inference.py:173] Translated 1016 sents
   I0824 16:31:20.045289 140674440693504 inference.py:220] Translated 1016 sents in 85.92710661888123 sec. Speed 11.823975459878325 sents/sec
   ```

   ```../test.output.ids``` will contain the translations in the form of SentencePiece vocabulary IDs.

1. Convert the English output from SentencePiece IDs to plain text:
   ```
   $ cd ..
   $ ~/sentencepiece/build/src/spm_decode \
     --model multilingual-model/sentencepiece.model \
     --input_format id \
     < test.output.ids \
     > test.output.txt
   ```

1. Use [Sacrebleu](https://aclweb.org/anthology/W18-6319) to evaluate the translation quality against the reference translation:

   ```
   $ ~/sacreBLEU/sacrebleu.py test.en < test.output.txt
   ```

   If you used the German-English test set, then you will see the following results:

   ```
   BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.3.7 = 22.4 58.5/30.5/17.7/10.7 (BP = 0.930 ratio = 0.932 hyp_len = 36563 ref_len = 39227)
   ```

   The most relevant numbers are the BLEU score (22.4) and the ratio value, which is the ratio between the lengths (in tokens) of the translation and reference documents.
   Here it is 0.932, indicating that the translations are slightly longer than the references, on average.

### Exercises

- Use the system to translate some arbitrary text, for example from a Hindi news site or from a [Japanese Wikipedia](https://ja.wikipedia.org/wiki/メインページ) article.
  The input should be one sentence per line.
  The target language doesn't have to be English: many widely spoken languages are supported - just make sure to use the correct [2-character language identifier ](https://en.wikipedia.org/wiki/ISO_639-1) in the language tag at the start of the source sentence.
  Note: don't expect high quality translation!

To answer the following questions, see the [documentation](https://github.com/EdinburghNLP/nematus#nematustranslatepy--use-an-existing-model-to-translate-a-source-text) for ```translate.py```'s command line options.

- Try adjusting the beam size and minibatch size to improve translation quality and speed (ideally both).

- Are the translations the right length (according to the reference)?
  Try adjusting the length normalization parameter (typically this is in the range 0.5 - 1.0).
  Can you push the length ratio nearer to 1?
  What is the effect on BLEU score?

- Use ```translate.py``` to produce a n-best list.
