# Part 2 - Train a Low-Resource RNN Model from Scratch

In this section we'll train a Kazakh-English translation model from scratch.
We'll be using the News Commentary v14 data provided for the WMT19 news translation task.
This is a low-resource setting as the data set only contains 7,729 sentence pairs (in contrast to the tens of millions of sentence pairs used for the majority of research and production MT systems).
Low-resource MT is a challenging scenario for neural models and requires care with modelling choices - applying the same training settings used in a high-resource system is likely to result in failure to learn anything useful.
We will follow the modelling choices made by [Sennrich and Zhang (2019)](https://www.aclweb.org/anthology/P19-1021) (specifically, the settings are based on system 8 in Table 5).

Note that even with so little data, we won't be able to fully train the system within the time constraints of this tutorial.
If you are using your own computing resources, then you can leave the system to train and you should have a usable system within a few hours.
If you are using the Google Cloud VMs, then we will just run a few training iterations.

1. In your home directory, create a new directory for this part of the tutorial:

   ```
   $ cd
   $ mkdir lab1.2
   $ cd lab1.2
   ```

1. Download and unpack the WMT19 training data and dev / test sets:

   ```
   $ wget 'http://data.statmt.org/news-commentary/v14/training/news-commentary-v14-wmt19.en-kk.tsv.gz'
   $ wget 'http://data.statmt.org/wmt19/translation-task/dev.tgz'
   $ wget 'http://data.statmt.org/wmt19/translation-task/test.tgz'
   $ gunzip news-commentary-v14-wmt19.en-kk.tsv.gz
   $ tar xzf dev.tgz
   $ tar xzf test.tgz
   ```

   See [here](http://statmt.org/wmt19/translation-task.html) for more information about the WMT19 news translation task and about the Kazakh-English data that was provided.

1. Split the tab-separated training data into Kazakh and English files:

   ```
   $ cut -f1 news-commentary-v14-wmt19.en-kk.tsv > train.en
   $ cut -f2 news-commentary-v14-wmt19.en-kk.tsv > train.kk
   ```

1. Extract plain text from the SGML dev and test sets:

   ```
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < dev/newsdev2019-kken-src.kk.sgm > dev.kk
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < dev/newsdev2019-kken-ref.en.sgm > dev.en
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < sgm/newstest2019-kken-src.kk.sgm > test.kk
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < sgm/newstest2019-kken-ref.en.sgm > test.en
   ```

1. Learn Kazakh and English tokenization models from the training data:

   ```
   $ ~/sentencepiece/build/src/spm_train \
     --input train.kk \
     --model_prefix sentencepiece.kk \
     --vocab_size 2000 \
     --character_coverage 1.0 \
     --model_type bpe

   $ ~/sentencepiece/build/src/spm_train \
     --input train.en \
     --model_prefix sentencepiece.en \
     --vocab_size 2000 \
     --character_coverage 1.0 \
     --model_type bpe
   ```

1. Encode the text as subword units using the sentencepiece model (we exclude the English test set as we only need the plain text):

   ```
   $ ~/sentencepiece/build/src/spm_encode \
     --model sentencepiece.kk.model \
     --output_format id \
     < train.kk \
     > train.kk.ids

   $ ~/sentencepiece/build/src/spm_encode \
     --model sentencepiece.kk.model \
     --output_format id \
     < dev.kk \
     > dev.kk.ids

   $ ~/sentencepiece/build/src/spm_encode \
     --model sentencepiece.kk.model \
     --output_format id \
     < test.kk \
     > test.kk.ids

   $ ~/sentencepiece/build/src/spm_encode \
     --model sentencepiece.en.model \
     --output_format id \
     < train.en \
     > train.en.ids

   $ ~/sentencepiece/build/src/spm_encode \
     --model sentencepiece.en.model \
     --output_format id \
     < dev.en \
     > dev.en.ids
   ```

1. Create JSON files specifying the Nematus model vocabulary.

   ```
   python3 ~/nematus/data/build_dictionary.py train.{kk,en}.ids
   ```

1. Create a directory for a Nematus training run.

   ```
   $ mkdir exp1
   ```

1. Copy in the files that will be needed during training:

   ```
   $ cp train.kk.ids \
        train.en.ids \
        dev.kk.ids \
        dev.en.ids \
        dev.en \
        train.kk.ids.json \
        train.en.ids.json \
        exp1
   ```

1. Create a training script

   ```
   $ cd exp1

   $ cat > train.sh << 'EOF'
   #!/usr/bin/env bash
 
   devices=0
 
   CUDA_VISIBLE_DEVICES=$devices python3 ~/nematus/nematus/train.py \
       --model model \
       --datasets train.{kk,en}.ids \
       --dictionaries train.{kk,en}.ids.json \
       --valid_datasets dev.{kk,en}.ids \
       --valid_script ./validate.sh \
       --reload latest_checkpoint \
       --embedding_size 512 \
       --state_size 1024 \
       --optimizer adam \
       --maxlen 50 \
       --loss_function per-token-cross-entropy \
       --lrate 0.0005 \
       --token_batch_size 1000 \
       --valid_token_batch_size 1000 \
       --layer_normalisation \
       --valid_freq 1000 \
       --disp_freq 100 \
       --save_freq 1000 \
       --sample_freq 0 \
       --tie_decoder_embeddings \
       --rnn_enc_depth 1 \
       --rnn_enc_transition_depth 2 \
       --rnn_dec_depth 1 \
       --rnn_dec_base_transition_depth 2 \
       --rnn_dropout_source 0.3 \
       --rnn_dropout_target 0.3 \
       --rnn_dropout_embedding 0.5 \
       --rnn_dropout_hidden 0.5 \
       --label_smoothing 0.2
   EOF
   ```

   As mentioned in the introduction, the settings here are based on system 8 (Table 5) of [Sennrich and Zhang (2019)](https://www.aclweb.org/anthology/P19-1021).
   We have reduced the maximum sentence length from 200 to 50 to speed up training.

   For a description of all command-line arguments available in ```train.py``` see [here](https://github.com/EdinburghNLP/nematus).

   Notice that the training command refers to a script called ```validate.sh```.
   We'll create that next.

1. Create the validation script:

   ```
   $ cat > validate.sh << 'EOF'
   #!/usr/bin/env bash

   ./postprocess.sh < $1 | ~/sacreBLEU/sacrebleu.py --score-only dev.en
   EOF
   ```

   For validation, Nematus translates the Kazakh dev set using beam search.
   The English translations (in the form of SentencePiece vocabulary IDs) are written to a temporary file and the path of the file is provided as the first argument ($1) of the validation script.
   The validation script is free to do anything it likes, but it has to write a  numeric score to standard output.
   In this script, we postprocess the output to produce plain text and then evaluate it against the reference translations using SacreBLEU.

   Since this script makes use of an external script, ```./postprocess.sh```, we also have to create that.

1. Create the postprocessing script:

   ```
   $ cat > postprocess.sh << 'EOF'
   #!/usr/bin/env bash

   ~/sentencepiece/build/src/spm_decode \
     --model ../sentencepiece.en.model \
     --input_format id
   EOF
   ```

1. Make the scripts executable:

   ```
   $ chmod u+x train.sh validate.sh postprocess.sh
   ```

1. Run the training script:

   ```
   $ nohup ./train.sh > stdout.txt 2> stderr.txt &
   ```

1. Monitor the output:

   ```
   $ tail -f stderr.txt
   ```

   You can get the shell prompt back by using ctrl-c to kill the tail command.

   After a few minutes you should see a message indicating that the first 100 updates have been made:

   ```
   I0825 11:31:46.983353 140263320643328 train.py:192] [2019-08-25 11:31:46] Epoch: 0 Update: 100 Loss/word: 0.21039992882614972 Words/sec: 985.5711271141755 Sents/sec: 30.840842589177694
   ```

   After 1000 updates, it will perform validation:

   ```
   ...
   I0825 11:44:23.360356 140263320643328 train.py:338] Validation cross entropy (AVG/SUM/N_SENTS/N_TOKENS): 161.54719513281907 219542.6381855011 1359 41185
   ...
   I0825 11:46:22.641608 140263320643328 train.py:376] Validation script score: 0.3
   ```

   If you are using the Google Cloud VM, then you can leave training to run until you next need the GPU.
   You can kill the script by foregrounding the job and then using ctrl-c:
   ```
   $ fg
   <ctrl-c>
   ```
   Alternatively, you can find its process id using ```nvidia-smi``` and then use ```kill```.

### Exercises

- Adjust the training script to add the lexical model from system 9 (Table 5) of [Sennrich and Zhang (2019)](https://www.aclweb.org/anthology/P19-1021)).

- Try adjusting some of the parameters to improve training speed (you can try this in a separate exp2 directory).
  It's easy to make training faster, but what do you think you can change without sacrificing (too much) translation quality?

- Download a pre-trained version of this model:

  ```
  wget 'http://data.statmt.org/mtm19-nematus-tutorial/pretrained-kk-en.tar.gz'
  ```

  Evaluate its performance on the WMT19 Kazakh-English dev and test sets.

  Take a look at the output.

- The model learned here is very weak.
  Take a look at the [WMT19 system description page](http://statmt.org/wmt19/translation-task.html).
  How could you improve the model with additional data? (hint: you can use data other than Kazakh-English)

- Take a look at the [WMT19 system description papers](http://statmt.org/wmt19/papers.html).
  What did participants do to build strong Kazakh-English systems?
