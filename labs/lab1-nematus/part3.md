# Part 3 - Fine-Tune an Existing Transformer Model

In this section we'll take the multilingual model from Part 1 and continue training on new data to improve performance for a particular application.
The application in question is Gujarati-to-English news translation, which was one of the WMT19 shared tasks.

The existing model was trained using approximately 380k sentence pairs of Gujarati-English (along with large amounts of data for other language pairs), which was taken from the [OPUS collection](http://opus.nlpl.eu).
While this sounds like a lot of data (at least compared to the 7k of Kazakh-English used in Part 2), the training data was all derived from GNOME, KDE, and Ubuntu localization files, which limits its suitability for news translation.

As in part 2, we won't be able to fully train the system within the time constraints of this tutorial.
If you are using your own computing resources, then you can leave the system to train and you should have a much improved Gujarati-to-English system within a few hours.
If you are using the Google Cloud VMs, then we will just run a few training iterations.

1. In your home directory, create a new directory for this part of the tutorial:

   ```
   $ cd
   $ mkdir lab1.3
   $ cd lab1.3
   ```

   Note: if you are using a VM, then the lab1.3 directory may already exist.
   If it does, then remove it with the command ```rm -rf ~/lab1.3```


1. Download the pre-trained Nematus model and supporting files.
   These are the same files used in Part 1, so you can reuse those files if you have already completed that part of the tutorial.
   ```
   $ wget 'http://data.statmt.org/mtm19-nematus-tutorial/multilingual-model.tar.gz'
   $ tar xzf multilingual-model.tar.gz
   ```

1. Download and decompress two of the Gujarati-English training corpora provided by the WMT19 organisers:

   ```
   $ wget 'http://data.statmt.org/wmt19/translation-task/wikipedia.gu-en.tsv.gz'
   $ wget 'http://data.statmt.org/wmt19/translation-task/govin-clean.gu-en.tsv.gz'
   $ gunzip wikipedia.gu-en.tsv.gz
   $ gunzip govin-clean.gu-en.tsv.gz
   ```

   The [WMT19 news translation task page](http://statmt.org/wmt19/translation-task.html) gives the following descriptions of the ```wikipedia``` corpus

   > A parallel corpus extracted from wikipedia and contributed by Alexander Molchanov of PROMT.

   and the ```govin-clean``` corpus:

   > A crawled corpus produced for this task. It is very noisy, but contains some parallel data. A cleaned version is also available, cleaned using language detection and simple length heuristics. We recommened that you either use the cleaned version, or apply your own cleaning to the raw version.


1. Split the tab-separated training data into Gujarati and English files and concatenate the two corpora:

   ```
   $ cut -f1 wikipedia.gu-en.tsv > finetune.gu
   $ cut -f2 wikipedia.gu-en.tsv > finetune.en
   $ cut -f1 govin-clean.gu-en.tsv >> finetune.gu
   $ cut -f2 govin-clean.gu-en.tsv >> finetune.en
   ```

   This is our main fine-tuning data - we will mix it with some multilingual data later.
   We now have 28,683 sentence pairs of Gujarati-English.
   This is less than was used in training the original system, but has the advantage of being more relevant to the task.

1. Download the WMT19 dev and test sets.
   Again, if you already have these files from a previous part of the tutorial then you can skip the download step and reuse them.

   ```
   $ wget 'http://data.statmt.org/wmt19/translation-task/dev.tgz'
   $ wget 'http://data.statmt.org/wmt19/translation-task/test.tgz'
   $ tar xzf dev.tgz
   $ tar xzf test.tgz
   ```

1. Extract plain text from the SGML dev and test sets:

   ```
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < dev/newsdev2019-guen-src.gu.sgm > dev.gu
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < dev/newsdev2019-guen-ref.en.sgm > dev.en
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < sgm/newstest2019-guen-src.gu.sgm > test.gu
   $ ~/moses-scripts/scripts/generic/input-from-sgm.perl < sgm/newstest2019-guen-ref.en.sgm > test.en
   ```

1. Download and decompress a sample of the training data that was used to train the multilingual model:

   ```
   $ wget 'http://data.statmt.org/mtm19-nematus-tutorial/opus.sample.tsv.gz'
   $ gunzip opus.sample.tsv.gz
   ```

   This is a random sample of 200k sentence pairs from the original multilingual training corpus.
   It contains a mixture of languages (as can be seen in the third field of the tab-separated file).
   This data is already preprocessed.

1. Encode the fine-tuning data as subword units using the SentencePiece model:

   ```
   $ ~/sentencepiece/build/src/spm_encode \
     --model multilingual-model/sentencepiece.model \
     --output_format id \
     < finetune.gu \
     > finetune.gu.ids

   $ ~/sentencepiece/build/src/spm_encode \
     --model multilingual-model/sentencepiece.model \
     --output_format id \
     < finetune.en \
     > finetune.en.ids
   ```

1. Mix the fine-tuning data with the multilingual sample to produce the training data:

   ```
   $ cut -f1 opus.sample.tsv > train.src.ids.2xx
   $ cut -f2 opus.sample.tsv > train.tgt.ids
   $ seq 10 | xargs -i sed 's/^/<2en> /' finetune.gu.ids >> train.src.ids.2xx
   $ seq 10 | xargs -i cat finetune.en.ids >> train.tgt.ids
   ```

   We are oversampling the Gujarati-English data by a factor of 10 to better balance out the original data.
   This gives us a total of 486,830 sentence pairs, roughly in a 3:2 ratio of Gujarati-English to mixed multilingual.

1. Encode the dev and test sets as subword units using the sentencepiece model:

   ```
   $ ~/sentencepiece/build/src/spm_encode \
     --model multilingual-model/sentencepiece.model \
     --output_format id \
     < dev.gu \
     > dev.gu.ids

   $ ~/sentencepiece/build/src/spm_encode \
     --model multilingual-model/sentencepiece.model \
     --output_format id \
     < dev.en \
     > dev.en.ids

   $ ~/sentencepiece/build/src/spm_encode \
     --model multilingual-model/sentencepiece.model \
     --output_format id \
     < test.gu \
     > test.gu.ids
   ```

1. Add the ```<2en>``` tags to the dev and test source files:

   ```
   $ sed 's/^/<2en> /' dev.gu.ids > dev.gu.ids.2en
   $ sed 's/^/<2en> /' test.gu.ids > test.gu.ids.2en
   ```

1. Truncate the dev set:

   ```
   $ cut -d' ' -f 1-100 dev.gu.ids.2en > dev-trunc.gu.ids.2en
   $ cut -d' ' -f 1-100 dev.en.ids > dev-trunc.en.ids
   ```

   The original dev set contains several very long sentences, which can cause Nematus to exhaust the GPU's memory.
   This limits sentences to a maximum of 100 tokens, keeping memory consumption in check.

1. Create a directory for a Nematus training run  and copy in the files that will be needed during training:

   ```
   $ mkdir exp1

   $ cp train.src.ids.2xx \
        train.tgt.ids \
        dev-trunc.gu.ids.2en \
        dev-trunc.en.ids \
        dev.en \
        multilingual-model/vocab.json \
        multilingual-model/model-990000* \
        exp1
   ```

1. In the training directory, create a TensorFlow checkpoint file:

   ```
   $ cd exp1

   $ cat > checkpoint << 'EOF'
   model_checkpoint_path: "/home/mtm/lab1.3/exp1/model-990000"
   all_model_checkpoint_paths: "/home/mtm/lab1.3/exp1/model-990000"
   EOF
   ```

   The checkpoint file will be used by Nematus (via TensorFlow) to determine the location of the model checkpoint files to use for loading the model parameters when training is restarted.
   If you are not using the Google Cloud VM, you should adjust the paths for ```model_checkpoint_path``` and ```all_model_checkpoint_paths```.

1. Create a training script:

   ```
   $ cat > train.sh << 'EOF'
   #!/usr/bin/env bash
   
   devices=0
   
   CUDA_VISIBLE_DEVICES=$devices python3 ~/nematus/nematus/train.py \
       --source_dataset train.src.ids.2xx \
       --target_dataset train.tgt.ids \
       --dictionaries vocab.json vocab.json \
       --model model \
       --model_type transformer \
       --reload latest_checkpoint \
       --no_reload_training_progress \
       --embedding_size 512 \
       --state_size 512 \
       --tie_encoder_decoder_embeddings \
       --tie_decoder_embeddings \
       --label_smoothing 0.1 \
       --optimizer adam \
       --loss_function per-token-cross-entropy \
       --adam_beta1 0.9 \
       --adam_beta2 0.98 \
       --adam_epsilon 1e-09 \
       --learning_schedule constant \
       --learning_rate 0.0010 \
       --maxlen 100 \
       --batch_size 256 \
       --token_batch_size 16384 \
       --max_tokens_per_device 3276 \
       --valid_source_dataset dev-trunc.gu.ids.2en \
       --valid_target_dataset dev-trunc.en.ids \
       --valid_batch_size 120 \
       --valid_token_batch_size 3276 \
       --valid_freq 1000 \
       --valid_script ./validate.sh \
       --save_freq 1000 \
       --disp_freq 100 \
       --sample_freq 0 \
       --beam_freq 0 \
       --beam_size 4 \
       --translation_maxlen 100 \
       --normalization_alpha 0.6 \
       --exponential_smoothing 0.0001
   EOF
   ```

   The training script here we are using for fine-tuning is almost identical to the script that was used to train the original multilingual model.
   Apart from switching in the new training data and dev set, the main differences are:

    - The learning rate is constant (and actually set quite high), since by update 990,000 the original model was far into the decay phase of its warmup-decay schedule.

    - The argument ```--no_reload_training_progress ``` is used to reset some of the progress information (in particular, so that Nematus doesn't decide to stop early because the validation entropy is high compared to the last values it recorded).

   - The display, validation, and save frequencies are shorter than they were originally because we are expecting the model to change more rapidly.

   The original training script was based on the one [here](https://github.com/EdinburghNLP/wmt17-transformer-scripts/tree/master/training), which closely follows the settings used for the Transformer Base model of [Vaswani et al., 2017](https://papers.nips.cc/paper/7181-attention-is-all-you-need)

   For a description of all command-line arguments available in ```train.py``` see [here](https://github.com/EdinburghNLP/nematus).

1. Create the validation script:

   ```
   $ cat > validate.sh << 'EOF'
   #!/usr/bin/env bash

   ./postprocess.sh < $1 | ~/sacreBLEU/sacrebleu.py --score-only dev.en
   EOF
   ```

1. Create the postprocessing script:

   ```
   $ cat > postprocess.sh << 'EOF'
   #!/usr/bin/env bash

   ~/sentencepiece/build/src/spm_decode \
     --model ../multilingual-model/sentencepiece.model \
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

   Use ctrl-c to kill the tail command and get the shell prompt back.

   After a few minutes you should see a message indicating that the first 100 up
dates have been made:

   ```
   I0825 20:03:12.973962 140047237023488 train.py:192] [2019-08-25 20:03:12] Epoch: 0 Update: 100 Loss/word: 0.15517688742759467 Words/sec: 4160.4199645402505 Sents/sec: 215.6410988673752
   ```

   After 1000 updates, it will perform validation:

   ```
   ...
   I0825 20:30:14.656135 140047237023488 train.py:338] Validation cross entropy (AVG/SUM/N_SENTS/N_TOKENS): 81.96459748019447 163765.26576542854 1998 52448
   ...
   I0825 20:33:03.988270 140047237023488 train.py:376] Validation script score: 9.7
   ```

   If you are using the Google Cloud VM, then you can leave training to run unti
l you next need the GPU.
   You can kill the script by foregrounding the job and then using ctrl-c:
   ```
   $ fg
   <ctrl-c>
   ```
   Alternatively, you can find its process id using ```nvidia-smi``` and then use ```kill```.

### Exercises

- Download a pre-fine-tuned version of this model:

  ```
  wget 'http://data.statmt.org/mtm19-nematus-tutorial/fine-tuned-gu-en.tar.gz'
  ```

  Evaluate its performance on the WMT19 Gujarati-English dev and test sets.

  How does translation quality compare to the original multilingual model:
   1. In terms of BLEU
   1. In your own view...
    Take a look at the translations for both models.
    How fluent is the English?
    How well do they reflect the meaning of the input sentences (if you don't speak Gujarati, use the English reference for comparison)?

- Take a look at the [WMT19 system description papers](http://statmt.org/wmt19/papers.html).
  What did participants do to build strong Gujarati-English systems?

- Take a look at the [WMT19 system description page](http://statmt.org/wmt19/translation-task.html).
  If you were building a Gujarati-English model from scratch, what would you do?
