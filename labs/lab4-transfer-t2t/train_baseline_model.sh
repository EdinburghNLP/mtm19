PROBLEM=child_problem
MODEL=transformer
HPARAMS=transformer_big_single_gpu

DATA_DIR=t2t_data
TMP_DIR=t2t_datagen
#if the TRAIN folder is empty, T2T will train from scratch
TRAIN_DIR=t2t_train/baseline
CODEBASE=codebase

mkdir -p $TRAIN_DIR


# Train child model
t2t-trainer \
	  --data_dir=$DATA_DIR \
	  --t2t_usr_dir=$CODEBASE \
      --problem=$PROBLEM \
      --model=$MODEL \
      --hparams_set=$HPARAMS \
	  --output_dir=$TRAIN_DIR \
	  --keep_checkpoint_max=100 \
	  --local_eval_frequency=1000 \
	  --train_steps=200000 \
	  --hparams='batch_size=2400,max_length=100,learning_rate_schedule=rsqrt_decay,optimizer=Adafactor,learning_rate_warmup_steps=16000'

