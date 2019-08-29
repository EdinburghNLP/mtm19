PROBLEM=parent_problem
MODEL=transformer
HPARAMS=transformer_big_single_gpu

DATA_DIR=t2t_data
TMP_DIR=t2t_datagen
TRAIN_DIR=t2t_train/parent/
CODEBASE=codebase

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
	--t2t_usr_dir=$CODEBASE \
    --problem=$PROBLEM

# Train
t2t-trainer \
	  --data_dir=$DATA_DIR \
      --problem=$PROBLEM \
      --model=$MODEL \
	  --t2t_usr_dir=$CODEBASE \
      --hparams_set=$HPARAMS \
	  --output_dir=$TRAIN_DIR \
	  --worker_gpu=1 \
	  --keep_checkpoint_max=100 \
	  --hparams='batch_size=2400,max_length=100,learning_rate_schedule=rsqrt_decay,optimizer=Adafactor,learning_rate_warmup_steps=16000'

# worker_gpu specifies number of GPUs
# keep_checkpoint_max will store latest 100 checkpints
# max_length drops training sentences longer than X (this is useful for increasing batch size)
