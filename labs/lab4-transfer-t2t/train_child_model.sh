
PROBLEM=child_problem
MODEL=transformer
HPARAMS=transformer_big_single_gpu

DATA_DIR=t2t_data
TMP_DIR=t2t_datagen
TRAIN_DIR=t2t_train/child
CODEBASE=codebase

# check if TRAIN folder contains parent model
if [[ "$(ls $TRAIN_DIR/model* 2> /dev/null | wc -l)" == 0 ]]; then 
	echo "Provide parent model into the folder $TRAIN_DIR"; 
	exit 0;
fi


BATCH_SIZE=2400
# maximal length of sentences, longer will be dropped from training
MAX_LENGTH=100
SCHEDULE=rsqrt_decay
OPTIMIZER=Adafactor
WARMUP=16000


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
	  --train_steps=1750000 \
	  --hparams="batch_size=$BATCH_SIZE,max_length=$MAX_LENGTH,learning_rate_schedule=$SCHEDULE,optimizer=$OPTIMIZER,learning_rate_warmup_steps=$WARMUP"

