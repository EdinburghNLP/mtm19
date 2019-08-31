if [ -z "$MODEL" ] ||  [ -z "$FILE" ]; then echo "Specify if translating by parent or child in MODEL\nUsage: MODEL=child/parent FILE=file_to_trainslate ./decode.sh"; exit 0; fi



DATA_DIR=t2t_data
DATAGEN_DIR=t2t_datagen
CODEBASE=codebase
HPARAMS=transformer_big_single_gpu

if [ "$MODEL" == "parent" ]; then
    TRAIN_DIR=t2t_train/parent
	PROBLEM=parent_problem
else
    TRAIN_DIR=t2t_train/child
	PROBLEM=child_problem
fi

OUTPUT_FILE=output.translation

BEAM_SIZE=1
ALPHA=0.7

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=transformer \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$CODEBASE \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$FILE \
  --decode_to_file=$OUTPUT_FILE

echo "Translation is in file "$OUTPUT_FILE


