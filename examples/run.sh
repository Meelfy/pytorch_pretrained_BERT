export SQUAD_DIR=/home/meelfy/working/tdt/01_data/squad
nohup python run_squad_v2.0.py \
  --bert_model /home/meelfy/working/tdt/03_bert/pytorch_pretrained_BERT/uncased_L-12_H-768_A-12/ \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ &


python eval_squad_v2.0.py $SQUAD_DIR/dev-v2.0.json /tmp/debug_squad/predictions.json
