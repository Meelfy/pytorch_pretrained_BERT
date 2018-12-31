import os
import itertools
# theta = [1, 3, 10]
theta = [1]
alpha = [0.5, 1., 2.]
beta  = [0]
for theta, alpha, beta in itertools.product(theta, alpha, beta):
    cmd   = []
    cmd.append("export CUDA_VISIBLE_DEVICES=2,3")
    cmd.append("export SQUAD_DIR=/data/nfsdata/meijie/data/SQuAD/")
    cmd.append("export PYTHONPATH=/home/meefly/working/pytorch_pretrained_BERT/:$PYTHONPATH")
    cmd.append("export SAVE_DIR=/tmp/SQuAD_v1-{0}_{1}_{2}_newloss/".format(theta, alpha, beta))
    cmd.append("python examples/run_squad.py \
              --bert_model /data/nfsdata/meijie/data/uncased_L-12_H-768_A-12 \
              --do_train \
              --do_predict \
              --do_lower_case \
              --train_file $SQUAD_DIR/train-v1.1.json \
              --predict_file $SQUAD_DIR/dev-v1.1.json \
              --train_batch_size 12 \
              --learning_rate 3e-5 \
              --num_train_epochs 5.0 \
              --max_seq_length 384 \
              --doc_stride 128 \
              --seed 1\
              --theta {0}\
              --alpha {1}\
              --beta {2}\
              --output_dir $SAVE_DIR > {0}_{1}_{2}_newloss.out 2>&1".format(theta, alpha, beta))
    cmd = ";".join(cmd)
    os.system(cmd)
