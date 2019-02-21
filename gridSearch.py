import os
import itertools
import time
theta = [1, 3, 10]
# theta = [1]
# alpha = [0.001, 0.03, 0.3]
alpha = [0, 0.6, 0.7, 1]
beta = [0, 1, 2, 4]
small_or_large = 'large'
for theta, alpha, beta in itertools.product(theta, alpha, beta):
    cmd = []
    cmd.append("export CUDA_VISIBLE_DEVICES=0,1,2")
    cmd.append("export SQUAD_DIR=/data/nfsdata/meijie/data/SQuAD")
    cmd.append("export PYTHONPATH=/home/meefly/working/pytorch_pretrained_BERT/:$PYTHONPATH")
    if small_or_large == 'small':
        cmd.append("export SAVE_DIR=/tmp/SQuAD_v1-{0}_{1}_{2}_newloss_saveLoss/".format(theta, alpha, beta))
        cmd.append("python examples/run_squad.py \
                    --bert_model /data/nfsdata/meijie/data/uncased_L-12_H-768_A-12 \
                    --do_train \
                    --do_predict \
                    --do_lower_case \
                    --train_file $SQUAD_DIR/train-v1.1.json \
                    --predict_file $SQUAD_DIR/dev-v1.1.json \
                    --train_batch_size 1 \
                    --learning_rate 3e-5 \
                    --num_train_epochs 3.0 \
                    --max_seq_length 384 \
                    --doc_stride 128 \
                    --seed 1\
                    --theta {0}\
                    --alpha {1}\
                    --beta {2}\
                    --output_dir $SAVE_DIR > ./out/{0}_{1}_{2}_newloss_saveLoss.out 2>&1"
                   .format(theta, alpha, beta))
    elif small_or_large == 'large':
        cmd.append("export SAVE_DIR=/tmp/SQuAD_v2-{0}_{1}_{2}_newloss_large_2/".format(theta, alpha, beta))
        cmd.append("python examples/run_squad.py \
                    --bert_model /data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-24_H-1024_A-16 \
                    --do_train \
                    --version_2_with_negative\
                    --do_predict \
                    --do_lower_case \
                    --train_file $SQUAD_DIR/train-v2.0.json \
                    --predict_file $SQUAD_DIR/dev-v2.0.json \
                    --learning_rate 3e-5 \
                    --num_train_epochs 2 \
                    --max_seq_length 384 \
                    --doc_stride 128 \
                    --output_dir $SAVE_DIR \
                    --train_batch_size 24 \
                    --fp16 \
                    --theta {0}\
                    --alpha {1}\
                    --beta {2}\
                    --gradient_accumulation_steps 2\
                    --loss_scale 128 > ./out/{0}_{1}_{2}_newloss_large_2.out 2>&1".format(theta, alpha, beta))
    cmd = ";".join(cmd)
    for i in range(4):
        return_code = os.system(cmd)
        if return_code == 0:
            break
        else:
            print('sleep for {} secs'.format(10 ** i))
            time.sleep(10 ** i)
