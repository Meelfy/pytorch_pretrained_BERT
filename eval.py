import os
import itertools
# theta = [1, 3, 10]
theta = [1]
#alpha = [0.001, 0.03, 0.3]
alpha = [0.6]
beta  = [0]

for theta, alpha, beta in itertools.product(theta, alpha, beta):
    cmd   = []
    cmd.append("export SQUAD_DIR=/data/nfsdata/meijie/data/SQuAD/")
    cmd.append('echo "SQuAD_v1-{0}_{1}_{2}_newloss_saveLoss/">>eval.out'.format(theta, alpha, beta))
    for epoch in range(3):
        cmd.append("python examples/evaluate-v1.1.py /data/nfsdata/meijie/data/SQuAD/dev-v1.1.json \
                /tmp/SQuAD_v1-{0}_{1}_{2}_newloss_saveLoss/predictions_{3}.json >>eval.out".format(theta, alpha, beta, epoch))
    #cmd.append('echo "\n">>eval.out')
    cmd = ";".join(cmd)
    os.system(cmd)
