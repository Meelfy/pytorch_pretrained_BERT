import os
import itertools
theta = [1, 3, 10]
alpha = [0.5, 1., 2.]
beta  = [1, 2, 4]
for theta, alpha, beta in itertools.product(theta, alpha, beta):
    cmd   = []
    cmd.append("export SQUAD_DIR=/data/nfsdata/meijie/data/SQuAD/")
    cmd.append('echo "SQuAD_v1-{0}_{1}_{2}_newloss/">>eval.out'.format(theta, alpha, beta))
    cmd.append("export SAVE_DIR=/tmp/SQuAD_v1-{0}_{1}_{2}_newloss/".format(theta, alpha, beta))
    cmd.append("python examples/evaluate-v1.1.py /data/nfsdata/meijie/data/SQuAD/dev-v1.1.json \
                /tmp/SQuAD_v1-{0}_{1}_{2}_newloss/predictions.json >>eval.out".format(theta, alpha, beta))
    cmd.append('echo "\n">>eval.out')
    cmd = ";".join(cmd)
    os.system(cmd)
