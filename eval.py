import os

filepath = '/tmp'
filenames = [filename for filename in os.listdir(filepath) if 'SQuAD' in filename]
with open('results.out', 'r') as f:
    evaled = [line.strip() for line in f.readlines()]
filenames = [filename for filename in filenames if filename not in evaled]
for filename in filenames:
    cmd = []
    cmd.append("export SQUAD_DIR=/data/nfsdata/meijie/data/SQuAD/")
    cmd.append('echo {}>>results.out'.format(filename))
    for epoch in range(5):
        try:
            cmd.append("python examples/evaluate-v1.1.py /data/nfsdata/meijie/data/SQuAD/dev-v1.1.json"
                       " /tmp/{0}/predictions_{1}.json >>results.out"
                       .format(filename, epoch))
        except Exception as e:
            pass
    cmd = ";".join(cmd)
    os.system(cmd)
