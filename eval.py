import os

filepath = '/tmp'
filenames = [filename for filename in os.listdir(filepath) if 'SQuAD' in filename]
with open('results.out', 'r') as f:
    evaled = [line.strip() for line in f.readlines()]
filenames = [filename for filename in filenames if filename not in evaled]
for filename in filenames:
    cmd = []
    cmd.append("export SQUAD_DIR=/home/meijie/data/squad")
    cmd.append('echo {}>>results.out'.format(filename))
    for epoch in range(5):
        try:
            if 'v1' in filename:
                cmd.append("python examples/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json"
                           " /tmp/{0}/predictions_{1}.json >>results.out"
                           .format(filename, epoch))
            elif 'v2' in filename:
                cmd.append("python examples/eval_squad_v2.0.py $SQUAD_DIR/dev-v2.0.json"
                           " /tmp/{0}/predictions_{1}.json >>results.out"
                           .format(filename, epoch))
        except Exception as e:
            print(e)
    cmd = ";".join(cmd)
    os.system(cmd)
