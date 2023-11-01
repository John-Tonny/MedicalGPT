
import json

filename = './data/finetune/train-jlw.jsonl'

outfile = 'eval.txt'
fo = open(outfile, 'w')

with open(filename, 'r') as f:
    line = f.readline()
    while line:
        result = json.loads(line.strip())
        query = result['conversations'][0]['value']
        fo.write(query + '\n')
        line = f.readline()

fo.close()



