import json

import pdb

def main():
    infile='/home/john/LLaMA-Factory/data/train-jlw.json'
    outfile='train-jlw.jsonl'

    json_datas = []
    fo=open(outfile, 'w')
    with open(infile, 'r') as f:
        json_datas = json.loads(f.read())
        for result in json_datas:
            query = result['instruction']
            fo.write('{"conversations": [{"from": "human", "value": "' + result['instruction'] + '"},{"from":"gpt","value":"' + result['output'] + '"}]}\n')
    fo.close()

if __name__ == "__main__":
    main()
