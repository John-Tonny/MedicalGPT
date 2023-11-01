# !/bin/bash

python inference.py --template_name chatglm2 --model_type chatglm --base_model /opt/chatglm2-6b --lora_model outputs-sft-bloom-v2-4bit --load_in_4bit true --data_file "eval.txt"  --eval_batch_size 10 --history 1
