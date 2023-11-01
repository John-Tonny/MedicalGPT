# !/bin/bash

python inference.py --template_name chatglm2 --model_type chatglm --base_model /opt/chatglm2-6b --lora_model outputs-sft-bloom-v2-4bit --interactive --load_in_4bit true --history 0 --temperature 1.0 --repetition_penalty 1.0
