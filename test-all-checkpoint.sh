# !/bin/bash

python test-all.py --template_name chatglm2 --model_type chatglm --base_model /opt/chatglm2-6b --lora_model outputs-sft-bloom-v1 --interactive --load_in_4bit true
