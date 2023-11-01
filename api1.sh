# !/bin/bash

python api_demo.py --template_name chatglm2  --model_type chatglm --base_model /opt/chatglm2-6b --lora_model outputs-sft-bloom-v2-4bit --port 8899 --load_in_4bit True 
