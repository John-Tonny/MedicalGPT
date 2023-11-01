# !/bin/bash

python inference.py --template_name chatglm2 --model_type chatglm --base_model exportmodel-sft-4bit --lora_model outputs-dpo-bloom-v2 --interactive --load_in_4bit true --history 0
