# !/bin/bash

python fastapi_server_demo.py --template_name chatglm2  --model_type chatglm --base_model /opt/chatglm2-6b --lora_model outputs-sft-bloom-v2 --port 7860
