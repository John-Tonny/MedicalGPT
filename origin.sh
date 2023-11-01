# !/bin/bash

python inference.py --model_type chatglm --base_model /opt/chatglm2-6b  --interactive --temperature 1.0 --load_in_4bit none
