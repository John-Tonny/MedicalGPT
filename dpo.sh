# !/bin/bash

python inference.py --template_name chatglm2 --model_type chatglm --base_model exportmodel-dpo-4bit --interactive --load_in_4bit true --history 0
