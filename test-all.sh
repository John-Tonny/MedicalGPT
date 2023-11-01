# !/bin/bash

python test-all.py --template_name chatglm2  --model_type chatglm --base_model  ./exportmodel-sft-4bit --interactive --load_in_4bit true --history 1
