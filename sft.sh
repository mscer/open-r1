#accelerate launch --config_file recipes/accelerate_configs/ddp.yaml src/open_r1/sft.py \
#    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml

python src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
