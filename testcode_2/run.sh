deepspeed --num_nodes 1 \
    --num_gpus 1 \
    run.py \
    --deepspeed "./config/default_offload_zero2.json" \
    --model_name_or_path model_xxx \
    --output_dir hh