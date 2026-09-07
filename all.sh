#!/bin/bash

DEVICE=0

## Main experiments

## CDDB
python main.py exps/clip/main_experiments/cddb,1-shot.json -gpu_id $DEVICE
python main.py exps/clip/main_experiments/cddb,2-shot.json -gpu_id $DEVICE
python main.py exps/clip/main_experiments/cddb,4-shot.json -gpu_id $DEVICE
python main.py exps/clip/main_experiments/cddb,8-shot.json -gpu_id $DEVICE

## CORe50
python main.py exps/clip/main_experiments/core50,1-shot.json -gpu_id $DEVICE
python main.py exps/clip/main_experiments/core50,2-shot.json -gpu_id $DEVICE
python main.py exps/clip/main_experiments/core50,4-shot.json -gpu_id $DEVICE
python main.py exps/clip/main_experiments/core50,8-shot.json -gpu_id $DEVICE

## DomainNet
python main.py exps/clip/main_experiments/domain_net,1-shot.json -gpu_id $DEVICE
python main.py exps/clip/main_experiments/domain_net,2-shot.json -gpu_id $DEVICE
python main.py exps/clip/main_experiments/domain_net,4-shot.json -gpu_id $DEVICE
python main.py exps/clip/main_experiments/domain_net,8-shot.json -gpu_id $DEVICE

# TODO: Uncomment the following lines to run ablation studies.
# ## --------------------------------------------------------------------------------------
# ## Ablation studies

# python main.py exps/clip/ablation/LoRA/core50,1-shot,LoRA.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/LoRA/core50,2-shot,LoRA.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/LoRA/core50,4-shot,LoRA.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/LoRA/core50,8-shot,LoRA.json -gpu_id $DEVICE

# ## Prompt with almost the same PEFT budget as DCP (11 vision / 16 text tokens).
# python main.py exps/clip/ablation/Prompt_matched/core50,1-shot,prompt.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/Prompt_matched/core50,2-shot,prompt.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/Prompt_matched/core50,4-shot,prompt.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/Prompt_matched/core50,8-shot,prompt.json -gpu_id $DEVICE

# ## Real class names without synonyms
# python main.py exps/clip/ablation/real_class_names/core50,1-shot,real_class_names.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/real_class_names/core50,2-shot,real_class_names.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/real_class_names/core50,4-shot,real_class_names.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/real_class_names/core50,8-shot,real_class_names.json -gpu_id $DEVICE

# python main.py exps/clip/ablation/only_task_specific-DCP/core50,1-shot,only_task_specific-DCPs.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/only_task_specific-DCP/core50,8-shot,only_task_specific-DCPs.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/only_task_specific-DCP/core50,4-shot,only_task_specific-DCPs.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/only_task_specific-DCP/core50,2-shot,only_task_specific-DCPs.json -gpu_id $DEVICE

# ## Prototype correction and shift correction 
# ## ("use_prototype_correction": false, zero shift coefficients, frozen embedding biases)
# python main.py exps/clip/ablation/without_prototype_and_shift/core50,1-shot.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/without_prototype_and_shift/core50,2-shot.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/without_prototype_and_shift/core50,4-shot.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/without_prototype_and_shift/core50,8-shot.json -gpu_id $DEVICE

# ## Without prototype interpolations ("use_interpolations": false)
# python main.py exps/clip/ablation/without_interpolations/core50,1-shot.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/without_interpolations/core50,2-shot.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/without_interpolations/core50,4-shot.json -gpu_id $DEVICE
# python main.py exps/clip/ablation/without_interpolations/core50,8-shot.json -gpu_id $DEVICE

