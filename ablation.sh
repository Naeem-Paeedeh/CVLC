#!/bin/bash

python main.py exps/clip/ablation/LoRA/core50,1-shot,LoRA.json -gpu_id 0
python main.py exps/clip/ablation/LoRA/core50,2-shot,LoRA.json -gpu_id 0
python main.py exps/clip/ablation/LoRA/core50,4-shot,LoRA.json -gpu_id 0
python main.py exps/clip/ablation/LoRA/core50,8-shot,LoRA.json -gpu_id 0

python main.py exps/clip/ablation/Prompt/core50,1-shot,prompt.json -gpu_id 0
python main.py exps/clip/ablation/Prompt/core50,2-shot,prompt.json -gpu_id 0
python main.py exps/clip/ablation/Prompt/core50,4-shot,prompt.json -gpu_id 0
python main.py exps/clip/ablation/Prompt/core50,8-shot,prompt.json -gpu_id 0

python main.py exps/clip/ablation/without_LSR/core50,1-shot,without_LSR.json -gpu_id 0
python main.py exps/clip/ablation/without_LSR/core50,2-shot,without_LSR.json -gpu_id 0
python main.py exps/clip/ablation/without_LSR/core50,4-shot,without_LSR.json -gpu_id 0
python main.py exps/clip/ablation/without_LSR/core50,8-shot,without_LSR.json -gpu_id 0

python main.py exps/clip/ablation/single_template/core50,1-shot,single_template.json -gpu_id 0
python main.py exps/clip/ablation/single_template/core50,2-shot,single_template.json -gpu_id 0
python main.py exps/clip/ablation/single_template/core50,4-shot,single_template.json -gpu_id 0
python main.py exps/clip/ablation/single_template/core50,8-shot,single_template.json -gpu_id 0

python main.py exps/clip/ablation/only_task_specific-DCP/core50,1-shot,only_task_specific-DCPs.json -gpu_id 0
python main.py exps/clip/ablation/only_task_specific-DCP/core50,8-shot,only_task_specific-DCPs.json -gpu_id 0
python main.py exps/clip/ablation/only_task_specific-DCP/core50,4-shot,only_task_specific-DCPs.json -gpu_id 0
python main.py exps/clip/ablation/only_task_specific-DCP/core50,2-shot,only_task_specific-DCPs.json -gpu_id 0
