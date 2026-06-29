#!/bin/bash

# CDDB
python main.py ./exps/clip/cddb,1-shot.json -gpu_id 0
python main.py ./exps/clip/cddb,2-shot.json -gpu_id 0
python main.py ./exps/clip/cddb,4-shot.json -gpu_id 0
python main.py ./exps/clip/cddb,8-shot.json -gpu_id 0

# CORe50
python main.py exps/clip/core50,1-shot.json -gpu_id 0
python main.py exps/clip/core50,2-shot.json -gpu_id 0
python main.py exps/clip/core50,4-shot.json -gpu_id 0
python main.py exps/clip/core50,8-shot.json -gpu_id 0
python main.py exps/clip/core50,5-shot.json -gpu_id 0


# DomainNet
python main.py ./exps/clip/domain_net,1-shot.json -gpu_id 0
python main.py ./exps/clip/domain_net,2-shot.json -gpu_id 0
python main.py ./exps/clip/domain_net,4-shot.json -gpu_id 0
python main.py ./exps/clip/domain_net,8-shot.json -gpu_id 0
python main.py ./exps/clip/domain_net,5-shot.json -gpu_id 0