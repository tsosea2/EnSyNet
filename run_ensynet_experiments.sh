#!/bin/bash


CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 1 --model_type distilbert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 2 --model_type distilbert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 3 --model_type distilbert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 4 --model_type distilbert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 5 --model_type distilbert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json


CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 1 --model_type bert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 2 --model_type bert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 3 --model_type bert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 4 --model_type bert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 5 --model_type bert-base-uncased --csv_path data/encouragement.csv --results_file encouragement.json

CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 1 --model_type distilbert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 2 --model_type distilbert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 3 --model_type distilbert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 4 --model_type distilbert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 5 --model_type distilbert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json

CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 1 --model_type bert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 2 --model_type bert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 3 --model_type bert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 4 --model_type bert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_length 128 --max_epochs 10 --seed 5 --model_type bert-base-uncased --csv_path data/sympathy.csv --results_file sympathy.json