# KDCN

## Requirements
- Python 3.6
- torch 1.4.0
- transformers 4.9.1
- nltk


## Run
python train.py --dataset pheme --model add_token --cuda 0,1 --epoch 50 --lr 0.00005 --lambda_orth 1.5 --batch 64

