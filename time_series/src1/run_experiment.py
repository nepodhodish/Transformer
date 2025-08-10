# torchrun --nproc_per_node 6 run_experiment.py > nohup.out 2>&1 &
# disown

import time
start = time.time()

import os
import sys
import math
import glob
import shutil
import argparse
import datetime

import numpy as np
import pandas as pd
import pickle as pk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import warnings
warnings.filterwarnings('ignore')

import parts


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MAIN_DIR = os.path.dirname(PROJECT_DIR)


def main():


    # parameters
    parser = argparse.ArgumentParser(description='Run GPT time series prediction experiments')
    parser.add_argument('--data_train', type=str, default=os.path.join(MAIN_DIR, 'data', 'train', 'data_done_train'), 
                        help='Path to train data')
    parser.add_argument('--data_test', type=str, default=os.path.join(MAIN_DIR, 'data', 'test', 'data_done_test'), 
                        help='Path to test data')
    parser.add_argument('--voc_size', type=list, default=[len(parts.make_voc(voc)) for voc in parts.VOCABULARY_INTERVALS], 
                        help='Size of vocabulary for each variable')
    parser.add_argument('--sequence', type=int, default=100, 
                        help='Maximum context length for GPT model')
    parser.add_argument('--batch', type=int, default=100, 
                        help='Batch size')
    parser.add_argument('--emb_dim', type=int, default=128, 
                        help="Embeddings' dimensions")
    parser.add_argument('--ff_hidden_layer', type=int, default=128*8*4, 
                        help='Dimensions of feed forward hidden layer')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropour rate')
    parser.add_argument('--num_heads', type=int, default=16, 
                        help='Number of heads in attention')
    parser.add_argument('--num_layers', type=int, default=4, 
                        help='Number of layers in GPT')
    parser.add_argument('--lr', type=float, default=5e-4, 
                        help='Learning rate')
    parser.add_argument('--warmup', type=float, default=0.05, 
                        help='Lr warmup')
    parser.add_argument('--wd', type=float, default=1e-2, 
                        help='Weight decay')
    parser.add_argument('--corr_weight_decay', type=bool, default=True, 
                        help='Do not apply wd to bias and norm layers')
    parser.add_argument('--norm_clip', type=float, default=1.0, 
                        help='Max norm clip')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='Number of training epochs')
    parser.add_argument('--max_train_batches_per_epoch', type=float, default=float('inf'), 
                        help='Max number train batches per epoch')
    parser.add_argument('--max_test_batches_per_epoch', type=float, default=float('inf'), 
                        help='Max number test batches per epoch')
    args = parser.parse_args()

    
    # create experiment folder
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = os.path.join(PROJECT_DIR, 'experiments', f'{timestamp}_{os.path.basename(SCRIPT_DIR)}')
    os.makedirs(exp_dir, exist_ok=True)


    # count total train batches per epoch
    train_batches_per_epoch = parts.count_batches(args.data_train, args.sequence, args.batch)
    parser.add_argument('--train_batches_per_epoch', type=int, default=int(min(train_batches_per_epoch, args.max_train_batches_per_epoch)), 
                        help='Number train batches per epoch')
    

    # count total test batches per epoch
    test_batches_per_epoch = parts.count_batches(args.data_test, args.sequence, args.batch)
    parser.add_argument('--test_batches_per_epoch', type=int, default=int(min(test_batches_per_epoch, args.max_test_batches_per_epoch)), 
                        help='Number test batches per epoch')
    

    # establish main model
    model = parts.GPT(args.voc_size,
                args.sequence, 
                args.emb_dim, 
                args.ff_hidden_layer,
                args.dropout, 
                args.num_heads,
                args.num_layers)
    parts.init_weights(model, args)

    
    # count number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parser.add_argument('--total_parameters', type=int, default=total_params, 
                        help='Total num. parameters')


    # save settings
    args = parser.parse_args()
    settings_path = os.path.join(exp_dir, 'experiment_settings.txt')
    with open(settings_path, 'w') as file:
        file.write(f'Command: {" ".join(sys.argv)} \n')
        file.write(f'Arguments:\n')
        for k,v in vars(args).items():
            file.write(f'  {k}: {v}\n')


    # backup scripts
    scripts_dir = os.path.join(exp_dir, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    for file in glob.glob(os.path.join(SCRIPT_DIR, '*.py')):
        shutil.copy(file, scripts_dir)

    
    
    # start parallelization
    parts.fsdp_main(model, args, exp_dir)


    
    


if __name__ == '__main__':
    main()
