from torch.optim import lr_scheduler
from torch import optim
import torch
from typing import Optional
import argparse
import os
import numpy as np
import json

from segment_anything import sam_model_registry
from scripts import BinaryMaskLoss, Trainer, DatasetSplitter, create_datasets, get_threshold, split_batch_images_labels, MetricsCalculator

import deepspeed
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import json

with open("temp.json",'r', encoding='UTF-8') as f:
     cmd_args = json.load(f)

# specify the model you want to train on your device
model = sam_model_registry['vit_h']('pretrained_weights/sam_vit_h_4b8939.pth')
# estimate the memory cost (both CPU and GPU)
model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=model.parameters)

dataset_path = 'dataset/TNBC'
dataset_splitter = DatasetSplitter(dataset_path)
dataset_splitter.split_dataset()
train_dataset, val_dataset, test_dataset = create_datasets(dataset_info=dataset_splitter.dataset_info)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
