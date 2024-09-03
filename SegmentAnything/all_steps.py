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


# ARGS ========================================================================
parser = argparse.ArgumentParser(description='Script with input arguments')
parser.add_argument('-e', '--epochs', type=int, default=1, help='epochs')
parser.add_argument('-bs', '--batch_size', type=int, default=2, help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('-ds', '--dataset', type=str, default='dataset/TNBC', help='dataset path')
parser.add_argument('-wp', '--weights_path', type=str, default='weights', help='Path to the directory where model weights are stored')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b', help='Size of Sam')
parser.add_argument('-f', '--freeze_list', nargs='+', type=str, default=['image_encoder', 'prompt_encoder'], help='List of names to freeze')
parser.add_argument('-p', '--pretrained_model', type=str, default=None, help='Address of the pretrained model')
parser.add_argument('-ad', '--additional_description', type=str, default='', help='Additional description')

args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
dataset_path = args.dataset
weights_save_path = args.weights_path
model_type = args.model_type
pretrained_weight_path = args.pretrained_model
freeze_list = args.freeze_list
epochs = args.epochs

add_str = 'lr{}_bs{}_pw{}_f{}_add{}'.format(learning_rate, batch_size, pretrained_weight_path!=None, len(freeze_list), args.additional_description)
evaluation_save_dir = 'logs/evaluation_logs/f{}'.format(len(freeze_list))
os.makedirs(evaluation_save_dir, exist_ok=True)


# MODEL =======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](pretrained_weight_path)
sam.train()
for param in sam.parameters():
    param.requires_grad = True

for name, param in sam.named_parameters():
    for _freeze in freeze_list:
        if _freeze in name:
            param.requires_grad = False
            continue

# DATA =======================================================================
dataset_splitter = DatasetSplitter(dataset_path)
dataset_splitter.split_dataset()
train_dataset, val_dataset, test_dataset = create_datasets(dataset_info=dataset_splitter.dataset_info)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

loss_func = BinaryMaskLoss()
optimizer = optim.Adam(sam.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)


# TRAINER ====================================================================
class SamTrainer(Trainer):
    def __init__(self, base_log_path: str | None = 'train_log') -> None:
        super().__init__(base_log_path)
    
    def cal_batch_loss(self, batch, step: Optional[int] = None):
        res = sam(batch[0].to(self.device), multimask_output=False)
        loss = self.loss_func(torch.sigmoid(res), torch.sigmoid(batch[1].to(self.device)))
        return loss


strain = SamTrainer(base_log_path='logs/train_logs')

strain.set_params(model=sam, 
                  device=device, 
                  epochs=epochs, 
                  batch_size=batch_size, 
                  save_path=weights_save_path, 
                  loss_func=loss_func, 
                  optimizer=optimizer, 
                  train_loader=train_loader, 
                  val_loader=val_loader, 
                  lr_scheduler=exp_lr_scheduler, 
                  add_description=add_str, 
                  train_info_save_path=evaluation_save_dir, 
                  )
strain.start_training()
strain.save_parameters()


# EVALUATION ====================================================================
def evaluate_segmentation_model(test_loader, segmentation_model, threshold, save_dir, device):
    test_images = []
    test_labels = []
    test_predictions = []
    for idx, batch in enumerate(test_loader):
        inputs = batch[0].to(device)
        outputs = segmentation_model(inputs, False)
        if device.type == 'cuda':
            outputs = outputs.cpu()
        for i in range(len(outputs)):
            probabilities = torch.sigmoid(outputs)[i].squeeze(0).detach().numpy()
            predictions = (probabilities > threshold).astype(np.uint8)
            test_predictions.append(predictions)
        temp_imgs, temp_labels = split_batch_images_labels(batch)
        test_images.extend(temp_imgs)
        test_labels.extend(temp_labels)

    eval_metrics_calculator = MetricsCalculator()
    evaluation_metrics = eval_metrics_calculator.calculate_metrics(test_labels, test_predictions)
    with open(os.path.join(save_dir, 'metrics_info.json'), 'w') as f:
        json.dump(evaluation_metrics, f)


sam.eval()
with torch.no_grad():
    evaluation_batch_size = batch_size
    threshold = get_threshold(val_loader, sam, device, evaluation_save_dir)
    evaluate_segmentation_model(test_loader, sam, threshold, evaluation_save_dir, device)