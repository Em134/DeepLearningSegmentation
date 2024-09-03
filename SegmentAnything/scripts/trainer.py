import torch
from typing import Optional, Tuple
import os
import logging
import time
import numpy as np
import torch
import json


class Trainer(object):
    def __init__(self, 
                 base_log_path: Optional[str] = 'train_log', 
                 ) -> None:
        self.training = True
        self.base_log_path = base_log_path
        self.save_times = 0
        self.best_valid_loss = np.inf
        self.best_train_loss = np.inf

    def get_logger(self, message_level: str) -> logging.Logger:
        logger = logging.getLogger(message_level)
        logger.setLevel(logging.INFO)  # Log等级总开关
        formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                    datefmt="%a %b %d %H:%M:%S %Y")
        # StreamHandler
        sHandler = logging.StreamHandler()
        sHandler.setFormatter(formatter)
        logger.addHandler(sHandler)

        # FileHandler
        work_dir = os.path.join(self.base_log_path, 
                                self.mid_fix, self.add_description, 
                                message_level,
                                time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
        fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        fHandler.setFormatter(formatter)  # 定义handler的输出格式
        logger.addHandler(fHandler)  # 将logger添加到handler里面

        return logger

    def set_params(self, 
                   model: torch.nn.Module,
                   device: torch.device,
                   epochs: int, 
                   batch_size: int, 
                   save_path: str, 
                   loss_func: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   train_loader: torch.utils.data.DataLoader, 
                   val_loader: Optional[torch.utils.data.DataLoader] = None, 
                   model_name: Optional[str] = None, 
                   lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, 
                   add_description: Optional[str] = '', 
                   train_info_save_path: Optional[str] = 'logs/evaluation_logs', 
                   only_best_weights: Optional[bool] = True, 
                   ) -> None:
        self.model = model
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_path = save_path
        self.model_name = model_name
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.add_description = add_description
        self.train_info_save_path = train_info_save_path
        self.only_best_weights = only_best_weights
        
        self.cur_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        
        if model_name is None: 
            self.model_name = type(model).__name__
        self.base_log_path = os.path.join(self.base_log_path, self.model_name)

        self.mid_fix = os.path.join(self.model_name, 
                                    type(self.loss_func).__name__, 
                                    )
        
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model,)

        self.batch_logger = self.get_logger('batch')
        self.epoch_logger = self.get_logger('epoch')

        self.batch_logger.info('[USING {} GPUs. ]'.format(torch.cuda.device_count()))
        self.epoch_logger.info('[USING {} GPUs. ]'.format(torch.cuda.device_count()))
        self.model.to(self.device)

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        self.start_training(epochs=kwds['epochs'])
        self.save_parameters()

    def start_training(self, epochs: Optional[int] = None):
        if epochs != None:
            self.epochs = epochs
        for epoch in range(self.epochs):
            self.run_a_epoch(epoch=epoch)
    
    def run_a_epoch(self, epoch) -> Tuple[float, float]:
        epoch_train_loss_list = []
        epoch_valid_loss_list = []
        
        # TRAIN
        self.training = True
        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.run_a_batch(batch=batch, epoch=epoch, batch_idx=batch_idx)
            epoch_train_loss_list.append(loss.item())
        epoch_average_train_loss = sum(epoch_train_loss_list) / len(epoch_train_loss_list)

        # EVAL
        if self.val_loader != None:
            self.training = False
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
                    loss = self.run_a_batch(batch=batch, epoch=epoch, batch_idx=batch_idx)
                    epoch_valid_loss_list.append(loss.item())
                epoch_average_valid_loss = sum(epoch_valid_loss_list) / len(epoch_valid_loss_list)
        else:
            epoch_average_valid_loss = np.inf
        
        if self.lr_scheduler != None:
            self.cur_lr = self.lr_scheduler.get_last_lr()
            self.lr_scheduler.step()
        else:
            self.cur_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        self.epoch_logger.info('[Epoch: {}/{}] [Epoch Average Train Loss: {:.4f}] [Epoch Average Val Loss: {:.4f}] [Learning Rate: {}]'.format(
            epoch + 1, 
            self.epochs, 
            epoch_average_train_loss, 
            epoch_average_valid_loss, 
            self.cur_lr, 
            ))
        
        self.save_model(epoch_average_train_loss, epoch_average_valid_loss)
        return epoch_average_train_loss, epoch_average_valid_loss

    def run_a_batch(self, batch, epoch, batch_idx) -> torch.Tensor:
        self.optimizer.zero_grad()
        loss = self.cal_batch_loss(batch=batch)

        if self.training:
            phase = 'Train'
            batch_len = len(self.train_loader)
        else:
            phase = 'Valid'
            batch_len = len(self.val_loader)
        self.batch_logger.info('  - [{}] [Epoch: {}/{}] [{} batch: {}/{}] [{} Loss: {:.4f}] [Learning Rate: {}]'.format(
                self.mid_fix + '_' + self.add_description, 
                epoch + 1, 
                self.epochs,
                phase, 
                batch_idx + 1,
                batch_len, 
                phase, 
                loss.item(), 
                self.cur_lr, 
                ))
        
        if self.training:
            loss.backward()  # 反向传播求一次偏导
            self.optimizer.step()  # 算完了偏导得step才能更新进去
            
        torch.cuda.empty_cache()

        return loss

    def cal_batch_loss(self, batch, step) -> torch.Tensor:
        pred_res = self.model(batch[0])
        loss = self.loss_func(pred_res, batch[1])
        return loss

    def save_model(self, epoch_average_train_loss, epoch_average_valid_loss) -> None: 
        os.makedirs(os.path.join(self.save_path, self.mid_fix, self.add_description), exist_ok=True)
        if self.only_best_weights:
            weights_save_path = os.path.join(self.save_path, 
                                    self.mid_fix, self.add_description, 
                                    'best_weights.pth')
        else:
            weights_save_path = os.path.join(self.save_path, 
                                    self.mid_fix, self.add_description, 
                                    '{}_{}.pth'.format(time.strftime("%Y-%m-%d-%H.%M"), self.save_times))
        self.final_save_path = weights_save_path
        
        if self.val_loader is None:
            if epoch_average_train_loss < self.best_train_loss:
                self.best_train_loss = epoch_average_train_loss
                self.epoch_logger.info('[Saving Model] [train_best_loss: {}] [valid_best_loss: {}]'.format(self.best_train_loss, self.best_valid_loss))
                if isinstance(self.model, torch.nn.DataParallel):
                    torch.save(self.model.module.state_dict(), weights_save_path)
                else:
                    torch.save(self.model.state_dict(), weights_save_path)
                self.save_times += 1
                
        elif epoch_average_valid_loss < self.best_valid_loss:
            if epoch_average_train_loss < self.best_train_loss:
                self.best_train_loss = epoch_average_train_loss
            self.best_valid_loss = epoch_average_valid_loss
            self.epoch_logger.info('[Saving Model] [train_best_loss: {}] [valid_best_loss: {}]'.format(self.best_train_loss, self.best_valid_loss))
            if isinstance(self.model, torch.nn.DataParallel):
                torch.save(self.model.module.state_dict(), weights_save_path)
            else:
                torch.save(self.model.state_dict(), weights_save_path)
            self.save_times += 1

    def save_parameters(self) -> None:
        """
        保存Trainer类中的所有参数到一个JSON文件，并将该文件保存到指定的目录下。

        参数：
            - save_dir (str)：保存参数JSON文件的目录路径。
        """
        parameters = {
            'base_log_path': self.base_log_path,
            'save_times': self.save_times,
            'best_valid_loss': self.best_valid_loss,
            'best_train_loss': self.best_train_loss,
            'device': str(self.device),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'save_path': self.save_path,
            'model_name': self.model_name,
            'add_description': self.add_description, 
            'final_save_path': self.final_save_path, 
        }

        if not os.path.exists(self.train_info_save_path):
            os.makedirs(self.train_info_save_path)

        save_path = os.path.join(self.train_info_save_path, 'train_info.json')
        with open(save_path, 'w') as f:
            json.dump(parameters, f)