import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from PIL import Image
import random
import json


class MyDataset(Dataset):
    def __init__(self, 
                 base_path: str, 
                 image_path: Optional[str] = None, 
                 mask_path: Optional[str] = None, 
                 name_list: Optional[List[str]] = None, 
                 ) -> None:
        super().__init__()
        self.base_path = base_path

        if name_list is not None:
            self.name_list = name_list
        else:
            self.name_list = os.listdir(os.path.join(base_path, 'images'))

        if image_path is not None:
            self.image_path = image_path
        else:
            self.image_path = os.path.join(base_path, 'images')

        if mask_path is not None:
            self.mask_path = mask_path
        else:
            self.mask_path = os.path.join(base_path, 'masks')

    def __len__(self):
        return len(self.name_list)
    
    def standardize_tensor(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std
    
    def normalize_tensor(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
    def __getitem__(self, index) -> Dict:
        image_array = np.array(Image.open(os.path.join(self.image_path, self.name_list[index])).convert("RGB"))
        mask_array = np.array(Image.open(os.path.join(self.mask_path, self.name_list[index])).convert("L"))

        image_tensor = torch.Tensor(image_array.transpose(2, 0, 1))
        mask_tensor = torch.Tensor(mask_array).unsqueeze(0)

        image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False).squeeze()
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False).squeeze(0)
        
        return self.normalize_tensor(self.standardize_tensor(image_tensor)), self.normalize_tensor(self.standardize_tensor(mask_tensor))
    

class EasyDataset(Dataset):
    def __init__(self, dataset_path) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.name_list = os.listdir(os.path.join(dataset_path, 'images'))

    def __len__(self):
        return len(self.name_list)
    
    def standardize_tensor(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image_array = np.array(Image.open(os.path.join(self.dataset_path, 'images', self.name_list[index])).convert("RGB"))
        mask_array = np.array(Image.open(os.path.join(self.dataset_path, 'masks', self.name_list[index])).convert("L"))

        image_tensor = self.standardize_tensor(torch.Tensor(image_array.transpose(2, 0, 1)))
        mask_tensor = self.standardize_tensor(torch.Tensor(mask_array).unsqueeze(0))

        return image_tensor, mask_tensor


class DatasetSplitter(object):
    def __init__(self, 
                 base_path: str, 
                 image_filefolder_name: Optional[str] = 'images', 
                 mask_filefolder_name: Optional[str] = 'masks'
                 ) -> None:
        """
        Initialize the dataset generator object.

        Args:
        - base_path (str): The base folder path of the dataset.
        - image_filefolder_name (str, optional): The folder name for storing image files. Defaults to 'images'.
        - mask_filefolder_name (str, optional): The folder name for storing segmentation label files. Defaults to 'masks'.
        """
        self.base_path = base_path
        self.image_path = os.path.join(base_path, image_filefolder_name)
        self.mask_path = os.path.join(base_path, mask_filefolder_name)

        assert os.path.exists(self.image_path), f"Please ensure that the '{image_filefolder_name}' folder exists in '{base_path}'."
        assert os.path.exists(self.mask_path), f"Please ensure that the '{mask_filefolder_name}' folder exists in '{base_path}'."

        image_files = os.listdir(self.image_path)
        mask_files = os.listdir(self.mask_path)

        assert len(image_files) == len(mask_files), "The number of image files and mask files should be the same."

        image_files.sort()
        mask_files.sort()

        for image_file, mask_file in zip(image_files, mask_files):
            assert os.path.isfile(os.path.join(self.image_path, image_file)), f"'{image_file}' is not a file in '{image_filefolder_name}' folder."
            assert os.path.isfile(os.path.join(self.mask_path, mask_file)), f"'{mask_file}' is not a file in '{mask_filefolder_name}' folder."

        self.name_list = image_files

    def split_dataset(self, 
                      proportions: Optional[List[int]] = [7, 2, 1], 
                      output_directory: Optional[str] = None, 
                      random_seed: Optional[int] = 42, 
                      save_name: Optional[str] = 'dataset_info', 
                      ) -> None:
        """
        Split the dataset into training, validation, and testing sets based on the given proportions.

        Args:
        - proportions (List[int]): List of three integers representing the proportions of training, validation, and testing sets respectively. The sum of the three integers should be 10.
        - output_directory (str, optional): The directory to store the resulting dictionary. Defaults to self.base_path.
        """
        assert sum(proportions) == 10, "The sum of proportions should be 10."

        random.seed(random_seed)  # Set a random seed for reproducibility

        dataset_size = len(self.name_list)
        indices = list(range(dataset_size))
        random.shuffle(indices)

        train_ratio = proportions[0] / 10
        val_ratio = proportions[1] / 10
        test_ratio = proportions[2] / 10

        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        dataset_info = {
            'random_seed': random_seed,
            'base_path': self.base_path,
            'image_path': self.image_path,
            'mask_path': self.mask_path, 
            'train': [self.name_list[i] for i in train_indices],
            'val': [self.name_list[i] for i in val_indices],
            'test': [self.name_list[i] for i in test_indices]
        }
        self.dataset_info = dataset_info

        output_directory = output_directory or self.base_path
        os.makedirs(output_directory, exist_ok=True)

        output_file = os.path.join(output_directory, save_name + '.json')
        with open(output_file, 'w') as f:
            json.dump(dataset_info, f)

        print(f"The dataset splits have been saved to: {output_file}")


def create_datasets(dataset_info_path: Optional[str] = None, 
                    dataset_info: Optional[dict] = None, 
                    ):
    """
    Create datasets based on the provided dataset information from either a JSON file or a dictionary.

    Args:
        dataset_info_path (str, optional): The path to the JSON file containing the dataset information.
            Defaults to None.
        dataset_info (dict, optional): The dictionary containing the dataset information.
            Defaults to None.

    Returns:
        tuple: A tuple containing three dataset objects - train_dataset, val_dataset, and test_dataset.
    """
    # Check if both parameters are provided
    if dataset_info_path is not None and dataset_info is not None:
        raise ValueError("Only one of 'dataset_info_path' and 'dataset_info' should be provided.")

    # Read dataset information from the JSON file path
    if dataset_info_path is not None:
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)

    # Extract parameters and lists from the dictionary
    base_path = dataset_info['base_path']
    image_path = dataset_info['image_path']
    mask_path = dataset_info['mask_path']
    train_list = dataset_info['train']
    val_list = dataset_info['val']
    test_list = dataset_info['test']

    # Create MyDataset objects
    train_dataset = MyDataset(base_path, image_path, mask_path, train_list)
    val_dataset = MyDataset(base_path, image_path, mask_path, val_list)
    test_dataset = MyDataset(base_path, image_path, mask_path, test_list)

    return train_dataset, val_dataset, test_dataset