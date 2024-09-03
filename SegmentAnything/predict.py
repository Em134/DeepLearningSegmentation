import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from scripts import create_datasets, plot_pr_curve_for_dataset, split_batch_images_labels, visualize_predictions, draw_contours_on_image
import numpy as np
from sklearn.metrics import precision_recall_curve
import cv2


batch_size = 1
train_dataset, val_dataset, test_dataset = create_datasets(dataset_info_path='dataset/TNBC/dataset_info.json')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model_type = 'vit_b'
pretrained_weight_path = 'weights/MySam/BinaryMaskLoss/lr0.001_bs2_pwTrue_fimage_encoder+prompt_encoder_add/2024-04-11-13.42_16.pth'
sam = sam_model_registry[model_type](pretrained_weight_path)
sam.eval()

pred_arr_list = []
label_list = []
for batch in test_loader:
    res = sam(batch[0], False)
    arr = torch.sigmoid(res).squeeze().detach().numpy()
    pred_arr_list.append(arr)
    label_list.append(batch[1].squeeze().numpy().astype(int))

threshold = 0.87
images_list = []
labels_list = []
preds_list = []

for idx, batch in enumerate(test_loader):
    res = sam(batch[0], False)
    for i in range(len(res)):
        arr = torch.sigmoid(res)[i].squeeze(0).detach().numpy()
        preds_list.append(arr)
    temp_imgs, temp_labels = split_batch_images_labels(batch)
    images_list.extend(temp_imgs)
    labels_list.extend(temp_labels)
    
for i in range(len(images_list)):
    arr = preds_list[i]
    original_label = labels_list[i].astype(np.uint8)
    ori_image = images_list[i]

    binary = arr > threshold
    binary = binary.astype(np.uint8)
    diff_mask = visualize_predictions(original_label, binary)
    ori_image = draw_contours_on_image(binary, ori_image)

    
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    # 设置第一个子图为预测结果
    axes[0].imshow(binary, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Prediction')

    # 设置第二个子图为原始标签
    axes[1].imshow(original_label, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Original Label')

    # 设置第三个子图为叠加结果
    axes[2].imshow(diff_mask)
    axes[2].axis('off')
    axes[2].set_title('G:TP R:FP B:FN')

    # 设置第三个子图为原始图像
    axes[3].imshow(ori_image)  # 假设原始图像是3通道的
    axes[3].axis('off')
    axes[3].set_title('Original Image')
    # 调整子图之间的间距
    plt.tight_layout()
    # 保存整个图形
    plt.savefig('predict_results/result_figure_{}.png'.format(i))
    plt.close()