import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
import cv2
import os
import json


def plot_pr_curve_for_dataset(y_true_list, y_pred_prob_list, show=False):
    all_y_true = np.concatenate(y_true_list)
    all_y_pred_prob = np.concatenate(y_pred_prob_list)

    precision, recall, thresholds = precision_recall_curve(all_y_true.flatten(), all_y_pred_prob.flatten())
    area_under_curve = np.trapz(precision, recall)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label='Precision-Recall curve (area = {:.2f})'.format(area_under_curve))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)

    # 寻找最佳阈值点
    best_threshold_index = np.argmax(precision + recall)
    best_threshold = thresholds[best_threshold_index]
    plt.scatter(recall[best_threshold_index], precision[best_threshold_index], c='red', s=50, label='Best Threshold: {:.2f}'.format(best_threshold))
    plt.legend(loc='best')

    if show:
        plt.show()
    return best_threshold


def visualize_predictions(label_image, prediction_image):
    # 确保两个图像具有相同的尺寸
    if label_image.shape != prediction_image.shape:
        raise ValueError("Label and prediction images must have the same dimensions.")

    # 创建新图像，尺寸与输入图像相同
    fn = (prediction_image.astype(int) - label_image.astype(int)) < 0
    fp = (label_image.astype(int) - prediction_image.astype(int)) < 0
    tp = prediction_image.astype(int) - fp.astype(int)

    return np.array([fp.astype(int), tp.astype(int), fn.astype(int)]).transpose(1, 2, 0).astype(np.uint8) * 255


def split_batch_images_labels(batch):
    images_list = []
    labels_list = []
    for i in range(len(batch[0])):
        image = batch[0][i].detach().numpy().transpose(1, 2, 0)  # 将通道维度放在最后
        label = batch[1][i].detach().numpy().squeeze(0)  # 去掉第一个维度，因为标签的维度是[bs, 1, h, w]

        images_list.append(image)
        labels_list.append(label)
    return images_list, labels_list


def draw_contours_on_image(binary_mask, original_image):
    # 寻找连通域
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像，与原始图像大小和通道数相同
    result_image = original_image.copy()

    # 在结果图像上绘制每个连通域的轮廓
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

    return result_image


def get_threshold(val_loader, segmentation_model, device, save_dir):
    if not os.path.exists(os.path.join(save_dir, 'postprocess_info.json')):
        threshold = calculate_threshold(val_loader, segmentation_model, device)
        save_threshold_to_json(threshold, save_dir)
    else:
        threshold = load_threshold_from_json(os.path.join(save_dir, 'postprocess_info.json'))

    return threshold


def calculate_threshold(val_loader, segmentation_model, device):
    predictions_list = []
    ground_truth_list = []
    for batch in val_loader:
        inputs = batch[0].to(device)
        targets = batch[1].squeeze().numpy().astype(int)
        outputs = segmentation_model(inputs, False)
        if device.type == 'cuda':
            outputs = outputs.cpu()
        probabilities = torch.sigmoid(outputs).squeeze().detach().numpy()
        predictions_list.append(probabilities)
        ground_truth_list.append(targets)

    threshold = plot_pr_curve_for_dataset(ground_truth_list, predictions_list)
    return threshold


def save_threshold_to_json(threshold, save_dir):
    parameters = {
        'threshold': float(threshold),
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'postprocess_info.json')
    with open(save_path, 'w') as f:
        json.dump(parameters, f)


def load_threshold_from_json(json_path):
    with open(json_path, 'r') as f:
        parameters = json.load(f)
        threshold = parameters['threshold']
    return threshold