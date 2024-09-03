import numpy as np
import inspect


class MetricsCalculator:
    def __init__(self):
        self.metrics = {}
        self.load_metrics()

    def load_metrics(self):
        classes = inspect.getmembers(
            inspect.getmodule(inspect.currentframe()),
            lambda cls: inspect.isclass(cls) and issubclass(cls, Metric) and cls is not Metric
        )

        for name, cls in classes:
            name = name.split('Metric')[0]
            instance = cls()
            if hasattr(instance, 'calculate_metric'):
                self.metrics[name] = instance.calculate_metric

    def calculate_metrics(self, gt_masks, pred_masks):
        num_classes = len(gt_masks)

        metrics = {name: [] for name in self.metrics}

        for i in range(num_classes):
            for name, fn in self.metrics.items():
                value = fn(gt_masks[i], pred_masks[i])
                metrics[name].append(value)

        metrics_dict = {
            name: np.mean(values) for name, values in metrics.items()
        }

        return metrics_dict


class Metric:
    def __init__(self) -> None:
        pass


class IOUMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.name = "IOU"


    @staticmethod
    def calculate_metric(gt_mask, pred_mask):
        intersection = np.logical_and(gt_mask, pred_mask)
        union = np.logical_or(gt_mask, pred_mask)
        iou = np.sum(intersection) / np.sum(union)
        return iou


class PrecisionMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Precision"

    @staticmethod
    def calculate_metric(gt_mask, pred_mask):
        true_positive = np.sum(np.logical_and(gt_mask, pred_mask))
        false_positive = np.sum(np.logical_and(np.logical_not(gt_mask), pred_mask))
        precision = true_positive / (true_positive + false_positive + 1e-7)
        return precision


class RecallMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Recall"

    @staticmethod
    def calculate_metric(gt_mask, pred_mask):
        true_positive = np.sum(np.logical_and(gt_mask, pred_mask))
        false_negative = np.sum(np.logical_and(gt_mask, np.logical_not(pred_mask)))
        recall = true_positive / (true_positive + false_negative + 1e-7)
        return recall


class F1ScoreMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.name = "F1-Score"

    @staticmethod
    def calculate_metric(gt_mask, pred_mask):
        true_positive = np.sum(np.logical_and(gt_mask, pred_mask))
        false_positive = np.sum(np.logical_and(np.logical_not(gt_mask), pred_mask))
        false_negative = np.sum(np.logical_and(gt_mask, np.logical_not(pred_mask)))
        precision = true_positive / (true_positive + false_positive + 1e-7)
        recall = true_positive / (true_positive + false_negative + 1e-7)
        f1_score = 2 * precision * recall / (precision + recall + 1e-7)
        return f1_score


class AccuracyMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Accuracy"

    @staticmethod
    def calculate_metric(gt_mask, pred_mask):
        true_positive = np.sum(np.logical_and(gt_mask, pred_mask))
        true_negative = np.sum(np.logical_and(np.logical_not(gt_mask), np.logical_not(pred_mask)))
        false_positive = np.sum(np.logical_and(np.logical_not(gt_mask), pred_mask))
        false_negative = np.sum(np.logical_and(gt_mask, np.logical_not(pred_mask)))
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative + 1e-7)
        return accuracy

