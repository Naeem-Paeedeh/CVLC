import os
import numpy as np
import torch
from torch import Tensor as T


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()
    elif isinstance(x, (int, float, np.integer, np.floating)):
        return np.array(x)
    else:
        raise TypeError("Input must be a torch.Tensor, int, or float.")


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def accuracy_domain(y_pred, y_true, nb_old, increment=2, class_num=1):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around((y_pred % class_num == y_true % class_num).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = np.around(
            ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def accuracy_domain_shot(
    labels_predicted: T,
    labels_ground_truth: T,
    known_classes: int,
    num_classes: int,
    many_shot=None,
    medium_shot=None,
    few_shot=None,
):
    assert labels_predicted.shape[0] == labels_ground_truth.shape[0], "Data length error."

    # Predictions/labels reduced modulo num_classes (compared everywhere)
    correct = (labels_predicted % num_classes) == (labels_ground_truth % num_classes)

    def acc(mask, empty=0.0):
        """Rounded % accuracy over a boolean/index mask (returns `empty` if no samples)."""
        n = int(mask.sum()) if mask.dtype == torch.bool else mask.numel()
        if n == 0:
            return float(empty)
        return round(correct[mask].sum().item() * 100 / n, 2)

    easy = num_classes == 2
    max_label = int(labels_ground_truth.max())
    acc_dict = {
        "total": acc(torch.ones_like(labels_ground_truth, dtype=torch.bool))
    }

    # Per-domain accuracy (+ shot-based analysis when not "easy")
    for domain_id, class_id in enumerate(range(0, max_label, num_classes)):
        # Labels that belong to this domain
        mask = (labels_ground_truth >= class_id) & (labels_ground_truth < class_id + num_classes)
        acc_dict[domain_id] = acc(mask)

        if not easy:
            task_labels = labels_ground_truth[mask]
            for name, shot in [("many", many_shot), ("medium", medium_shot), ("few", few_shot)]:
                if shot is None:
                    continue
                shot = torch.as_tensor(shot)
                shot_mask = torch.isin(task_labels, shot)
                acc_dict[f"{domain_id}_{name}"] = acc(correct[mask][shot_mask], empty=-1) \
                    if shot_mask.any() else -1.0

    # Grouped (domain) accuracy
    if easy:
        domain = [
            acc((labels_ground_truth >= c) & (labels_ground_truth < c + num_classes))
            for c in range(0, max_label, 1)
        ]
        acc_dict["domain"] = float(torch.tensor(domain).mean())
    else:
        for class_id in range(0, max_label, num_classes):
            mask = (labels_ground_truth >= class_id) & (labels_ground_truth < class_id + num_classes)
            label = f"{class_id:02d}-{class_id + num_classes - 1:02d}"
            acc_dict[label] = acc(mask)

    # --- Old / New accuracy ---
    acc_dict["old"] = acc(labels_ground_truth < known_classes, empty=0)
    acc_dict["new"] = acc(labels_ground_truth >= known_classes)

    return acc_dict


# def accuracy_binary(y_pred, y_true, nb_old, increment=2):
#     assert len(y_pred) == len(y_true), "Data length error."
#     all_acc = {}
#     all_acc["total"] = np.around((y_pred % 2 == y_true % 2).sum() * 100 / len(y_true), decimals=2)

#     # Grouped accuracy
#     for class_id in range(0, np.max(y_true), increment):
#         idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
#         label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
#         all_acc[label] = np.around(((y_pred[idxes] % 2) == (y_true[idxes] % 2)).sum() * 100 / len(idxes), decimals=2)

#     # Old accuracy
#     idxes = np.where(y_true < nb_old)[0]
#     # all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),decimals=2)
#     all_acc["old"] = (
#         0
#         if len(idxes) == 0
#         else np.around(((y_pred[idxes] % 2) == (y_true[idxes] % 2)).sum() * 100 / len(idxes), decimals=2)
#     )

#     # New accuracy
#     idxes = np.where(y_true >= nb_old)[0]
#     all_acc["new"] = np.around(((y_pred[idxes] % 2) == (y_true[idxes] % 2)).sum() * 100 / len(idxes), decimals=2)

#     return all_acc
