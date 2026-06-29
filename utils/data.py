import os
from typing import Any, List
import numpy as np
import torch
from numpy._typing._array_like import NDArray
from torchvision import datasets, transforms
from torchvision.transforms import v2
from utils.toolkit import split_images_labels
from utils.datautils.core50data import CORE50
import random
import json
import csv
from configs import Configuration
from torchvision.transforms import InterpolationMode
from abc import ABC, abstractmethod
import our_utils as ou
import logging


class iData(object):
    def __init__(self, cfg: Configuration) -> None:
        self.cfg = cfg
        self.train_data = np.ndarray
        self.train_targets = np.ndarray
        self.test_data = np.ndarray
        self.test_targets = np.ndarray
        self.num_classes = -1
        
        self.train_transform = []
        self.test_transform = []
        self.common_transform = []
        self.class_order = None
        self.domain_names = []
        self.class_names_real = np.array([])
        self.class_names = np.array([])
        
        self.class_names_real, self.class_names = ou.obtain_class_names(cfg.dataset_name)
        
        pass
    
    def count_samples_per_class(self, labels: np.ndarray):
        counts_all = np.zeros(len(self.domain_names) * self.num_classes)
        indices_unique_elements, counts_unique_elements = np.unique(labels, return_counts=True)
        counts_all[indices_unique_elements] = counts_unique_elements
        return counts_all, indices_unique_elements, counts_unique_elements
    
    def remove_classes_with_insufficient_samples(
        self,
        num_shots_train: int,
        num_shots_test: int
    ):
        # This method removes the class from all domains if it doesnt' have sufficient samples in one of the domains.
        
        num_classes_old = self.num_classes      # There are 345 classes for DomainNet at the beginning.
        
        assert num_shots_train > 0
        assert num_shots_test > 0
        
        counts_train, indices_useful_classes_train, counts_useful_classes_train = self.count_samples_per_class(labels=self.train_targets)
        counts_test, indices_useful_classes_test, counts_useful_classes_test = self.count_samples_per_class(labels=self.test_targets)
        
        classes_to_remove_train = np.nonzero(counts_train < num_shots_train)[0]
        classes_to_remove_test = np.nonzero(counts_test < num_shots_test)[0]
        
        # Labels by ignoring the domain
        classes_to_remove_train = classes_to_remove_train % num_classes_old
        classes_to_remove_test = classes_to_remove_test % num_classes_old
        
        # If a class does not have sufficient samples for train or test set in one of the domains, we discard that class.
        classes_to_remove_DIL = np.array(list(set(classes_to_remove_train) | set(classes_to_remove_test)))
        
        flag = len(classes_to_remove_DIL) > 0
        
        if flag:
            logging.info(f"The following {len(classes_to_remove_DIL)} classes are discarded because their samples are insufficient: \n{classes_to_remove_DIL}\n{self.class_names_real[classes_to_remove_DIL]}")
        
            self.num_classes -= len(classes_to_remove_DIL)
        
            classes_to_remove_from_all_domains = np.array([])
        
            # We prepare the same classes from all domains to be ignored.
            for domain_id, _ in enumerate(self.domain_names):
                classes_to_remove_from_all_domains = np.concatenate([classes_to_remove_from_all_domains, classes_to_remove_DIL + domain_id * num_classes_old], axis=0)
            
            mask_train = ~np.isin(self.train_targets, classes_to_remove_from_all_domains)
            mask_test = ~np.isin(self.test_targets, classes_to_remove_from_all_domains)
            
            # self.domain_names = [name for i, name in enumerate(self.domain_names) if i not in classes_to_remove_train]
            
            self.train_data = self.train_data[mask_train]
            self.train_targets = self.train_targets[mask_train]
            
            self.test_data = self.test_data[mask_test]
            self.test_targets = self.test_targets[mask_test]
        
        # The new class order
        self.class_order = np.arange(len(self.domain_names) * self.num_classes).tolist()
        
        if flag:
            classes_to_keep = np.ones(num_classes_old, dtype=np.long)
            classes_to_keep[classes_to_remove_DIL] = 0
            
            mask_classes_to_keep = classes_to_keep == 1
            
            self.class_names_real = self.class_names_real[mask_classes_to_keep]
            self.class_names = self.class_names[mask_classes_to_keep]
            
            # Remaps the labels from zero to the new number of classes
            labels_unique = np.unique(self.test_targets)
        
        assert len(labels_unique) == self.num_classes * len(self.domain_names)
        assert len(labels_unique) > 1
        
        # Remapping the labels to start from zero without any gap.
        if flag:
            train_targets_old_copy = self.train_targets.copy()
            test_targets_old_copy = self.test_targets.copy()
            
            self.train_targets = np.zeros_like(self.train_targets) - 1
            self.test_targets = np.zeros_like(self.test_targets) - 1
            
            for new_label, old_label in enumerate(labels_unique.tolist()):
                self.train_targets[train_targets_old_copy == old_label] = new_label
                self.test_targets[test_targets_old_copy == old_label] = new_label
                
        # Final verification
        assert self.train_targets.min().item() == 0
        assert self.test_targets.min().item() == 0
        
        assert self.train_targets.max().item() == self.num_classes * len(self.domain_names) - 1
        assert self.test_targets.max().item() == self.num_classes * len(self.domain_names) - 1
        
        assert len(np.unique(self.train_targets)) == self.num_classes * len(self.domain_names)
        assert len(np.unique(self.test_targets)) == self.num_classes * len(self.domain_names)
        pass
            

class iGanFake(iData):
    def __init__(self, cfg: Configuration):
        super().__init__(cfg)
        self.cfg = cfg
        class_order = cfg.class_order
        self.class_order = class_order
        self.num_classes = 2
        
        self.MANY_SHOT_THRES = 70
        self.FEW_SHOT_THRES = 20
        self.use_path = True
        
        self.train_transform = [
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
        ]
        
        self.test_transform = [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
        
        self.common_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    def make_imb(self, dataset_dict, root_dir):
        imb_info = {
            "gaugan": [3000, 150],
            "biggan": [120, 1200],
            "wild": [155, 3115],
            "whichfaceisreal": [600, 120],
            "san": [130, 130],
        }
        for name in dataset_dict.keys():
            pos_trian_num = imb_info[name][0]
            neg_train_num = imb_info[name][1]
            pos_train = dataset_dict[name]["train_pos_naive"]
            neg_train = dataset_dict[name]["train_neg_naive"]
            random.shuffle(pos_train)
            pos_train = pos_train[:pos_trian_num]
            random.shuffle(neg_train)
            neg_train = neg_train[:neg_train_num]
            dataset_dict[name]["train_pos"] = pos_train
            dataset_dict[name]["train_neg"] = neg_train

        with open(os.path.join("utils/datautils", "CDDB.json"), "w") as f:
            json.dump(dataset_dict, f)

    def download_data(self):
        with open(os.path.join("utils/datautils", "CDDB.json"), "r") as f:
            dataset_dict = json.load(f)
        train_dataset = []
        test_dataset = []
        task_list = self.get_domain_names()
        print(task_list)
        for id, name in enumerate(task_list):
            pos_list = dataset_dict[name]["train_pos"]
            neg_list = dataset_dict[name]["train_neg"]
            for imgname in pos_list:
                train_dataset.append((os.path.join(self.cfg.data_path, imgname), 1 + 2 * id))
            for imgname in neg_list:
                train_dataset.append((os.path.join(self.cfg.data_path, imgname), 0 + 2 * id))
            for imgname in dataset_dict[name]["test_pos"]:
                test_dataset.append((os.path.join(self.cfg.data_path, imgname), 1 + 2 * id))
            for imgname in dataset_dict[name]["test_neg"]:
                test_dataset.append((os.path.join(self.cfg.data_path, imgname), 0 + 2 * id))
            print("Task {}, train_pos {}, train_neg {}".format(name, len(pos_list), len(neg_list)))
        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)

    def get_domain_names(
        self
    ) -> List:
        order_list = [
            ["wild", "whichfaceisreal", "san", "gaugan", "biggan"],     # Order 1
            ["gaugan", "biggan", "wild", "whichfaceisreal", "san"],     # Order 2
            ["whichfaceisreal", "gaugan", "wild", "san", "biggan"],     # Order 3
            ["gaugan", "whichfaceisreal", "san", "wild", "biggan"],     # Order 4
            ["wild", "biggan", "gaugan", "san", "whichfaceisreal"]      # Order 5
        ]
        return order_list[self.cfg.order - 1]


# This updated class is compatible with the PGO-BEn's evaluation protocol.
class iCore50(iData):
    def __init__(self, cfg: Configuration):
        super().__init__(cfg)
        self.cfg = cfg
        self.num_classes = 50
        self.class_order = np.arange(8 * 50).tolist()   # 400 DIL "classes"

        # 8 sessions are the 8 domains (order is applied inside CORE50)
        self.domain_names = [f"s{i + 1}" for i in range(8)]
        self.order_list_0based: list = None

        self.MANY_SHOT_THRES = 60
        self.FEW_SHOT_THRES = 20
        self.use_path = True
        self.train_transform = [
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_transform = [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
        ]
        self.common_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    def download_data(self):
        datagen = CORE50(
            root=self.cfg.data_path,
            scenario="ni",                 # Domain-incremental setting
            order=self.cfg.order,
            preload=False,
        )
        
        self.order_list_0based = datagen.order_list_0based

        # Fixed seed -> identical train/test split across all sessions & runs
        split_seed = self.cfg.seed_current + 1993 #  getattr(self.cfg, "core50_split_seed", 1993)
        rng = np.random.default_rng(split_seed)
        test_fraction = getattr(self.cfg, "core50_test_fraction", 0.2)

        train_x, train_y = [], []
        test_x, test_y = [], []

        # Each training batch == one session == one domain (already reordered)
        for domain_id, batch_idx in enumerate(datagen._train_batches_idx):
            batch_idx = np.asarray(batch_idx)
            batch_y = np.asarray(datagen._train_batches_y[domain_id], dtype=np.int32)  # 0..49
            domain_y = batch_y + domain_id * 50                                        # domain-aware

            # Stratified per-class split so every domain gets a test set for all 50 classes
            for cls in np.unique(batch_y):
                pos = rng.permutation(np.where(batch_y == cls)[0])
                n_test = max(1, int(round(len(pos) * test_fraction)))
                test_pos, train_pos = pos[:n_test], pos[n_test:]

                train_x.extend(os.path.join(datagen.root, datagen.paths[batch_idx[p]]) for p in train_pos)
                train_y.extend(domain_y[p] for p in train_pos)

                test_x.extend(os.path.join(datagen.root, datagen.paths[batch_idx[p]]) for p in test_pos)
                test_y.extend(domain_y[p] for p in test_pos)

        self.train_data    = np.array(train_x)
        self.train_targets = np.array(train_y, dtype=np.int32)
        self.test_data     = np.array(test_x)
        self.test_targets  = np.array(test_y, dtype=np.int32)
        
    def get_domain_names(self):
        order_list_0based = self.order_list_0based
        domain_names_for_this_order = [self.domain_names[order_list_0based[i]] for i in range(8)]
        return domain_names_for_this_order


class iDomainNet(iData):
    def __init__(self, cfg: Configuration):
        super().__init__(cfg)
        self.cfg = cfg
        self.num_classes = 345
        self.domain_order = cfg.order
        self.domain_names = self.get_domain_names()
        print(self.domain_names)
        self.class_order = np.arange(len(self.domain_names) * self.num_classes).tolist()
        
        self.MANY_SHOT_THRES = 100
        self.FEW_SHOT_THRES = 20
        self.use_path = True
        
        self.train_transform = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
        ]
        
        self.test_transform = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
        ]

        self.common_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    def get_domain_names(self):
        if self.domain_order == 1:
            return ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        elif self.domain_order == 2:
            return ["infograph", "painting", "sketch", "clipart", "quickdraw", "real"]
        elif self.domain_order == 3:
            return ["painting", "quickdraw", "real", "sketch", "clipart", "infograph"]
        elif self.domain_order == 4:
            return ["real", "sketch", "painting", "infograph", "quickdraw", "clipart"]
        elif self.domain_order == 5:
            return ["sketch", "clipart", "quickdraw", "real", "infograph", "painting"]
        elif self.domain_order == 6:        # PGO-BEn
            return ["real", "painting", "clipart", "sketch", "quickdraw", "infograph"]
        else:
            raise NotImplementedError
        
    def download_data(self):
        self.image_list_root = self.cfg.data_path
        reversed_data = {name: index for index, name in enumerate(self.class_names_real)}
        train_x, train_y = [], []
        test_x, test_y = [], []
        with open(os.path.join("utils/datautils", "domainnet.csv"), "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            print(header)
            row_number = 1
            for row in csv_reader:
                domain_type = row[0]
                data_path = row[2]
                cls_name = os.path.basename(os.path.dirname(data_path))
                data_type = row[3]
                cls_id = reversed_data[cls_name]
                domain_id = self.domain_names.index(domain_type)
                absulute_path = os.path.join(self.image_list_root, data_path)
                if data_type == "train":
                    train_x.append(absulute_path)
                    train_y.append(cls_id + domain_id * 345)
                else:
                    test_x.append(absulute_path)
                    test_y.append(cls_id + domain_id * 345)
                    
                row_number += 1
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)
        
        # We consider the worst case scenario because we want to always use the same model trained on the first domain and the same classes.
        self.remove_classes_with_insufficient_samples(num_shots_train=8, num_shots_test=self.cfg.min_shots_test_set)


class iOfficeHome(iData):
    def __init__(self, cfg: Configuration):
        super().__init__(cfg)
        self.cfg = cfg
        class_order = np.arange(4 * 65).tolist()
        self.class_order = class_order
        self.domain_order = cfg.order
        self.num_classes = 65
        self.domain_names = self.get_domain_names()
        print(self.domain_names)
        self.MANY_SHOT_THRES = 60
        self.FEW_SHOT_THRES = 20
        self.use_path = True
        self.train_transform = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
        ]
        self.test_transform = [transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)]

        self.common_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    def get_domain_names(self):
        if self.domain_order == 1:
            return ["Art", "Clipart", "Product", "Real World"]
        elif self.domain_order == 2:
            return ["Clipart", "Art", "Real World", "Product"]
        elif self.domain_order == 3:
            return ["Product", "Clipart", "Art", "Real World"]
        elif self.domain_order == 4:
            return ["Real World", "Product", "Clipart", "Art"]
        elif self.domain_order == 5:
            return ["Art", "Real World", "Product", "Clipart"]
        else:
            raise NotImplementedError

    def download_data(self):
        self.image_list_root = self.cfg.data_path
        reversed_data = {name: index for index, name in enumerate(self.class_names_real)}
        train_x, train_y = [], []
        test_x, test_y = [], []
        with open(os.path.join("utils/datautils", "officehome.csv"), "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            print(header)
            for row in csv_reader:
                domain_type = row[0]
                data_path = row[2]
                data_path = data_path.replace("office_home/", "")
                cls_name = os.path.basename(os.path.dirname(data_path))
                data_type = row[3]
                cls_id = reversed_data[cls_name]
                domain_id = self.domain_names.index(domain_type)
                absulute_path = os.path.join(self.image_list_root, data_path)
                if data_type == "train":
                    train_x.append(absulute_path)
                    train_y.append(cls_id + domain_id * 65)
                else:
                    test_x.append(absulute_path)
                    test_y.append(cls_id + domain_id * 65)
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)
