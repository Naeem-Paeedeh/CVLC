import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import logging
from torchvision.transforms import InterpolationMode
from utils.data import iGanFake, iCore50, iDomainNet, iOfficeHome
from configs import Configuration
import new_types as nt


class MultipleTransforms:
    def __init__(self, transforms_list: list):
        self.transforms_list = transforms_list

    def __call__(self, x):
        return [transform(x) for transform in self.transforms_list]


class DataManager(object):
    def __init__(
        self,
        cfg: Configuration,
        shuffle_class_order: bool,
        seed: int,
    ):
        self.cfg = cfg
        self.dataset_name = cfg.dataset_name
        self.shuffle = shuffle_class_order
        self.seed = seed
        
        self.num_classes = -1
        self.class_names_real = []
        self.class_names: list[str] = []
        
        self._class_order: list = []
        
        self.domain_names_for_this_order: list = []
        
        self._setup_data(self.dataset_name, shuffle_class_order=shuffle_class_order, seed=seed)
        
        init_cls = self.num_classes
        increment = self.num_classes
        
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments: list[int] = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]
    
    def get_dataset(
        self,
        indices,
        source,
        aug_modes_list: list,
        num_shots: int = -1,
        appendent=None
        # ret_data=False
        ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # The dataloader returns all transformations for the same images at the same time.
        transforms_list = []

        for mode in aug_modes_list:
            if mode == "train":
                transforms_list.append(transforms.Compose([*self._train_tansform, *self._common_transform]))
            elif mode == "test":
                transforms_list.append(transforms.Compose([*self._test_transform, *self._common_transform]))
            elif mode == "flip":
                transforms_list.append(transforms.Compose([*self._test_transform, transforms.RandomHorizontalFlip(p=1.0), *self._common_transform]))
            elif mode == "CLIP_aug":

                transform_clip = transforms.Compose([
                    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
                transforms_list.append(transform_clip)
            else:
                raise ValueError(f"Unknown mode: {mode}.")

        trsf = MultipleTransforms(transforms_list)

        data, targets = [], []
        rng = np.random.default_rng()

        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            
            # assert int(np.unique(class_targets)[0]) == idx
            
            if num_shots == -1 or num_shots == 0:     # We use all sample for this class
                data.append(class_data)
                targets.append(class_targets)
            elif num_shots > 0:
                # assert len(class_targets) >= num_shots
                class_data_shuffled = rng.permutation(class_data)
                data.append(class_data_shuffled[:num_shots])
                targets.append(class_targets[:num_shots])
            else:
                raise NotImplementedError("Incorrect setting!")
            
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        return DummyDataset(data, targets, trsf, self.use_path)

    def get_anchor_dataset(self, mode, appendent=None, ret_data=False):
        if mode == "train":
            trsf = transforms.Compose([*self._train_tansform, *self._common_transform])
        elif mode == "flip":
            trsf = transforms.Compose([*self._test_transform, transforms.RandomHorizontalFlip(p=1.0), *self._common_transform])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_transform, *self._common_transform])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_tansform, *self._common_transform])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_transform, *self._common_transform])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), DummyDataset(
            val_data, val_targets, trsf, self.use_path
        )

    def _setup_data(self, dataset_name, shuffle_class_order: bool, seed):
        idata = _get_idata(dataset_name, self.cfg)
        idata.download_data()
        self.num_classes = idata.num_classes
        self.class_names_real = idata.class_names_real
        self.class_names = idata.class_names.tolist()
        self.domain_names_for_this_order = idata.get_domain_names()

        # Data
        self._train_data = idata.train_data
        self._train_targets = idata.train_targets
        self._test_data = idata.test_data
        self._test_targets = idata.test_targets
        
        self.use_path = idata.use_path
        if hasattr(idata, "MANY_SHOT_THRES"):
            print("Splitting classes based on shot thresholds")
            self.split_class(self._train_targets, idata.MANY_SHOT_THRES, idata.FEW_SHOT_THRES)
        else:
            print("Splitting classes based on default shot thresholds")
            self.split_class(self._train_targets, 100, 20)
        # Transforms
        self._train_tansform = idata.train_transform
        self._test_transform = idata.test_transform
        self._common_transform = idata.common_transform

        # Order
        if dataset_name == "officehome":
            order = list(range(65 * 4))
        else:
            order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle_class_order:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        # logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def split_class(self, targets, many_shot_thres, few_shot_thres):
        class_counts = {}
        for target in targets:
            if target not in class_counts:
                class_counts[target] = 0
            class_counts[target] += 1

        # Categorize classes based on counts
        self.many_shot_classes = []
        self.medium_shot_classes = []
        self.few_shot_classes = []

        for class_id, count in class_counts.items():
            if count >= many_shot_thres:
                self.many_shot_classes.append(class_id)
            elif count <= few_shot_thres:
                self.few_shot_classes.append(class_id)
            else:
                self.medium_shot_classes.append(class_id)

        # Sort for consistency
        self.many_shot_classes.sort()
        self.medium_shot_classes.sort()
        self.few_shot_classes.sort()
        self.many_shot_thres = many_shot_thres
        self.few_shot_thres = few_shot_thres
        
    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        labels = y[idxes]
        # names_list = [self.class_names[label] for label in labels]
        return x[idxes], labels


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cddb":
        return iGanFake(args)
    elif name == "core50":
        return iCore50(args)
    elif name == "domainnet":
        return iDomainNet(args)
    elif name == "officehome":
        return iOfficeHome(args)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
