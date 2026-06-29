#!/usr/bin/env python

""" Data Loader for the CORe50 Dataset """

from __future__ import print_function, division, absolute_import

import numpy as np
import pickle as pkl
import os
import logging
from hashlib import md5
from PIL import Image
from typing import List, Optional, Sequence, Union


class CORE50(object):
    """
    CORe50 Data Loader with support for domain-incremental task order reordering.

    Args:
        root (str): Root path containing CORe50 files:
            - core50_imgs.npz (or pre-cached core50_imgs.bin)
            - paths.pkl
            - LUP.pkl
            - labels.pkl
            - core50_128x128/ (optional; used if loading from paths with PIL)
        preload (bool): If True, preloads the entire dataset (high RAM usage).
        scenario (str): One of {"ni", "nc", "nic", "nicv2_79", "nicv2_196", "nicv2_391"}.
                        For domain-incremental learning, "ni" should be used.
        cumul (bool): If True, each batch accumulates all previous batches.
        run (int): One of the 10 official runs (from 0 to 9).
        start_batch (int): Start from this training batch index (0-based in the TASK ORDER).
        order (Union[int, Sequence[int], None]):
            - None or 0: original official order (no reordering)
            - 1 to 5: select a predefined 8-domain order (for NI)
            - custom list/tuple of length nbatch (0- or 1-based indices of the original order)
    """

    nbatch = {
        "ni": 8,
        "nc": 9,
        "nic": 79,
        "nicv2_79": 79,
        "nicv2_196": 196,
        "nicv2_391": 391,
    }

    def __init__(
        self,
        root: str = "",
        preload: bool = False,
        scenario: str = "ni",
        cumul: bool = False,
        run: int = 0,
        start_batch: int = 0,
        order: Optional[Union[int, Sequence[int]]] = None,
    ):
        self.root = os.path.expanduser(root)
        self.preload = preload
        self.scenario = scenario
        self.cumul = cumul
        self.run = run
        
        self.order_list_0based: list = []

        # Load CORe50 metadata
        self._load_metadata()

        # Preload all image bytes if requested (for fast iteration with high RAM usage)
        if self.preload:
            self._preload_images()

        # Build ordered list of training batches (indices + labels), with custom task order if requested
        self._prepare_batches(order)

        self._ptr = start_batch
        self._validate_start_batch(start_batch)

    def __iter__(self):
        # Reset iterator each time
        self._ptr = max(0, min(self._ptr, len(self._train_batches_idx)))
        return self

    def __next__(self):
        if self._ptr >= len(self._train_batches_idx):
            raise StopIteration

        batch_idx = self._ptr
        self._ptr += 1
        return self._load_batch_by_pointer(batch_idx)

    # Python 2 alias
    next = __next__

    def __len__(self):
        return len(self._train_batches_idx)

    def get_test_set(self):
        """
        Returns:
            test_x (np.ndarray uint8 HxWxC) and test_y (np.ndarray int)
        Note:
            The test set in CORe50 is the last entry in LUP/labels for each run/scenario.
        """
        scen = self.scenario
        run = self.run

        test_idx_list = self.LUP[scen][run][-1]
        if self.preload:
            test_x = np.take(self.x, test_idx_list, axis=0)
        else:
            # Build absolute disk paths and load on-the-fly
            test_paths = [os.path.join(self.root, self.paths[idx]) for idx in test_idx_list]
            test_x = self.get_batch_from_paths(test_paths).astype(np.uint8)

        test_y = np.asarray(self.labels[scen][run][-1], dtype=np.int32)
        return test_x, test_y

    def get_data_batchidx(self, idx: int):
        """
        Backward-compatible method to get a specific (possibly cumulative) batch by order pointer index.
        """
        if idx < 0 or idx >= len(self._train_batches_idx):
            raise IndexError(f"Batch idx {idx} out of range [0, {len(self._train_batches_idx) - 1}]")
        return self._load_batch_by_pointer(idx)

    def _validate_start_batch(self, start_batch: int):
        if start_batch < 0 or start_batch > len(self._train_batches_idx):
            raise ValueError(
                f"start_batch={start_batch} is invalid for {len(self._train_batches_idx)} training batches"
            )

    def _load_metadata(self):
        # Load core pickle structures
        print("Loading paths...")
        with open(os.path.join(self.root, "paths.pkl"), "rb") as f:
            self.paths = pkl.load(f)
        
        print("Loading LUP...")
        with open(os.path.join(self.root, "LUP.pkl"), "rb") as f:
            self.LUP = pkl.load(f)
        
        print("Loading labels...")
        with open(os.path.join(self.root, "labels.pkl"), "rb") as f:
            self.labels = pkl.load(f)

    def _preload_images(self):
        print("Loading data...")
        bin_path = os.path.join(self.root, "core50_imgs.bin")
        if os.path.exists(bin_path):
            with open(bin_path, "rb") as f:
                # Known shape for CORe50 128x128 RGB
                # self.x = np.fromfile(f, dtype=np.uint8).reshape(-1, 128, 128, 3)
                # Memory-map the cached .bin instead of reading it fully into RAM:
                self.x = np.memmap(bin_path, dtype=np.uint8, mode="r").reshape(-1, 128, 128, 3)
        else:
            npz_path = os.path.join(self.root, "core50_imgs.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(
                    f"Neither {bin_path} nor {npz_path} found. Place CORe50 image archive in {self.root}"
                )
            with open(npz_path, "rb") as f:
                npzfile = np.load(f)
                self.x = npzfile["x"]
            # Cache a flat bin for faster reloads
            self.x.tofile(bin_path)

    def _prepare_batches(self, order: Optional[Union[int, Sequence[int]]]):
        scen = self.scenario
        run = self.run

        if scen not in self.nbatch:
            raise ValueError(f"Unsupported scenario '{scen}'. Supported: {list(self.nbatch.keys())}")

        nb = self.nbatch[scen]

        # Original official train batches (EXCLUDING test batch at -1)
        orig_idx_batches: List[List[int]] = self.LUP[scen][run][:nb]
        orig_y_batches: List[List[int]] = self.labels[scen][run][:nb]

        # Resolve order mapping (mapping from pointer index -> original train batch index)
        self.order_list_0based = self._resolve_order(order, nb=nb)

        # Reorder batches according to desired order
        self._train_batches_idx = [orig_idx_batches[i] for i in self.order_list_0based]
        self._train_batches_y = [orig_y_batches[i] for i in self.order_list_0based]

    def _resolve_order(self, order: Optional[Union[int, Sequence[int]]], nb: int):
        """
        Returns a 0-based order list of length nb telling which original batch to serve at each pointer step.
        """
        # No reordering
        if order in (None, 0, "original"):
            return list(range(nb))

        # Predefined permutations for NI (8 domains) from DCE
        preset_orders_1based = [
            [8, 3, 2, 7, 1, 5, 4, 6],  # order=1
            [2, 7, 1, 5, 4, 6, 8, 3],  # order=2
            [3, 1, 7, 2, 4, 5, 6, 8],  # order=3
            [1, 7, 2, 4, 5, 6, 8, 3],  # order=4
            [7, 2, 4, 5, 6, 8, 3, 1],  # order=5
        ]

        if isinstance(order, int):
            if order < 0:
                raise ValueError("order must be >= 0")
            if order >= 1:
                if nb != 8:
                    raise ValueError(
                        f"Preset orders are defined for nbatch=8 (NI). Got nbatch={nb} for scenario."
                    )
                if order > len(preset_orders_1based):
                    raise ValueError(f"order={order} is out of range for presets (1..{len(preset_orders_1based)})")
                # convert to 0-based
                return [i - 1 for i in preset_orders_1based[order - 1]]

            # order==0 already handled above, but keep a fallback
            return list(range(nb))

        # List or tuple provided
        if isinstance(order, (list, tuple)):
            if len(order) != nb:
                raise ValueError(f"Custom order must have length {nb}. Got len={len(order)}.")
            min_v, max_v = min(order), max(order)
            # Detect base and normalize to 0-based
            if 0 <= min_v and max_v < nb:
                return list(order)
            if 1 <= min_v and max_v <= nb:
                return [i - 1 for i in order]
            raise ValueError(
                f"Custom order values must be either 0..{nb-1} (0-based) or 1..{nb} (1-based). Got {order}."
            )

        raise TypeError("Order must be None, int, or a sequence of ints")

    def _load_batch_by_pointer(self, ptr: int):
        """
        Loads a batch by pointer index with cumulative logic if enabled.
        Returns:
            (x, y) where
                x: np.ndarray uint8 [N, 128, 128, 3]
                y: np.ndarray int32  [N]
        """
        if self.cumul:
            # Accumulate indices/labels from batch 0..ptr inclusive
            idx_list = []
            for k in range(ptr + 1):
                idx_list += self._train_batches_idx[k]
        else:
            idx_list = self._train_batches_idx[ptr]

        if self.preload:
            x = np.take(self.x, idx_list, axis=0)
        else:
            # Construct absolute paths
            train_paths = [os.path.join(self.root, self.paths[idx]) for idx in idx_list]
            x = self.get_batch_from_paths(train_paths).astype(np.uint8)

        if self.cumul:
            y_list = []
            for k in range(ptr + 1):
                y_list += self._train_batches_y[k]
            y = np.asarray(y_list, dtype=np.int32)
        else:
            y = np.asarray(self._train_batches_y[ptr], dtype=np.int32)

        return x, y

    # Image loader
    @staticmethod
    def get_batch_from_paths(
        paths: List[str],
        compress: bool = False,
        snap_dir: str = "",
        on_the_fly: bool = True,
        verbose: bool = False,
    ):
        """
        Given a list of absolute paths returns a uint8 numpy array [N, 128, 128, 3] of images.
        Optionally caches to disk when on_the_fly=False.
        """
        log = logging.getLogger("core50_loader")
        num_imgs = len(paths)
        hexdigest = md5("".join(paths).encode("utf-8")).hexdigest()
        log.debug("Paths Hex: %s", hexdigest)

        loaded = False
        x = None
        file_path = None

        if compress:
            file_path = snap_dir + hexdigest + ".npz"
            if os.path.exists(file_path) and not on_the_fly:
                loaded = True
                with open(file_path, "rb") as f:
                    npzfile = np.load(f)
                    x = npzfile["x"]
        else:
            x_file_path = snap_dir + hexdigest + "_x.bin"
            if os.path.exists(x_file_path) and not on_the_fly:
                loaded = True
                with open(x_file_path, "rb") as f:
                    x = np.fromfile(f, dtype=np.uint8).reshape(num_imgs, 128, 128, 3)

        if not loaded:
            x = np.zeros((num_imgs, 128, 128, 3), dtype=np.uint8)
            for i, path in enumerate(paths):
                if verbose:
                    print("\r{} processed: {}/{}".format(path, i + 1, num_imgs), end="")
                x[i] = np.array(Image.open(path))
            if verbose:
                print()

            if not on_the_fly:
                if compress and file_path is not None:
                    with open(file_path, "wb") as g:
                        np.savez_compressed(g, x=x)
                else:
                    # Save raw binary for fast reload
                    (snap_dir and os.path.isdir(snap_dir)) or (snap_dir and os.makedirs(snap_dir, exist_ok=True))
                    x.tofile(snap_dir + hexdigest + "_x.bin")

        assert x is not None, "Problems loading data. x is None!"
        return x
