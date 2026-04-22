from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data_processing import DataProcessing


@dataclass(frozen=True)
class UnifiedDataSplit:
    features: torch.Tensor
    targets: torch.Tensor
    train_features: torch.Tensor
    test_features: torch.Tensor
    train_targets: torch.Tensor
    test_targets: torch.Tensor
    train_indices: torch.Tensor
    test_indices: torch.Tensor
    label_names: list[str]
    data_path: Path
    seed: int
    train_ratio: float


def build_shared_split(
    data_path: Path | None = None,
    train_ratio: float = 0.7,
    seed: int = 42,
    stratify: bool = True,
) -> UnifiedDataSplit:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be strictly between 0 and 1.")

    data_handler = DataProcessing()
    if data_path is not None:
        data_handler.excel_path = data_path

    features, targets = data_handler.process_data(data_handler.excel_path)
    targets = targets.squeeze().to(torch.int64)

    n_rows = int(features.shape[0])
    if n_rows == 0:
        raise ValueError("No rows were loaded from the data source.")

    n_train = int(n_rows * train_ratio)
    if n_train <= 0 or n_train >= n_rows:
        raise ValueError(
            f"The computed train split size is invalid: {n_train} for n_rows={n_rows}."
        )

    generator = torch.Generator().manual_seed(seed)

    if stratify:
        train_chunks: list[torch.Tensor] = []
        test_chunks: list[torch.Tensor] = []
        class_ids = torch.unique(targets).tolist()

        for class_id in class_ids:
            class_indices = torch.where(targets == int(class_id))[0]
            class_size = int(class_indices.shape[0])
            class_perm = class_indices[torch.randperm(class_size, generator=generator)]

            if class_size <= 1:
                # Single-sample classes cannot be split across train/test.
                class_train = class_perm
                class_test = class_perm.new_empty((0,), dtype=torch.long)
            else:
                class_train_size = int(class_size * train_ratio)
                class_train_size = min(max(class_train_size, 1), class_size - 1)
                class_train = class_perm[:class_train_size]
                class_test = class_perm[class_train_size:]

            train_chunks.append(class_train)
            test_chunks.append(class_test)

        train_indices = torch.cat(train_chunks)
        test_indices = torch.cat(test_chunks)
        train_indices = train_indices[torch.randperm(len(train_indices), generator=generator)]
        test_indices = test_indices[torch.randperm(len(test_indices), generator=generator)]
    else:
        permutation = torch.randperm(n_rows, generator=generator)
        train_indices = permutation[:n_train]
        test_indices = permutation[n_train:]

    train_features = features[train_indices]
    test_features = features[test_indices]
    train_targets = targets[train_indices]
    test_targets = targets[test_indices]

    target_map = data_handler.mapping.get(data_handler.features.target_column, {})
    label_names = [str(name) for name, _ in sorted(target_map.items(), key=lambda item: item[1])]
    if not label_names:
        label_names = [str(idx) for idx in sorted(np.unique(targets.numpy()).tolist())]

    # Split integrity checks to guarantee deterministic and exhaustive partitions.
    train_set = set(train_indices.tolist())
    test_set = set(test_indices.tolist())
    if train_set.intersection(test_set):
        raise RuntimeError("Train and test indices overlap.")
    if len(train_set) + len(test_set) != n_rows:
        raise RuntimeError("Train and test indices do not cover all rows exactly once.")

    return UnifiedDataSplit(
        features=features,
        targets=targets,
        train_features=train_features,
        test_features=test_features,
        train_targets=train_targets,
        test_targets=test_targets,
        train_indices=train_indices,
        test_indices=test_indices,
        label_names=label_names,
        data_path=data_handler.excel_path,
        seed=seed,
        train_ratio=train_ratio,
    )
