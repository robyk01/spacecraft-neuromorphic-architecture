from __future__ import annotations

import random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .config import DatasetConfig

import os

def set_seeds(seed: int) -> None:
    """
    Set the random seed for Python, NumPy, and TensorFlow.

    This improves reproducibility between training runs.
    Exact reproducibility is not guaranteed on every platform or device.
    """
    print(f"[set_seeds] Setting global seed = {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow
    tf.keras.utils.set_random_seed(seed)

    # Try enabling deterministic ops (supported in newer TF builds)
    try:
        tf.config.experimental.enable_op_determinism()
        print("[set_seeds] Enabled TensorFlow op determinism.")
    except Exception as e:
        print(f"[set_seeds] Could not enable op determinism (non-fatal). Reason: {e}")
        
        
def load_eurosat_tfds(
    cfg: DatasetConfig,
) -> tuple[dict[str, tf.data.Dataset], tfds.core.DatasetInfo]:
    """
    Load EuroSAT from TensorFlow Datasets and create train, validation,
    and test splits.

    EuroSAT exposes one main 'train' split through TFDS. We divide that
    split according to cfg.train_pct, cfg.val_pct, and cfg.test_pct.

    Returns:
        datasets:
            Dictionary containing the raw, unbatched datasets:
            - datasets["train"]
            - datasets["val"]
            - datasets["test"]

        dataset_info:
            TFDS metadata containing class names, image shape, and the
            number of classes.
    """
    split_total = cfg.train_pct + cfg.val_pct + cfg.test_pct

    if split_total != 100:
        raise ValueError(
            "Dataset split percentages must add up to 100. "
            f"Current total: {split_total}"
        )

    if cfg.train_pct <= 0:
        raise ValueError("train_pct must be greater than 0.")

    if cfg.val_pct <= 0:
        raise ValueError("val_pct must be greater than 0.")

    if cfg.test_pct <= 0:
        raise ValueError("test_pct must be greater than 0.")
    
    
    print(f"[load_eurosat_tfds] Loading dataset '{cfg.tfds_name}' from TFDS...")

    # TFDS EuroSAT typically exposes only a 'train' split, so we slice it.
    train_split = f"train[:{cfg.train_pct}%]"
    val_split = f"train[{cfg.train_pct}%:{cfg.train_pct + cfg.val_pct}%]"
    test_split = f"train[{cfg.train_pct + cfg.val_pct}%:]"

    print("[load_eurosat_tfds] Using splits:")
    print(f"  - train: {train_split}")
    print(f"  - val:   {val_split}")
    print(f"  - test:  {test_split}")
    
    data_dir = str(cfg.data_dir) if cfg.data_dir is not None else None

    if cfg.tfds_name == "eurosat/rgb":
        builder = tfds.builder(cfg.tfds_name, data_dir=data_dir)

        # DFKI URL is currently returning 403. Use the official Zenodo RGB zip instead.
        builder.builder_config.download_url = (
            "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
        )
        
        # Critical: Zenodo's RGB zip does not use TFDS's original "2750" folder name.
        builder.builder_config.subdir = "EuroSAT_RGB"

        builder.download_and_prepare()

        ds_train, ds_val, ds_test = builder.as_dataset(
            split=[train_split, val_split, test_split],
            as_supervised=True,
            shuffle_files=False,
        )

        ds_info = builder.info
    else:
        # as_supervised=True returns (image, label)
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            cfg.tfds_name,
            split=[train_split, val_split, test_split],
            as_supervised=True,
            with_info=True,
            data_dir=data_dir,
            shuffle_files=False,  # deterministic file order; we control randomness in pipeline
        )
    
    image_dtype = ds_info.features["image"].np_dtype
    label_dtype = ds_info.features["label"].np_dtype
    print("[load_eurosat_tfds] Feature dtypes:")
    print(f"  - image dtype: {image_dtype}")
    print(f"  - label dtype: {label_dtype}")

    # Basic dataset metadata
    num_classes = ds_info.features["label"].num_classes
    class_names = ds_info.features["label"].names
    image_shape = ds_info.features["image"].shape

    print("[load_eurosat_tfds] Dataset info:")
    print(f"  - num_classes: {num_classes}")
    print(f"  - image_shape: {image_shape}")
    print(f"  - classes: {class_names}")

    datasets = {"train": ds_train, "val": ds_val, "test": ds_test}
    return datasets, ds_info

def build_tfdata_pipelines(
    datasets: dict[str, tf.data.Dataset],
    cfg: DatasetConfig,
) -> dict[str, tf.data.Dataset]:
    """
    Build the TensorFlow data pipelines used for training and evaluation.

    Image format:
        - dtype: uint8
        - range: [0, 255]

    Normalization is not performed here. The Keras model contains a
    Rescaling layer that converts the input to floating point and scales
    it to [0, 1].

    Training pipeline:
        preprocess
        cache
        shuffle
        augment
        batch
        prefetch

    Validation and test pipelines:
        preprocess
        cache
        batch
        prefetch
    """
    
    required_splits = {"train", "val", "test"}
    missing_splits = required_splits - datasets.keys()

    if missing_splits:
        raise ValueError(
            "The datasets dictionary is missing these splits: "
            f"{sorted(missing_splits)}"
        )

    autotune = tf.data.AUTOTUNE

    def preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # TFDS EuroSAT/RGB provides uint8 images already. Keep that contract.
        image = tf.cast(image, tf.uint8)

        if cfg.resize_to is not None:
            # tf.image.resize produces float; convert back to uint8.
            img_f = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
            img_f = tf.image.resize(img_f, cfg.resize_to, method="bilinear")
            image = tf.image.convert_image_dtype(img_f, tf.uint8, saturate=True)

        return image, label  # label stays int64

    def augment_train(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # Geometric augmentations: flips
        image = tf.image.random_flip_left_right(image, seed=cfg.seed)
        image = tf.image.random_flip_up_down(image, seed=cfg.seed)

        # 90-degree rotations (simple + robust for land-cover)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=cfg.seed)
        image = tf.image.rot90(image, k=k)

        # Photometric augmentations MUST be done in float
        img_f = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
        img_f = tf.image.random_brightness(img_f, max_delta=0.08, seed=cfg.seed)
        img_f = tf.image.random_contrast(img_f, lower=0.9, upper=1.1, seed=cfg.seed)
        
        # Ensure range still [0,1] after jitter
        img_f = tf.clip_by_value(img_f, 0.0, 1.0)

        # Convert back to uint8 [0..255]
        image = tf.image.convert_image_dtype(img_f, tf.uint8, saturate=True)
        return image, label

    def finalize_pipeline(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
        """
        Apply the appropriate transformations to one dataset split.
        """
        dataset = dataset.map(preprocess, num_parallel_calls=autotune)
        
        # Cache before augmentation so random augmentation can still change
        # between epochs.
        if cfg.cache:
            dataset = dataset.cache()
        
        if training:
            dataset = dataset.shuffle(
                buffer_size=cfg.shuffle_buffer,
                seed=cfg.seed,
                reshuffle_each_iteration=True,
            )

            if cfg.augment:
                dataset = dataset.map(
                    augment_train,
                    num_parallel_calls=autotune,
                )

        dataset = dataset.batch(cfg.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(autotune)
        return dataset

    print("[build_tfdata_pipelines] Building train pipeline...")
    ds_train = finalize_pipeline(
        datasets["train"],
        training=True,
    )
    ds_val = finalize_pipeline(
        datasets["val"],
        training=False,
    )
    ds_test = finalize_pipeline(
        datasets["test"],
        training=False,
    )

    # Print one batch sanity check
    # Check one batch so errors are found before training begins.
    sample_images, sample_labels = next(iter(ds_train.take(1)))

    print("[build_tfdata_pipelines] Train batch check:")
    print(
        f"  image shape: {sample_images.shape}, "
        f"dtype: {sample_images.dtype}"
    )
    print(
        f"  image range: "
        f"{tf.reduce_min(sample_images).numpy()} to "
        f"{tf.reduce_max(sample_images).numpy()}"
    )
    print(
        f"  label shape: {sample_labels.shape}, "
        f"dtype: {sample_labels.dtype}"
    )


    return {"train": ds_train, "val": ds_val, "test": ds_test}