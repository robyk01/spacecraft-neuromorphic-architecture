from __future__ import annotations

import random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .config import DatasetConfig

import os

# 8-class list
eurosat_8_classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "PermanentCrop",
    "Residential",
    "River"
]

label_mapping = tf.constant([0, 1, 2, 3, 4, 2, 5, 6, 7, -1])

def remap_and_filter_split(dataset: tf.data.Dataset) -> tf.data.Dataset:
    def _remap(image, label):
        new_label = tf.gather(label_mapping, tf.cast(label, tf.int32))
        return image, new_label

    dataset = dataset.map(_remap, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.filter(lambda image, label: label >= 0)

    return dataset

# Noise injection functions
def tf_gaussian_noise(image, mean=0, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=tf.float32)
    tf_image = image + noise
    return tf.clip_by_value(tf_image, clip_value_min=0.0, clip_value_max=1.0)

def tf_rayleigh(image, intensity=0.0):
    img = tf.image.adjust_contrast(image, contrast_factor=0.75)

    haze_color = tf.constant([150.0 / 255.0, 180.0 / 255.0, 220.0 / 255.0], dtype=tf.float32)

    blended = (1.0 - intensity) * img + intensity * haze_color
    return tf.clip_by_value(blended, 0.0, 1.0)

def tf_jitter(image: tf.Tensor, max_brightness: float = 0.15, lower_contrast: float = 0.8, upper_contrast: float = 1.2) -> tf.Tensor:
    img = tf.image.random_brightness(image, max_delta=max_brightness)
    img = tf.image.random_contrast(img, lower=lower_contrast, upper=upper_contrast)
    return tf.clip_by_value(img, 0.0, 1.0)

def tf_sun_glare(image: tf.Tensor, max_glare: float = 0.6) -> tf.Tensor:
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    
    solar_color = tf.constant([255.0 / 255.0, 245.0 / 255.0, 235.0 / 255.0], dtype=tf.float32)
    
    gradient = tf.linspace(max_glare, 0.0, w)
    mask = tf.tile(tf.expand_dims(gradient, 0), [h, 1])  # (H, W)
    mask = tf.expand_dims(mask, -1)                      # (H, W, 1)
    
    if tf.random.uniform([]) > 0.5:
        mask = tf.image.flip_left_right(mask)
        
    blended = (1.0 - mask) * image + mask * solar_color
    return tf.clip_by_value(blended, 0.0, 1.0)

def tf_motion_blur(image: tf.Tensor, kernel_size: int = 5) -> tf.Tensor:
    k_center = kernel_size // 2
    
    kernel_1d = tf.zeros((kernel_size, kernel_size), dtype=tf.float32)
    indices = [[k_center, i] for i in range(kernel_size)]
    updates = tf.ones((kernel_size,), dtype=tf.float32) / float(kernel_size)
    kernel_2d = tf.tensor_scatter_nd_update(kernel_1d, indices, updates)
    
    kernel_4d = tf.expand_dims(tf.expand_dims(kernel_2d, -1), -1)
    kernel_4d = tf.tile(kernel_4d, [1, 1, 3, 1])
    
    img_4d = tf.expand_dims(image, 0)
    blurred = tf.nn.depthwise_conv2d(img_4d, kernel_4d, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(blurred, axis=0)

def tf_defocus_blur(image: tf.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> tf.Tensor:
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    g = tf.exp(-(x ** 2) / (2.0 * sigma ** 2))
    g = g / tf.reduce_sum(g)
    
    g_2d = tf.tensordot(g, g, axes=0)
    
    kernel_4d = tf.expand_dims(tf.expand_dims(g_2d, -1), -1)
    kernel_4d = tf.tile(kernel_4d, [1, 1, 3, 1])
    
    img_4d = tf.expand_dims(image, 0)
    blurred = tf.nn.depthwise_conv2d(img_4d, kernel_4d, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(blurred, axis=0)

def tf_clouds(image: tf.Tensor, thickness: float = 0.5, coverage: float = 1.0) -> tf.Tensor:
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    
    low_res = tf.random.uniform((1, 4, 4, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
    
    cloud_mask = tf.image.resize(low_res, [h, w], method="bicubic")
    cloud_mask = tf.squeeze(cloud_mask, axis=0)  # (H, W, 1)
    
    cloud_mask = tf.clip_by_value(cloud_mask * coverage * thickness, 0.0, 1.0)
    
    white_layer = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
    blended = (1.0 - cloud_mask) * image + cloud_mask * white_layer
    return tf.clip_by_value(blended, 0.0, 1.0)

def apply_data_augmentation(image_float: tf.Tensor) -> tf.Tensor:
    """
    Applies realistic independent effects.
    """

    # Optical and atmospheric disturbances that affect incoming light before hitting the lens
    optics_prob = tf.random.uniform([])
    if optics_prob < 0.25:
        # Rayleigh haze
        image_float = tf_rayleigh(image_float, intensity=tf.random.uniform([], 0.15, 0.35))

    elif optics_prob < 0.40:
        # Sun glare
        image_float = tf_sun_glare(image_float, max_glare=tf.random.uniform([], 0.3, 0.5))

    elif optics_prob < 0.55:
        # Light clouds
        image_float = tf_clouds(image_float, thickness=tf.random.uniform([], 0.2, 0.4), coverage=1.0)


    # Dynamics / Lens focus which happen during image exposure and sensor capture
    blur_prob = tf.random.uniform([])
    if blur_prob < 0.20:
        # Motion blur
        image_float = tf_motion_blur(image_float, kernel_size=5)

    elif blur_prob < 0.35:
        # Defocus blur
        image_float = tf_defocus_blur(image_float, kernel_size=5, sigma=tf.random.uniform([], 1.0, 1.8))


    # Electrical / Sensor noise which happen at readout and radiation hits
    if tf.random.uniform([]) < 0.30:
        # Gaussian noise
        image_float = tf_gaussian_noise(image_float, stddev=tf.random.uniform([], 0.03, 0.08))


    return image_float

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

    ds_train = remap_and_filter_split(ds_train)
    ds_val = remap_and_filter_split(ds_val)
    ds_test = remap_and_filter_split(ds_test)

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
        
        img_f = apply_data_augmentation(img_f)

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