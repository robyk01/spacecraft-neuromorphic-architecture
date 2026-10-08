"""
Step 1: Reproducibility + TFDS EuroSAT load + tf.data pipelines

- set_seeds(seed)
- load_eurosat_tfds(...)
- build_tfdata_pipelines(...)
- main()

Notes:
- Uses TFDS "eurosat/rgb" (images are typically 64x64x3, labels are integer class ids).
- Creates our own splits from the single TFDS "train" split.
"""

from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations for more deterministic behavior
# os.environ["TF_USE_LEGACY_KERAS"] = "1" # Use legacy Keras behavior (if needed)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warnings

from dataclasses import dataclass
from typing import Optional, Tuple,Any, List, Dict

from pathlib import Path
project_root = Path.cwd().resolve().parents[0]

import random
import numpy as np
import time
import matplotlib.pyplot as plt

import tensorflow as tf
# tf.get_logger().setLevel("ERROR")  # reduce TF python-level warnings
import tensorflow_datasets as tfds
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import convert, set_akida_version, AkidaVersion, get_akida_version
from akida import devices


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class PipelineConfig:
    seed: int = 42

    # TFDS dataset name for EuroSAT RGB
    tfds_name: str = "eurosat/rgb"

    # Split ratios (from the TFDS 'train' split)
    # Example: 70/15/15 split
    train_pct: int = 70
    val_pct: int = 15
    test_pct: int = 15

    # Input sizing
    # EuroSAT RGB is commonly 64x64 already; keep as-is unless you explicitly change it.
    resize_to: Optional[Tuple[int, int]] = None  # e.g. (128, 128) or None for native size

    # Training pipeline params
    batch_size: int = 64
    shuffle_buffer: int = 10_000
    cache: bool = True
    augment: bool = True

    # Dataset download directory (optional)
    data_dir: Optional[str] = None
    
@dataclass
class TrainConfig:
    epochs: int = 2
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # "adam" or "sgd"
    label_smoothing: float = 0.0

    # Callbacks / logging
    checkpoint_dir: str = project_root / "checkpoints"
    run_name: str = "eurosat_cnn_baseline"
    early_stop_patience: int = 2
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

    use_tensorboard: bool = False
    tensorboard_dir: str = project_root / "logs"
    
@dataclass
class QuantizeConfig:
    # Quantization scheme
    input_weight_bits: int = 8
    weight_bits: int = 8
    activation_bits: int = 8

    # Calibration (BrainChip recommended defaults)
    num_samples: int = 1024
    batch_size: int = 100
    epochs: int = 2

    # Data preprocessing must match what the model expects
    resize_to: Optional[Tuple[int, int]] = None  # must match your pipeline if used
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # "adam" or "sgd"
    
@dataclass
class AkidaConvertConfig:
    # Where to save the converted Akida model (".fbz" will be added if missing).
    save_path: Optional[str] = None

    # Target Akida version context for conversion ("v2" default).
    # CNN2SNN supports setting this context via set_akida_version. :contentReference[oaicite:1]{index=1}
    akida_version: str = "v1"  # "v1" or "v2"

    # Optional: map the converted model to a device (if present).
    map_to_device: bool = False
    hw_only: bool = True  # map only hardware-compatible sequences (if mapping)



# -----------------------------
# 1) Reproducibility
# -----------------------------

def set_seeds(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and TensorFlow.
    Attempts to enable deterministic TF ops when available.
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


# -----------------------------
# 2) Load EuroSAT from TFDS
# -----------------------------

def load_eurosat_tfds(
    tfds_name: str,
    train_pct: int,
    val_pct: int,
    test_pct: int,
    data_dir: Optional[str] = None,
) -> Tuple[Dict[str, tf.data.Dataset], tfds.core.DatasetInfo]:
    """
    Load EuroSAT from TFDS and create deterministic splits from the 'train' split.

    Returns:
        datasets: dict with keys {"train", "val", "test"}
        ds_info: TFDS dataset info (contains class names, feature shapes, etc.)
    """
    print(f"[load_eurosat_tfds] Loading dataset '{tfds_name}' from TFDS...")
    assert train_pct + val_pct + test_pct == 100, "Split percentages must sum to 100."

    # TFDS EuroSAT typically exposes only a 'train' split, so we slice it.
    train_split = f"train[:{train_pct}%]"
    val_split = f"train[{train_pct}%:{train_pct + val_pct}%]"
    test_split = f"train[{train_pct + val_pct}%:]"

    print("[load_eurosat_tfds] Using splits:")
    print(f"  - train: {train_split}")
    print(f"  - val:   {val_split}")
    print(f"  - test:  {test_split}")

    # skips file validation for testing
    # dl_config = tfds.download.DownloadConfig(register_checksums=True)

    # as_supervised=True returns (image, label)
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        tfds_name,
        split=[train_split, val_split, test_split],
        as_supervised=True,
        with_info=True,
        data_dir=data_dir,
        shuffle_files=False,  # deterministic file order; we control randomness in pipeline
        # download_and_prepare_kwargs={"download_config": dl_config}
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


# -----------------------------
# 3) Build tf.data pipelines
# -----------------------------

def build_tfdata_pipelines(
    datasets: Dict[str, tf.data.Dataset],
    batch_size: int,
    seed: int,
    resize_to: Optional[Tuple[int, int]] = None,
    shuffle_buffer: int = 10_000,
    cache: bool = True,
    augment: bool = True,
) -> Dict[str, tf.data.Dataset]:
    """
    Build efficient tf.data pipelines:
      - preprocess: float32, scale to [0, 1], optional resize
      - augment (train only): flips + small photometric jitter (kept simple)
      - shuffle (train only), cache, batch, prefetch

    Labels remain integer class ids (good for SparseCategoricalCrossentropy).
    """

    autotune = tf.data.AUTOTUNE

    def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # TFDS EuroSAT/RGB provides uint8 images already. Keep that contract.
        image = tf.cast(image, tf.uint8)

        if resize_to is not None:
            # tf.image.resize produces float; convert back to uint8.
            img_f = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
            img_f = tf.image.resize(img_f, resize_to, method="bilinear")
            image = tf.image.convert_image_dtype(img_f, tf.uint8, saturate=True)

        return image, label  # label stays int64

    def augment_train(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Geometric augmentations: flips
        image = tf.image.random_flip_left_right(image, seed=seed)
        image = tf.image.random_flip_up_down(image, seed=seed)

        # 90-degree rotations (simple + robust for land-cover)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)

        # Photometric augmentations MUST be done in float
        img_f = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
        img_f = tf.image.random_brightness(img_f, max_delta=0.08, seed=seed)
        img_f = tf.image.random_contrast(img_f, lower=0.9, upper=1.1, seed=seed)
        
        # Ensure range still [0,1] after jitter
        img_f = tf.clip_by_value(img_f, 0.0, 1.0)

        # Convert back to uint8 [0..255]
        image = tf.image.convert_image_dtype(img_f, tf.uint8, saturate=True)
        return image, label

    def finalize(ds: tf.data.Dataset, training: bool) -> tf.data.Dataset:
        ds = ds.map(preprocess, num_parallel_calls=autotune)

        if training and augment:
            ds = ds.map(augment_train, num_parallel_calls=autotune)

        if cache:
            ds = ds.cache()

        if training:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(autotune)
        return ds

    print("[build_tfdata_pipelines] Building pipelines...")
    ds_train = finalize(datasets["train"], training=True)
    ds_val = finalize(datasets["val"], training=False)
    ds_test = finalize(datasets["test"], training=False)

    # Print one batch sanity check
    sanity_ds = datasets["train"].map(preprocess, num_parallel_calls=autotune).batch(batch_size).take(1)
    sample_images, sample_labels = next(iter(sanity_ds))
    print("[build_tfdata_pipelines] Sanity check (one train batch):")
    print(f"  - images shape: {sample_images.shape}, dtype={sample_images.dtype}, "
          f"min={tf.reduce_min(sample_images).numpy()}, max={tf.reduce_max(sample_images).numpy()}")
    print(f"  - labels shape: {sample_labels.shape}, dtype={sample_labels.dtype}, "
          f"labels[0:10]={sample_labels.numpy()[:10]}")

    return {"train": ds_train, "val": ds_val, "test": ds_test}

# -----------------------------
# 4) Build baseline CNN model
# -----------------------------

def build_baseline_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    base_filters: int = 32,
    dense_units: int = 128,
    weight_decay: float = 1e-4,
) -> tf.keras.Model:
    """
    Build a compact CNN baseline suitable for:
      - EuroSAT RGB (64x64x3)
      - later quantization with QuantizeML
      - later conversion to Akida SNN with cnn2snn.convert

    Design choices for Akida/quantization friendliness:
      - DepthwiseConv2D + 1x1 Conv2D blocks (efficient and common in Akida examples)
      - ReLU as explicit layers
      - Downsampling via strided convolutions (avoids MaxPool ordering constraints)
      - No Dropout / no exotic activations
      - Output logits (Dense(num_classes) without softmax)
    """
    print("[build_baseline_cnn] Building baseline CNN...")
    print(f"[build_baseline_cnn] input_shape={input_shape}, num_classes={num_classes}")

    l2 = tf.keras.regularizers.l2(weight_decay)

    inputs = tf.keras.Input(shape=input_shape, dtype=tf.uint8, name="input")

    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_u8_to_f32")(inputs)

    # Stem: strided conv to reduce spatial size early (64 -> 32)
    x = tf.keras.layers.Conv2D(
        filters=base_filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2,
        name="stem_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    def dw_pw_block(x, filters: int, stride: int, block_name: str):
        # Depthwise conv
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            depthwise_regularizer=l2,
            name=f"{block_name}_dw",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{block_name}_dw_bn")(x)
        x = tf.keras.layers.ReLU(name=f"{block_name}_dw_relu")(x)

        # Pointwise (1x1) conv
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2,
            name=f"{block_name}_pw",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{block_name}_pw_bn")(x)
        x = tf.keras.layers.ReLU(name=f"{block_name}_pw_relu")(x)
        return x

    # Blocks: progressively downsample (32 -> 16 -> 8) and increase channels
    x = dw_pw_block(x, filters=base_filters * 2, stride=1, block_name="b1")  # 32x32
    x = dw_pw_block(x, filters=base_filters * 2, stride=2, block_name="b2")  # 16x16

    x = dw_pw_block(x, filters=base_filters * 4, stride=1, block_name="b3")  # 16x16
    x = dw_pw_block(x, filters=base_filters * 4, stride=2, block_name="b4")  # 8x8

    x = dw_pw_block(x, filters=base_filters * 8, stride=1, block_name="b5")  # 8x8

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(
        dense_units,
        use_bias=False,
        kernel_regularizer=l2,
        name="fc1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="fc1_bn")(x)
    x = tf.keras.layers.ReLU(name="fc1_relu")(x)

    outputs = tf.keras.layers.Dense(num_classes, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="eurosat_baseline_cnn")

    print("[build_baseline_cnn] Model built.")
    model.summary()
    return model

# -----------------------------
# 5) Train CNN model
# -----------------------------

def _safe_cardinality(ds: tf.data.Dataset) -> Optional[int]:
    """
    Return dataset cardinality if known, else None.
    """
    card = tf.data.experimental.cardinality(ds)
    if card == tf.data.INFINITE_CARDINALITY or card == tf.data.UNKNOWN_CARDINALITY:
        return None
    try:
        return int(card.numpy())
    except Exception:
        return None


def train_cnn_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: TrainConfig,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, str]:
    """
    Compile and train the baseline cnn (non-quantized) TF-Keras model.

    Key choices for later quantization / cnn2snn:
      - Model outputs logits (no softmax layer in model).
      - Loss uses from_logits=True.
      - Keeps training simple and stable for a clean baseline.

    Returns:
      model: trained model (weights restored to best val accuracy by EarlyStopping)
      history: Keras History
      best_ckpt_path: path to the best checkpoint file saved during training
    """
    print("\n[train_cnn_model] Starting CNN training...")
    print(f"[train_cnn_model] run_name={cfg.run_name}")
    print(f"[train_cnn_model] epochs={cfg.epochs}, lr={cfg.learning_rate}, optimizer={cfg.optimizer}")

    train_batches = _safe_cardinality(train_ds)
    val_batches = _safe_cardinality(val_ds)
    print(f"[train_cnn_model] train batches: {train_batches if train_batches is not None else 'unknown'}")
    print(f"[train_cnn_model] val batches:   {val_batches if val_batches is not None else 'unknown'}")

    # Optimizer
    opt_name = cfg.optimizer.lower().strip()
    if opt_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    elif opt_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'. Use 'adam' or 'sgd'.")

    # Loss & metrics (logits output)
    try:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            label_smoothing=cfg.label_smoothing,
        )
    except TypeError:
        print("[train_cnn_model] WARNING: label_smoothing not supported in this TF/Keras build. Using label_smoothing=0.0.")
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    print("[train_cnn_model] Model compiled.")

    # Callbacks
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}_best.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,   # keep full model for convenience
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
    ]

    if cfg.use_tensorboard:
        log_dir = os.path.join(cfg.tensorboard_dir, cfg.run_name, time.strftime("%Y%m%d-%H%M%S"))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
        print(f"[train_cnn_model] TensorBoard enabled: {log_dir}")

    # Train
    print("[train_cnn_model] Fitting...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Final eval (on val, since test is for later)
    print("\n[train_cnn_model] Training done. Evaluating best-restored model on val set...")
    val_metrics = model.evaluate(val_ds, verbose=1)
    print("[train_cnn_model] Val metrics:", dict(zip(model.metrics_names, val_metrics)))
    print(f"[train_cnn_model] Best checkpoint saved at: {best_ckpt_path}")

    return model, history, best_ckpt_path

# -----------------------------
# 6) Evaluate model
# -----------------------------

def _prepare_dataset_for_model_input(
    ds: tf.data.Dataset,
    model: tf.keras.Model,
) -> tf.data.Dataset:
    """
    Ensure dataset image dtype matches model input dtype.

    With the new direction, the dataset yields uint8 [0..255].
    Some QuantizeML models may still declare float32 input; in that case we cast to float32
    while keeping the numeric range [0..255] (Rescaling in-model will handle normalization).
    """
    expected = model.inputs[0].dtype  # e.g. tf.uint8 or tf.float32

    def _cast(image, label):
        if image.dtype != expected:
            image = tf.cast(image, expected)
        return image, label

    # Only map if needed (cheap check)
    sample_spec = ds.element_spec[0]
    if sample_spec.dtype == expected:
        return ds

    print(f"[prepare_dataset_for_model_input] Casting images: {sample_spec.dtype} -> {expected}")
    return ds.map(_cast, num_parallel_calls=tf.data.AUTOTUNE)

def evaluate_model(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    class_names: List[str],
    split_name: str = "test",
    from_logits: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a TF-Keras classifier on a tf.data.Dataset and print:
      - model.evaluate() metrics
      - confusion matrix
      - per-class precision/recall/F1
      - overall accuracy from predictions

    Works with uint8 pipelines by adapting the dataset dtype to the model input dtype.
    """
    num_classes = len(class_names)
    print(f"\n[evaluate_model] Evaluating on split='{split_name}' (num_classes={num_classes}) ...")
    
    # Ensure dtype matches model expectation (important for quantized models)
    ds_eval = _prepare_dataset_for_model_input(ds, model)

    # 1) Keras evaluation (loss/metrics)
    print("[evaluate_model] Running model.evaluate() ...")
    try:
        keras_metrics = model.evaluate(ds_eval, verbose=1, return_dict=True)
    except TypeError:
        keras_vals = model.evaluate(ds_eval, verbose=1)
        keras_metrics = dict(zip(model.metrics_names, keras_vals))
    print("[evaluate_model] Keras metrics:", keras_metrics)

    # 2) Manual predictions for confusion matrix + detailed metrics
    print("[evaluate_model] Collecting predictions for confusion matrix / per-class metrics ...")
    y_true_all = []
    y_pred_all = []

    for b, (x, y_true) in enumerate(ds_eval):
        if max_batches is not None and b >= max_batches:
            print(f"[evaluate_model] max_batches reached ({max_batches}). Stopping early.")
            break

        out = model(x, training=False)
        if from_logits:
            out = tf.nn.softmax(out, axis=-1)

        y_pred = tf.argmax(out, axis=-1, output_type=y_true.dtype)

        y_true_all.append(y_true.numpy())
        y_pred_all.append(y_pred.numpy())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    overall_acc = float((y_true_all == y_pred_all).mean())
    print(f"[evaluate_model] Overall accuracy (from predictions): {overall_acc:.4f}")

    cm = tf.math.confusion_matrix(
        y_true_all,
        y_pred_all,
        num_classes=num_classes,
        dtype=tf.int32,
    ).numpy()

    print("\n[evaluate_model] Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Per-class metrics
    eps = 1e-9
    per_class = []
    for i, name in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        support = float(cm[i, :].sum())
        class_acc = tp / (support + eps)

        per_class.append({
            "class_id": i,
            "class_name": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc": class_acc,
            "support": support,
        })

    macro_precision = float(np.mean([d["precision"] for d in per_class]))
    macro_recall = float(np.mean([d["recall"] for d in per_class]))
    macro_f1 = float(np.mean([d["f1"] for d in per_class]))

    print("\n[evaluate_model] Per-class metrics:")
    print("  {:>2}  {:<22}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}".format(
        "id", "class", "prec", "recall", "f1", "acc", "support"
    ))
    for d in per_class:
        print("  {:>2}  {:<22}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.0f}".format(
            d["class_id"], d["class_name"][:22],
            d["precision"], d["recall"], d["f1"], d["acc"], d["support"]
        ))

    print("\n[evaluate_model] Macro averages:")
    print(f"  - macro_precision: {macro_precision:.4f}")
    print(f"  - macro_recall:    {macro_recall:.4f}")
    print(f"  - macro_f1:        {macro_f1:.4f}")

    return {
        "keras_metrics": keras_metrics,
        "overall_acc": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
    }
    
def show_single_prediction(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    class_names: List[str],
    index: int = 0,
    top_k: int = 5,
    from_logits: bool = True,
    show_image: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Show predictions for a single image drawn from a (batched) tf.data.Dataset.

    - Unbatches the dataset, skips 'index' samples, takes 1 sample.
    - Runs inference, prints true label + top-k predictions.
    - Optionally displays and/or saves the image.

    Returns a dict with: true_id, true_name, topk_ids, topk_names, topk_probs.
    """
    assert top_k >= 1, "top_k must be >= 1"
    num_classes = len(class_names)
    print(f"\n[show_single_prediction] Fetching sample index={index} from dataset...")

    # Get one sample (image, label)
    sample_ds = ds.unbatch().skip(index).take(1)

    image, label = None, None
    for x, y in sample_ds:
        image, label = x, y

    if image is None:
        raise RuntimeError("[show_single_prediction] Could not fetch a sample from dataset.")

    # Ensure batch dimension
    x_in = tf.expand_dims(image, axis=0)

    # Forward pass
    out = model(x_in, training=False)  # shape: (1, num_classes)
    if from_logits:
        probs = tf.nn.softmax(out, axis=-1)
    else:
        probs = out

    probs = tf.squeeze(probs, axis=0)  # shape: (num_classes,)

    # Top-k
    k = min(top_k, num_classes)
    topk = tf.math.top_k(probs, k=k)
    topk_probs = topk.values.numpy()
    topk_ids = topk.indices.numpy().astype(int)

    true_id = int(label.numpy())
    true_name = class_names[true_id] if 0 <= true_id < num_classes else str(true_id)

    pred_id = int(topk_ids[0])
    pred_name = class_names[pred_id] if 0 <= pred_id < num_classes else str(pred_id)

    print("[show_single_prediction] Ground truth:")
    print(f"  - true_id={true_id}, true_class='{true_name}'")
    print("[show_single_prediction] Top predictions:")
    for rank, (cid, p) in enumerate(zip(topk_ids, topk_probs), start=1):
        cname = class_names[cid] if 0 <= cid < num_classes else str(cid)
        print(f"  {rank:>2}. {cname:<22}  p={p:.4f}  (class_id={cid})")

    # Display / save image
    if show_image or save_path is not None:
        img_np = image.numpy()
        plt.figure()
        plt.imshow(img_np)
        plt.axis("off")
        plt.title(f"True: {true_name} | Pred: {pred_name} ({topk_probs[0]:.3f})")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[show_single_prediction] Saved figure to: {save_path}")

        if show_image:
            plt.show()

        plt.close()

    return {
        "true_id": true_id,
        "true_name": true_name,
        "topk_ids": topk_ids,
        "topk_names": [class_names[i] for i in topk_ids],
        "topk_probs": topk_probs,
    }
    
# -----------------------------
# 7) Quantize + calibrate with QuantizeML
# -----------------------------
    
def _extract_calibration_samples(
    raw_train_ds: tf.data.Dataset,
    num_samples: int,
    resize_to: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Extract 'num_samples' images from the raw TFDS train dataset (no augmentation),
    applying the same preprocessing the model expects: uint8 in [0,255] + optional resize.

    Returns:
      samples: np.ndarray of shape (num_samples, H, W, C), dtype float32
    """
    print(f"[calibration] Extracting {num_samples} calibration samples (no augmentation)...")

    def preprocess_only(image: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        image = tf.cast(image, tf.uint8)

        if resize_to is not None:
            # Resize in float, then convert back to uint8.
            img_f = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
            img_f = tf.image.resize(img_f, resize_to, method="bilinear")
            image = tf.image.convert_image_dtype(img_f, tf.uint8, saturate=True)

        return image


    ds = raw_train_ds.map(preprocess_only, num_parallel_calls=tf.data.AUTOTUNE).take(num_samples)

    # Materialize into numpy
    samples_list = [img.numpy() for img in ds]
    if not samples_list:
        raise RuntimeError("[calibration] No samples extracted. Check dataset pipeline.")

    samples = np.stack(samples_list, axis=0).astype(np.uint8)  # shape: (num_samples, H, W, C)

    print("[calibration] Calibration samples ready:")
    print(f"  - shape: {samples.shape}")
    print(f"  - dtype: {samples.dtype}")
    print(f"  - min/max: {samples.min()} / {samples.max()}")

    return samples


def quantize_and_calibrate(
    model_cnn: tf.keras.Model,
    raw_train_ds: tf.data.Dataset,
    cfg: QuantizeConfig,
) -> tf.keras.Model:
    """
    Quantize + calibrate a TF-Keras model using QuantizeML.

    This follows the recommended approach of calibrating OutputQuantizers using real samples
    passed to quantize(...), with commonly used settings (1024 samples, batch_size=100, epochs=2). :contentReference[oaicite:2]{index=2}

    Returns:
      model_quantized: quantized TF-Keras model
    """
    print("\n[quantize_and_calibrate] Starting quantization + calibration...")
    print("[quantize_and_calibrate] Quantization scheme:")
    print(f"  - input_weight_bits={cfg.input_weight_bits}, weight_bits={cfg.weight_bits}, activation_bits={cfg.activation_bits}")
    print("[quantize_and_calibrate] Calibration settings:")
    print(f"  - num_samples={cfg.num_samples}, batch_size={cfg.batch_size}, epochs={cfg.epochs}")

    # Import QuantizeML with a clear error if missing
    # try:
    #     from quantizeml.models import quantize, QuantizationParams
    # except Exception as e:
    #     raise ImportError(
    #         "QuantizeML not available. Please install/configure BrainChip QuantizeML before running quantization."
    #     ) from e

    # Build qparams (8/8/8 default; 8/4/4 later if needed)
    qparams = QuantizationParams(
        input_weight_bits=cfg.input_weight_bits,
        weight_bits=cfg.weight_bits,
        activation_bits=cfg.activation_bits,
    )

    # Extract real samples from raw TFDS train split (no augmentation)
    samples = _extract_calibration_samples(
        raw_train_ds=raw_train_ds,
        num_samples=cfg.num_samples,
        resize_to=cfg.resize_to,
    )

    # Quantize + calibrate in one call
    # Passing samples triggers calibration of OutputQuantizers. :contentReference[oaicite:3]{index=3}
    try:
        model_quantized = quantize(
            model_cnn,
            qparams=qparams,
            samples=samples,
            num_samples=cfg.num_samples,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
        )
    except Exception as e:
        print("[quantize_and_calibrate] WARNING: quantize() failed with uint8 samples. Falling back to float32 [0,1].")
        print(f"[quantize_and_calibrate] Reason: {e}")
        samples_f = (samples.astype(np.float32) / 255.0)
        model_quantized = quantize(
            model_cnn,
            qparams=qparams,
            samples=samples_f,
            num_samples=cfg.num_samples,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
        )

    print("[quantize_and_calibrate] Quantization done.")
    
    # Optimizer
    opt_name = cfg.optimizer.lower().strip()
    if opt_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    elif opt_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'. Use 'adam' or 'sgd'.")

    # Loss & metrics (logits output)
    # try:
    #     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    #         from_logits=True,
    #         label_smoothing=cfg.label_smoothing,
    #     )
    # except TypeError:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model_quantized.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    
    model_quantized.summary()
    
    return model_quantized

# -----------------------------
# 8) Convert to Akida with CNN2SNN
# -----------------------------

def convert_to_akida(
    model_quantized: tf.keras.Model,
    cfg: AkidaConvertConfig,
):
    """
    Convert a QuantizeML-quantized TF-Keras model to an Akida model using CNN2SNN.

    - Uses cnn2snn.convert(model, file_path=..., input_scaling=None)
    - For QuantizeML-quantized models, input_scaling is not used by CNN2SNN. :contentReference[oaicite:2]{index=2}
    - Returns an akida.Model. :contentReference[oaicite:3]{index=3}
    """
    print("\n[convert_to_akida] Converting quantized model to Akida...")

    # # Import CNN2SNN conversion utilities
    # try:
    #     cnn2snn import confromvert, set_akida_version, AkidaVersion, get_akida_version
    # except Exception as e:
    #     raise ImportError(
    #         "cnn2snn is not available. Ensure BrainChip CNN2SNN toolkit is installed and importable."
    #     ) from e

    # Set target Akida version context (v2 is default in CNN2SNN) :contentReference[oaicite:4]{index=4}
    v = cfg.akida_version.lower().strip()
    if v == "v1":
        set_akida_version(AkidaVersion.v1)
    elif v == "v2":
        set_akida_version(AkidaVersion.v2)
    else:
        raise ValueError("AkidaConvertConfig.akida_version must be 'v1' or 'v2'.")

    print(f"[convert_to_akida] Target Akida version context: {get_akida_version()}")
    print(f"[convert_to_akida] Keras input dtype: {model_quantized.inputs[0].dtype}")

    # Convert (for QuantizeML models, input_scaling is ignored by CNN2SNN) :contentReference[oaicite:5]{index=5}
    model_akida = convert(
        model_quantized,
        file_path=cfg.save_path,   # may be None
        input_scaling=None,
    )

    print("[convert_to_akida] Conversion complete.")
    try:
        # Akida Model has summary(), input_shape, output_shape, ip_version, etc. :contentReference[oaicite:6]{index=6}
        print(f"[convert_to_akida] Akida ip_version: {model_akida.ip_version}")
        print(f"[convert_to_akida] Akida input_shape: {model_akida.input_shape}")
        print(f"[convert_to_akida] Akida output_shape: {model_akida.output_shape}")
        model_akida.summary()
    except Exception as e:
        print(f"[convert_to_akida] Warning: could not print full Akida summary (non-fatal): {e}")

    # Optional mapping to hardware (if you have a device connected)
    if cfg.map_to_device:
        try:
            devs = devices()
            if not devs:
                print("[convert_to_akida] No Akida devices found. Skipping mapping.")
            else:
                device = devs[0]
                print(f"[convert_to_akida] Mapping to device: {device}")
                model_akida.map(device, hw_only=cfg.hw_only)
                print("[convert_to_akida] Mapping done. Updated model summary:")
                model_akida.summary()
        except Exception as e:
            print(f"[convert_to_akida] Mapping failed (non-fatal). Reason: {e}")

    return model_akida

# -----------------------------
# 9) Evaluate Akida model
# -----------------------------

def _ensure_uint8_images(x: np.ndarray) -> np.ndarray:
    """
    Ensure images are uint8 [0..255]. Accepts:
      - uint8 already
      - float [0,1] or [0,255] (best-effort)
    """
    if x.dtype == np.uint8:
        return x

    x_f = x.astype(np.float32)

    # Heuristic: if max <= 1.5 assume [0,1]; else assume already [0,255]-ish
    mx = float(np.max(x_f))
    if mx <= 1.5:
        x_f = x_f * 255.0

    x_f = np.clip(x_f, 0.0, 255.0)
    return (x_f + 0.5).astype(np.uint8)

def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D arrays."""
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def _collect_numpy_from_tfdata(
    ds: tf.data.Dataset,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (images, labels) from a batched tf.data.Dataset into numpy arrays.
    Preserves image dtype (now typically uint8).
    """
    xs, ys = [], []
    collected = 0

    for xb, yb in ds:
        xb_np = xb.numpy()
        yb_np = yb.numpy()

        if max_samples is not None:
            remaining = max_samples - collected
            if remaining <= 0:
                break
            xb_np = xb_np[:remaining]
            yb_np = yb_np[:remaining]

        xs.append(xb_np)
        ys.append(yb_np)
        collected += len(yb_np)

        if max_samples is not None and collected >= max_samples:
            break

    if not xs:
        raise RuntimeError("[_collect_numpy_from_tfdata] No data collected from dataset.")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int32)
    return x, y


def evaluate_akida_model(
    model_akida,
    ds: tf.data.Dataset,
    class_names: List[str],
    split_name: str = "test_akida",
    max_samples: Optional[int] = None,
    batch_size: int = 0,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate an Akida model on a tf.data.Dataset (which yields float images in [0,1]).
    Converts inputs to uint8 and uses Akida:
      - model_akida.evaluate(inputs, labels, num_classes=?, batch_size=?) -> accuracy :contentReference[oaicite:2]{index=2}
      - model_akida.predict_classes(inputs, num_classes=?, batch_size=?) -> predicted labels :contentReference[oaicite:3]{index=3}
    """
    n_classes = num_classes if num_classes is not None else len(class_names)
    print(f"\n[evaluate_akida_model] Evaluating Akida model on split='{split_name}' ...")
    print(f"[evaluate_akida_model] num_classes={n_classes}, max_samples={max_samples}, batch_size={batch_size}")

    # 1) Collect data from tf.data -> numpy
    x_np, y_true = _collect_numpy_from_tfdata(ds, max_samples=max_samples)
    print("[evaluate_akida_model] Collected data:")
    print(f"  - x_np: shape={x_np.shape}, dtype={x_np.dtype}, min/max={x_np.min():.3f}/{x_np.max():.3f}")
    print(f"  - y_true:  shape={y_true.shape}, dtype={y_true.dtype}")

    # 2) Convert to uint8 for Akida evaluate/predict_classes
    x_u8 = _ensure_uint8_images(x_np)

    print("[evaluate_akida_model] Converted inputs for Akida:")
    print(f"  - x_u8: shape={x_u8.shape}, dtype={x_u8.dtype}, min/max={x_u8.min()}/{x_u8.max()}")

    # 3) Akida accuracy
    acc = float(model_akida.evaluate(x_u8, y_true, num_classes=n_classes, batch_size=batch_size))
    print(f"[evaluate_akida_model] Akida evaluate() accuracy: {acc:.4f}")

    # 4) Predictions for confusion matrix / per-class metrics
    y_pred = model_akida.predict_classes(x_u8, num_classes=n_classes, batch_size=batch_size).astype(np.int64)
    overall_acc = float((y_pred == y_true).mean())
    print(f"[evaluate_akida_model] Overall accuracy (from predict_classes): {overall_acc:.4f}")

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=n_classes, dtype=tf.int32).numpy()
    print("\n[evaluate_akida_model] Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Per-class metrics
    eps = 1e-9
    per_class = []
    for i in range(n_classes):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        support = float(cm[i, :].sum())
        class_acc = tp / (support + eps)

        name = class_names[i] if i < len(class_names) else str(i)
        per_class.append({
            "class_id": i,
            "class_name": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc": class_acc,
            "support": support,
        })

    macro_precision = float(np.mean([d["precision"] for d in per_class]))
    macro_recall = float(np.mean([d["recall"] for d in per_class]))
    macro_f1 = float(np.mean([d["f1"] for d in per_class]))

    print("\n[evaluate_akida_model] Per-class metrics:")
    print("  {:>2}  {:<22}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}".format(
        "id", "class", "prec", "recall", "f1", "acc", "support"
    ))
    for d in per_class:
        print("  {:>2}  {:<22}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.0f}".format(
            d["class_id"], d["class_name"][:22],
            d["precision"], d["recall"], d["f1"], d["acc"], d["support"]
        ))

    print("\n[evaluate_akida_model] Macro averages:")
    print(f"  - macro_precision: {macro_precision:.4f}")
    print(f"  - macro_recall:    {macro_recall:.4f}")
    print(f"  - macro_f1:        {macro_f1:.4f}")

    return {
        "akida_eval_acc": acc,
        "overall_acc": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
    }
    
def show_single_prediction_akida(
    model_akida,
    ds: tf.data.Dataset,
    class_names: List[str],
    top_k: int = 5,
    show_image: bool = True,
    save_path: Optional[str] = None,
    batch_size: int = 0,
    num_classes: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Show Akida predictions for a random sample taken from the first batch of `ds`.

    Why this approach:
      - avoids ds.unbatch().skip(k), which can be slow
      - works well for debugging and quick checks

    Args:
      seed: if provided, makes the random pick reproducible.
    """
    n_classes = num_classes if num_classes is not None else len(class_names)
    k = min(max(top_k, 1), n_classes)

    # Grab a single batch
    xb, yb = next(iter(ds))
    x_np = xb.numpy()
    y_np = yb.numpy().astype(np.int64)

    # Ensure uint8 images
    x_u8_batch = _ensure_uint8_images(x_np)

    # Choose random index within the batch
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    idx = int(rng.integers(low=0, high=x_u8_batch.shape[0]))

    img_u8 = x_u8_batch[idx]                 # (H,W,C) uint8
    true_id = int(y_np[idx])
    true_name = class_names[true_id] if 0 <= true_id < len(class_names) else str(true_id)

    print(f"\n[show_single_prediction_akida] Random sample from first batch: idx_in_batch={idx}")
    print(f"[show_single_prediction_akida] Ground truth: id={true_id}, class='{true_name}'")

    inp = img_u8[None, ...]  # (1,H,W,C)

    # Prefer predict(); fallback to forward()
    try:
        scores = model_akida.predict(inp, batch_size=batch_size)
    except RuntimeError as e:
        print(f"[show_single_prediction_akida] predict() not available. Falling back to forward(). Reason: {e}")
        scores = model_akida.forward(inp, batch_size=batch_size).astype(np.float32)

    scores = np.squeeze(scores, axis=0)

    # Handle possible shapes
    if scores.ndim == 3:
        scores_vec = scores.mean(axis=(0, 1))
    elif scores.ndim == 1:
        scores_vec = scores
    else:
        scores_vec = scores.reshape(-1)

    probs = _softmax_np(scores_vec[:n_classes])
    topk_ids = np.argsort(-probs)[:k]
    topk_probs = probs[topk_ids]

    pred_id = int(topk_ids[0])
    pred_name = class_names[pred_id] if 0 <= pred_id < len(class_names) else str(pred_id)

    print("[show_single_prediction_akida] Top predictions:")
    for rank, (cid, p) in enumerate(zip(topk_ids, topk_probs), start=1):
        cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        print(f"  {rank:>2}. {cname:<22}  p~{p:.4f}  (class_id={cid})")

    if show_image or save_path is not None:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(img_u8)
        plt.axis("off")
        plt.title(f"True: {true_name} | Pred: {pred_name} ({topk_probs[0]:.3f})")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[show_single_prediction_akida] Saved figure to: {save_path}")

        if show_image:
            plt.show()
        plt.close()

    return {
        "idx_in_batch": idx,
        "true_id": true_id,
        "true_name": true_name,
        "topk_ids": topk_ids,
        "topk_names": [class_names[i] for i in topk_ids],
        "topk_probs": topk_probs,
    }

# -----------------------------
# main()
# -----------------------------

def main() -> None:
    cfg = PipelineConfig()

    if cfg.seed is not None:
        set_seeds(cfg.seed)

    raw_datasets, ds_info = load_eurosat_tfds(
        tfds_name=cfg.tfds_name,
        train_pct=cfg.train_pct,
        val_pct=cfg.val_pct,
        test_pct=cfg.test_pct,
        data_dir=cfg.data_dir,
    )

    pipelines = build_tfdata_pipelines(
        datasets=raw_datasets,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        resize_to=cfg.resize_to,
        shuffle_buffer=cfg.shuffle_buffer,
        cache=cfg.cache,
        augment=cfg.augment,
    )

    # Additional quick checks (optional but helpful)
    num_classes = ds_info.features["label"].num_classes
    input_shape = ds_info.features["image"].shape if cfg.resize_to is None else (*cfg.resize_to, 3)
    print("[main] Ready for next step (model definition).")
    print(f"[main] num_classes={num_classes}, input_shape={input_shape}")
    print("[main] Pipelines keys:", list(pipelines.keys()))
    
    model = build_baseline_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        base_filters=32,
        dense_units=128,
        weight_decay=1e-4,
    )
    
    train_cfg = TrainConfig(
        epochs=30,
        learning_rate=1e-3,
        optimizer="adam",
        checkpoint_dir=project_root / "checkpoints",
        run_name="eurosat_cnn_baseline",
        early_stop_patience=7,
        reduce_lr_patience=3,
        use_tensorboard=False,
    )

    model, history, best_path = train_cnn_model(
        model=model,
        train_ds=pipelines["train"],
        val_ds=pipelines["val"],
        cfg=train_cfg,
    )
    
    class_names = ds_info.features["label"].names
    
    _ = evaluate_model(
        model=model,
        ds=pipelines["val"],
        class_names=class_names,
        split_name="val",
        from_logits=True,
    )

    _ = evaluate_model(
        model=model,
        ds=pipelines["test"],
        class_names=class_names,
        split_name="test",
        from_logits=True,
    )
    
    # # Show a prediction from the test set (index 0)
    # _ = show_single_prediction(
    #     model=model,
    #     ds=pipelines["test"],
    #     class_names=class_names,
    #     index=0,
    #     top_k=5,
    #     from_logits=True,
    #     show_image=True,
    #     save_path=None,  # or "single_pred.png"
    # )
    
    qcfg = QuantizeConfig(
        input_weight_bits=8,
        weight_bits=8,
        activation_bits=8,
        num_samples=1024,
        batch_size=100,
        epochs=2,
        resize_to=cfg.resize_to,  # must match your pipeline choice
    )
    
    model_quant = quantize_and_calibrate(
        model_cnn=model,
        raw_train_ds=raw_datasets["train"],  # IMPORTANT: raw (unbatched, unaugmented) TFDS split
        cfg=qcfg,
    )

    _ = evaluate_model(
        model=model_quant,
        ds=pipelines["test"],
        class_names=class_names,
        split_name="test_quant",
        from_logits=True)
    
    akcfg = AkidaConvertConfig(
        save_path=project_root / "checkpoints" / "eurosat_akida_model.fbz",
        akida_version="v1",
        map_to_device=False,  # set True if you have hardware and want to map now
        hw_only=True,
    )
    
    model_akida = convert_to_akida(
        model_quantized=model_quant,
        cfg=akcfg,
    )
    
    _ = evaluate_akida_model(
        model_akida=model_akida,
        ds=pipelines["test"],
        class_names=class_names,
        split_name="test_akida",
        max_samples=None,   # set e.g. 1000 for quicker debug
        batch_size=256,     # Akida processes inputs in chunks; tune for your RAM
    )
    
    _ = show_single_prediction_akida(
        model_akida=model_akida,
        ds=pipelines["test"],
        class_names=class_names,
        top_k=5,
        show_image=True,
    )


if __name__ == "__main__":

    main()
