from __future__ import annotations

import tensorflow as tf

from .config import TrainConfig
from .paths import ModelPaths

def _safe_cardinality(ds: tf.data.Dataset) -> int | None:
    """
    Return dataset cardinality if known, else None.
    """
    cardinality = tf.data.experimental.cardinality(ds)
    if cardinality == tf.data.INFINITE_CARDINALITY or cardinality == tf.data.UNKNOWN_CARDINALITY:
        return None
    try:
        return int(cardinality.numpy())
    except Exception:
        return None
    
def train_cnn_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: TrainConfig,
    paths: ModelPaths,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History, dict[str, float]]:
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
    
    if cfg.epochs <= 0:
        raise ValueError(
            f"epochs must be greater than 0. Received: {cfg.epochs}"
        )

    if cfg.learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than 0. "
            f"Received: {cfg.learning_rate}"
        )
        
    train_batches = _safe_cardinality(train_ds)
    val_batches = _safe_cardinality(val_ds)
    
    print("\n[train_cnn_model] Starting CNN training...")
    print(f"[train_cnn_model] epochs={cfg.epochs}, lr={cfg.learning_rate}, optimizer={cfg.optimizer}")
    
    print(f"[train_cnn_model] train batches: {train_batches if train_batches is not None else 'unknown'}")
    print(f"[train_cnn_model] val batches:   {val_batches if val_batches is not None else 'unknown'}")

    # Compile here instead of inside keras_model.py because compilation is
    # part of the training process, not part of the architecture.
    compile_classifier(
        model=model,
        cfg=cfg,
    )

    callbacks = build_training_callbacks(
        cfg=cfg,
        paths=paths,
    )

    # Train
    print("[train_cnn_model] Fitting...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ModelCheckpoint saved the model with the highest val_acc.
    # Reload it explicitly because EarlyStopping only reliably restores
    # the best weights when early stopping is actually triggered.
    model = tf.keras.models.load_model(
        str(paths.keras),
        compile=False,
    )
    
    # load_model(..., compile=False) restores the architecture and weights,
    # but the model must be compiled again before evaluate() is called.
    compile_classifier(
        model=model,
        cfg=cfg,
    )
    
    print(
        "[train_cnn_model] Evaluating the best checkpoint "
        "on validation data..."
    )
    
    try:
        validation_metrics = model.evaluate(
            val_ds,
            verbose=1,
            return_dict=True,
        )

    except TypeError:
        # Fallback for older TensorFlow/Keras versions that do not support
        # return_dict=True.
        metric_values = model.evaluate(
            val_ds,
            verbose=1,
        )

        validation_metrics = dict(
            zip(
                model.metrics_names,
                metric_values,
            )
        )
        
    # TensorFlow values are sometimes NumPy scalar types.
    # Convert them to normal Python floats for easier printing and JSON saving.
    validation_metrics = {
        name: float(value)
        for name, value in validation_metrics.items()
    }

    print("[train_cnn_model] Validation metrics:")
    
    for metric_name, metric_value in validation_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print(
        "[train_cnn_model] Best Keras model saved to: "
        f"{str(paths.keras)}"
    )

    return model, history, validation_metrics

    
def create_optimizer(
    cfg: TrainConfig,
) -> tf.keras.optimizers.Optimizer:
    """
    Create the optimizer selected in TrainConfig.

    Currently supported:
        - adam
        - sgd
    """
    optimizer_name = cfg.optimizer.strip().lower()

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
        )

    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=cfg.learning_rate,
            momentum=0.9,
            nesterov=True,
        )

    raise ValueError(
        f"Unsupported optimizer: '{cfg.optimizer}'. "
        "Supported optimizers are 'adam' and 'sgd'."
    )
    
def create_loss(
    cfg: TrainConfig,
) -> tf.keras.losses.Loss:
    """
    Create the classification loss.

    The model outputs logits because its final Dense layer does not contain
    a softmax activation. Therefore, from_logits must be True.
    """
    try:
        return tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            label_smoothing=cfg.label_smoothing,
        )

    except TypeError:
        # Some TensorFlow/Keras versions do not support label_smoothing
        # for SparseCategoricalCrossentropy.
        print(
            "[create_loss] This TensorFlow version does not support "
            "label_smoothing for sparse labels."
        )
        print("[create_loss] Continuing without label smoothing.")

        return tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
        )

def compile_classifier(
    model: tf.keras.Model,
    cfg: TrainConfig,
) -> None:
    """
    Compile the Keras classifier using the current training configuration.

    This function is used for both:
        - newly created models;
        - models loaded from a .keras file.

    Loaded models are compiled again so the new TrainConfig controls the
    optimizer and learning rate.
    """
    optimizer = create_optimizer(cfg)
    loss = create_loss(cfg)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="acc",
            ),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=3,
                name="top3_acc",
            ),
        ],
    )

    print("[compile_classifier] Model compiled.")
    print(f"  optimizer:     {cfg.optimizer}")
    print(f"  learning rate: {cfg.learning_rate}")
    print(f"  label smoothing: {cfg.label_smoothing}")
    
def build_training_callbacks(
    cfg: TrainConfig,
    paths: ModelPaths,
) -> list[tf.keras.callbacks.Callback]:
    """
    Create the callbacks used during training.

    The best model is saved according to validation accuracy.
    """
    # Make sure models/<run_name>/ exists before saving anything.
    paths.create()

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(paths.keras),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
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
            mode="min",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
    ]

    if cfg.use_tensorboard:
        # TensorBoard files are kept inside the current model run folder.
        tensorboard_folder = paths.folder / "tensorboard"
        tensorboard_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_folder),
            )
        )

        print(
            "[build_training_callbacks] TensorBoard enabled: "
            f"{tensorboard_folder}"
        )

    return callbacks
