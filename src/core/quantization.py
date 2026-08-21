from __future__ import annotations

import numpy as np
import tensorflow as tf

from quantizeml.models import QuantizationParams, quantize

from .config import QuantizeConfig

def _extract_calibration_samples(
    raw_train_ds: tf.data.Dataset,
    num_samples: int,
    resize_to: tuple[int, int] | None,
    seed: int,
) -> np.ndarray:
    """
    Extract 'num_samples' images from the raw TFDS train dataset (no augmentation),
    applying the same preprocessing the model expects: uint8 in [0,255] + optional resize.

    Returns:
      samples: np.ndarray of shape (num_samples, H, W, C), dtype float32
    """
    
    if num_samples <= 0:
        raise ValueError(
            "num_samples must be greater than 0. "
            f"Received: {num_samples}"
        )
    
    print(f"[calibration] Extracting {num_samples} calibration samples (no augmentation)...")

    def preprocess_only(image: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """
        Prepare one image using the same resizing logic as dataset.py.

        The label argument is required because the dataset produces
        (image, label), but calibration does not use the label.
        """
        del label
        
        image = tf.cast(image, tf.uint8)

        if resize_to is not None:
            # Resize in float, then convert back to uint8.
            img_f = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
            img_f = tf.image.resize(img_f, resize_to, method="bilinear")
            image = tf.image.convert_image_dtype(img_f, tf.uint8, saturate=True)

        return image


    #deterministic shuffle
    ds = (
        raw_train_ds
        .shuffle(
            buffer_size=10_000,
            seed=seed,
            reshuffle_each_iteration=False,
        )
        .map(preprocess_only, num_parallel_calls=tf.data.AUTOTUNE)
        .take(num_samples)
    )

    # Materialize into numpy
    samples_list = [img.numpy() for img in ds]
    if not samples_list:
        raise RuntimeError("[calibration] No samples extracted. Check dataset pipeline.")

    samples = np.stack(samples_list, axis=0).astype(np.uint8)  # shape: (num_samples, H, W, C)

    print("[calibration] Calibration samples ready:")
    print(f"  - shape: {samples.shape}")
    print(f"  - dtype: {samples.dtype}")
    print(f"  - min/max: {samples.min()} / {samples.max()}")
    
    if len(samples) < num_samples:
        print(
            "[calibration] Warning: requested "
            f"{num_samples} samples, but only {len(samples)} were available."
        )

    return samples

def quantize_and_calibrate(
    model: tf.keras.Model,
    raw_train_ds: tf.data.Dataset,
    cfg: QuantizeConfig,
    seed: int = 42,
) -> tf.keras.Model:
    """
    Quantize + calibrate a TF-Keras model using QuantizeML.

    This follows the recommended approach of calibrating OutputQuantizers using real samples
    passed to quantize(...), with commonly used settings (1024 samples, batch_size=100, epochs=2). :contentReference[oaicite:2]{index=2}

    Returns:
      model_quantized: quantized TF-Keras model
    """
    if cfg.input_weight_bits <= 0:
        raise ValueError(
            "input_weight_bits must be greater than 0."
        )

    if cfg.weight_bits <= 0:
        raise ValueError(
            "weight_bits must be greater than 0."
        )

    if cfg.activation_bits <= 0:
        raise ValueError(
            "activation_bits must be greater than 0."
        )

    if cfg.batch_size <= 0:
        raise ValueError(
            "Quantization batch_size must be greater than 0."
        )

    if cfg.epochs <= 0:
        raise ValueError(
            "Quantization epochs must be greater than 0."
        )
    
    print("\n[quantize_and_calibrate] Starting quantization + calibration...")
    print("[quantize_and_calibrate] Quantization scheme:")
    print(f"  - input_weight_bits={cfg.input_weight_bits}, weight_bits={cfg.weight_bits}, activation_bits={cfg.activation_bits}")
    print("[quantize_and_calibrate] Calibration settings:")
    print(f"  requested samples: {cfg.num_samples}")
    print(f"  batch size:        {cfg.batch_size}")
    print(f"  epochs:            {cfg.epochs}")
    print(f"  resize to:         {cfg.resize_to}")
    
    # The EuroSAT Keras model was designed to receive uint8 images directly.
    expected_input_dtype = tf.as_dtype(
        model.inputs[0].dtype
    )

    if expected_input_dtype != tf.uint8:
        print(
            "[quantize_and_calibrate] Warning: the model input dtype is "
            f"{expected_input_dtype.name}, not uint8."
        )
        print(
            "[quantize_and_calibrate] The current EuroSAT pipeline is "
            "designed around uint8 [0, 255] images."
        )

    # Build qparams (8/8/8 default; 8/4/4 later if needed)
    qparams = QuantizationParams(
        input_weight_bits=cfg.input_weight_bits,
        weight_bits=cfg.weight_bits,
        activation_bits=cfg.activation_bits,
        per_tensor_activations=cfg.per_tensor_activations,
        input_dtype="uint8",
        output_bits=8,
        buffer_bits=32,
    )

    # Extract real samples from raw TFDS train split (no augmentation)
    samples = _extract_calibration_samples(
        raw_train_ds=raw_train_ds,
        num_samples=cfg.num_samples,
        resize_to=cfg.resize_to,
        seed=seed,
    )
    
    actual_num_samples = int(
        samples.shape[0]
    )
    
    print(
        "[quantize_and_calibrate] Running QuantizeML quantize()..."
    )

    # Quantize + calibrate in one call
    # Passing samples triggers calibration of OutputQuantizers. :contentReference[oaicite:3]{index=3}
    try:
        model_quantized = quantize(
            model,
            qparams=qparams,
            samples=samples,
            num_samples=actual_num_samples,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
        )
    except Exception as e:
        raise RuntimeError(
            "Quantization failed. Calibration samples were uint8 [0,255]."
        ) from e

    print("[quantize_and_calibrate] Quantization done.")
    
    _compile_quantized_model(
        model=model_quantized,
        optimizer_name=cfg.optimizer,
        learning_rate=cfg.learning_rate,
    )

    print("[quantize_and_calibrate] Quantized model summary:")
    model_quantized.summary()

    return model_quantized

def _compile_quantized_model(
    model: tf.keras.Model,
    optimizer_name: str,
    learning_rate: float,
) -> None:
    """
    Compile the quantized Keras model.

    QuantizeML returns a Keras model, but the returned model is not guaranteed
    to retain the compilation settings of the original model.

    Compilation is required before calling model.evaluate() from evaluation.py.
    """
    optimizer_name = optimizer_name.strip().lower()

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        )

    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
        )

    else:
        raise ValueError(
            f"Unsupported optimizer: '{optimizer_name}'. "
            "Supported optimizers are 'adam' and 'sgd'."
        )

    # The final model layer returns logits rather than softmax probabilities.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
    )

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

    print("[_compile_quantized_model] Quantized model compiled.")
    print(f"  optimizer: {optimizer_name}")
    print(f"  learning rate: {learning_rate}")

def fine_tune_quantized_model(
    quantized_model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    cfg: QuantizeConfig,
) -> tuple[
    tf.keras.Model,
    tf.keras.callbacks.History | None,
    dict[str, float],
]:
    """
    Perform quantization-aware fine-tuning on a QuantizeML model.

    During QAT, the quantized layers remain active during forward passes.
    Training therefore adapts the weights to the errors introduced by
    4-bit weight and activation quantization.

    Returns:
        quantized_model:
            Fine-tuned quantized model with the best validation weights.

        history:
            QAT History object, or None when QAT is disabled.

        validation_metrics:
            Validation metrics after QAT.
    """
    if not cfg.qat_enabled:
        print(
            "[fine_tune_quantized_model] QAT is disabled."
        )

        validation_metrics = quantized_model.evaluate(
            val_dataset,
            verbose=1,
            return_dict=True,
        )

        validation_metrics = {
            name: float(value)
            for name, value in validation_metrics.items()
        }

        return (
            quantized_model,
            None,
            validation_metrics,
        )

    if cfg.qat_epochs <= 0:
        raise ValueError(
            "qat_epochs must be greater than 0."
        )

    if cfg.qat_learning_rate <= 0:
        raise ValueError(
            "qat_learning_rate must be greater than 0."
        )

    print("\n[fine_tune_quantized_model] Starting QAT...")
    print(f"  epochs:        {cfg.qat_epochs}")
    print(f"  learning rate: {cfg.qat_learning_rate}")
    print(f"  optimizer:     {cfg.optimizer}")

    # Recompile with the lower QAT learning rate.
    _compile_quantized_model(
        model=quantized_model,
        optimizer_name=cfg.optimizer,
        learning_rate=cfg.qat_learning_rate,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=cfg.qat_early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=cfg.qat_reduce_lr_factor,
            patience=cfg.qat_reduce_lr_patience,
            min_lr=cfg.qat_min_lr,
            verbose=1,
        ),
    ]

    history = quantized_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=cfg.qat_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(
        "[fine_tune_quantized_model] "
        "QAT completed. Evaluating validation data..."
    )

    try:
        raw_validation_metrics = quantized_model.evaluate(
            val_dataset,
            verbose=1,
            return_dict=True,
        )

    except TypeError:
        metric_values = quantized_model.evaluate(
            val_dataset,
            verbose=1,
        )

        raw_validation_metrics = dict(
            zip(
                quantized_model.metrics_names,
                metric_values,
            )
        )

    validation_metrics: dict[str, float] = {
        name: float(value)
        for name, value in raw_validation_metrics.items()
    }

    print("[fine_tune_quantized_model] Validation metrics:")

    for metric_name, metric_value in validation_metrics.items():
        print(
            f"  {metric_name}: {metric_value:.4f}"
        )

    return (
        quantized_model,
        history,
        validation_metrics,
    )
    
def clone_without_regularizers(
    model: tf.keras.Model,
) -> tf.keras.Model:
    """
    Clone a trained Keras model while removing layer regularizers.

    The trained weights are copied unchanged. This preserves the effect
    that regularization had during float training, but prevents unusually
    large regularization penalties from dominating QAT.
    """

    regularizer_fields = (
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "depthwise_regularizer",
        "pointwise_regularizer",
    )

    def clone_layer(
        layer: tf.keras.layers.Layer,
    ) -> tf.keras.layers.Layer:
        layer_config = layer.get_config()

        for field_name in regularizer_fields:
            if field_name in layer_config:
                layer_config[field_name] = None

        return layer.__class__.from_config(
            layer_config
        )

    cloned_model = tf.keras.models.clone_model(
        model,
        clone_function=clone_layer,
    )

    cloned_model.set_weights(
        model.get_weights()
    )

    print(
        "[clone_without_regularizers] "
        "Created weight-identical model without regularizers."
    )
    print(
        "[clone_without_regularizers] "
        f"Regularization terms: {len(cloned_model.losses)}"
    )

    return cloned_model