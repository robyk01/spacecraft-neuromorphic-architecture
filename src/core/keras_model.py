from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from .config import ModelConfig

def _separable_block(
    block_input: tf.Tensor,
    filters: int,
    stride: int,
    block_name: str,
    regularizer: tf.keras.regularizers.Regularizer,
    relu_max_value: float,
) -> tf.Tensor:
    """
    Apply one Akida 1.0-compatible separable convolution block.

    SeparableConv2D performs the depthwise and pointwise operations as one
    fused Keras layer. This corresponds to the fused separable operation
    used by Akida 1.0.

    The ReLU is bounded because unbounded ReLU is not supported for an
    Akida 1.0 hardware target.
    """
    block_output = tf.keras.layers.SeparableConv2D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
        name=f"{block_name}_sep",
    )(block_input)

    block_output = tf.keras.layers.BatchNormalization(
        name=f"{block_name}_bn",
    )(block_output)

    block_output = tf.keras.layers.ReLU(
        max_value=relu_max_value,
        name=f"{block_name}_relu",
    )(block_output)

    return block_output

def build_baseline_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    cfg: ModelConfig,
) -> tf.keras.Model:
    """
    Build a compact CNN baseline suitable for:
      - EuroSAT RGB (64x64x3)
      - later quantization with QuantizeML
      - later conversion to Akida SNN with cnn2snn.convert
      
    The model expects uint8 images in the range [0, 255].
    Image normalization is performed inside the model.
    
    The final layer returns logits rather than probabilities.
    Training must therefore use a loss configured with from_logits=True.

    Design choices for Akida/quantization friendliness:
      - DepthwiseConv2D + 1x1 Conv2D blocks (efficient and common in Akida examples)
      - ReLU as explicit layers
      - Downsampling via strided convolutions (avoids MaxPool ordering constraints)
      - No Dropout / no exotic activations
      - Output logits (Dense(num_classes) without softmax)
    """
    
    if len(input_shape) != 3:
        raise ValueError(
            "input_shape must contain height, width, and channels. "
            f"Received: {input_shape}"
        )

    if num_classes <= 1:
        raise ValueError(
            "num_classes must be greater than 1. "
            f"Received: {num_classes}"
        )
    
    print("[build_baseline_cnn] Building baseline CNN...")
    print(f"  input shape:  {input_shape}")
    print(f"  classes:      {num_classes}")
    print(f"  base filters: {cfg.base_filters}")
    print(f"  dense units:  {cfg.dense_units}")
    print(f"  weight decay: {cfg.weight_decay}")


    l2 = tf.keras.regularizers.l2(cfg.weight_decay)

    inputs = tf.keras.Input(shape=input_shape, dtype=tf.uint8, name="input")

    # The dataset provides uint8 images in the range [0, 255].
    # Convert them to float32 in the range [0, 1].
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_u8_to_f32")(inputs)

    # Initial convolution.
    # For 64x64 input images, this reduces the size to 32x32.
    x = tf.keras.layers.Conv2D(
        filters=cfg.base_filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2,
        name="stem_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu", max_value=cfg.relu_max_value)(x)

    # Blocks: progressively downsample (32 -> 16 -> 8) and increase channels
    x = _separable_block(block_input=x, filters=cfg.base_filters * 2, stride=1, block_name="b1", regularizer=l2, relu_max_value=cfg.relu_max_value)  # 32x32
    x = _separable_block(block_input=x, filters=cfg.base_filters * 2, stride=2, block_name="b2", regularizer=l2, relu_max_value=cfg.relu_max_value)  # 16x16

    x = _separable_block(block_input=x, filters=cfg.base_filters * 4, stride=1, block_name="b3", regularizer=l2, relu_max_value=cfg.relu_max_value)  # 16x16
    x = _separable_block(block_input=x, filters=cfg.base_filters * 4, stride=2, block_name="b4", regularizer=l2, relu_max_value=cfg.relu_max_value)  # 8x8

    # Do not use _separable_block() here because GAP must be positioned
    # before BatchNormalization and ReLU for Akida 1.0.
    x = tf.keras.layers.SeparableConv2D(
        filters=cfg.base_filters * 8,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        depthwise_regularizer=l2,
        pointwise_regularizer=l2,
        name="b5_sep",
    )(x)


    # Convert the feature map into one vector per image.
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    x = tf.keras.layers.BatchNormalization(name="b5_bn")(x)
    x = tf.keras.layers.ReLU(name="b5_relu", max_value=cfg.relu_max_value)(x)
    
    # ---------------------------------------------------------
    # Dense classification head
    # ---------------------------------------------------------
    
    x = tf.keras.layers.Dense(
        units=cfg.dense_units,
        use_bias=False,
        kernel_regularizer=l2,
        name="fc1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="fc1_bn")(x)
    x = tf.keras.layers.ReLU(name="fc1_relu", max_value=cfg.relu_max_value)(x)

    outputs = tf.keras.layers.Dense(units=num_classes, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="eurosat_akida_v1_cnn")

    print("[build_baseline_cnn] Model built.")
    model.summary()
    
    return model

def load_keras_model(
    model_path: Path,
) -> tf.keras.Model:
    """
    Load an existing .keras model.

    The model is loaded without its saved optimizer state because training.py
    will compile it again using the current TrainConfig.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Keras model not found: {model_path}"
        )

    if not model_path.is_file():
        raise ValueError(
            f"The Keras model path is not a file: {model_path}"
        )

    if model_path.suffix.lower() != ".keras":
        raise ValueError(
            "Expected a file with the .keras extension. "
            f"Received: {model_path.name}"
        )

    print(f"[load_keras_model] Loading model from: {model_path}")

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )

    print("[load_keras_model] Model loaded successfully.")
    print(f"  model name:   {model.name}")
    print(f"  input shape:  {model.input_shape}")
    print(f"  output shape: {model.output_shape}")

    return model


def validate_model_compatibility(
    model: tf.keras.Model,
    expected_input_shape: tuple[int, int, int],
    expected_num_classes: int,
) -> None:
    """
    Verify that a loaded Keras model is compatible with the current dataset.

    This should be called after loading a model that will continue training.
    """
    actual_input_shape = tuple(model.input_shape[1:])
    actual_num_classes = int(model.output_shape[-1])

    if actual_input_shape != expected_input_shape:
        raise ValueError(
            "Loaded model has an incompatible input shape.\n"
            f"Expected: {expected_input_shape}\n"
            f"Received: {actual_input_shape}"
        )

    if actual_num_classes != expected_num_classes:
        raise ValueError(
            "Loaded model has an incompatible output size.\n"
            f"Expected {expected_num_classes} classes.\n"
            f"Received {actual_num_classes} classes."
        )

    print("[validate_model_compatibility] Model is compatible.")
    print(f"  input shape: {actual_input_shape}")
    print(f"  classes:     {actual_num_classes}")