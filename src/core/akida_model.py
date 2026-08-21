from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

from akida import Model, devices
from cnn2snn import (
    AkidaVersion,
    check_model_compatibility,
    convert,
    get_akida_version,
    set_akida_version,
)

from .config import AkidaConvertConfig
from .evaluation import (
    calculate_classification_metrics,
    print_classification_metrics,
)

def _ensure_uint8_images(images: np.ndarray,) -> np.ndarray:
    """
    Ensure images are uint8 [0..255]. Accepts:
      - uint8 already
      - float [0,1] or [0,255] (best-effort)
    """
    images = np.asarray(images)
    
    if images.dtype == np.uint8:
        return images

    images_float = images.astype(np.float32)
    
    if images_float.size == 0:
        raise ValueError(
            "Cannot convert an empty image array to uint8."
        )

    # Heuristic: if max <= 1.5 assume [0,1]; else assume already [0,255]-ish
    mx = float(np.max(images_float))
    if mx <= 1.5:
        images_float = images_float * 255.0

    images_float = np.clip(images_float, 0.0, 255.0)
    
    # Adding 0.5 gives normal rounding before converting to integers.
    return (images_float + 0.5).astype(np.uint8)

def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D arrays."""
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def _collect_numpy_from_tfdata(
    ds: tf.data.Dataset,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    
    x = _ensure_uint8_images(x)
    
    return x, y

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
    
    save_path: Path | None = None

    if cfg.save_path is not None:
        save_path = Path(cfg.save_path)

        if save_path.suffix.lower() != ".fbz":
            raise ValueError(
                "The Akida model output path must end with .fbz. "
                f"Received: {save_path}"
            )

        # Create models/<run_name>/ if it does not already exist.
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        
    target_version = _resolve_akida_version(
        cfg.akida_version
    )

    print(f"[convert_to_akida] Target Akida version context: {target_version.value}")
    print(f"[convert_to_akida] Keras input dtype: {model_quantized.inputs[0].dtype}")
    
    if save_path is None:
        print("  output file: not requested")
    else:
        print(f"  output file: {save_path}")

    # Convert (for QuantizeML models, input_scaling is ignored by CNN2SNN) :contentReference[oaicite:5]{index=5}
    try:
        # Both compatibility rules and conversion behavior depend on the
        # active Akida version context.
        with set_akida_version(target_version):
            print(
                "[convert_to_akida] Active Akida version context: "
                f"{get_akida_version()}"
            )

            model_akida = convert(
                model_quantized,
                file_path=(
                    str(save_path)
                    if save_path is not None
                    else None
                ),
                input_scaling=None,
            )

    except Exception as error:
        raise RuntimeError(
            "CNN2SNN could not convert the quantized Keras model "
            f"for Akida {target_version.value}. "
            "Check the architecture, quantization settings, and "
            "selected Akida target."
        ) from error

    if save_path is not None:
        if not save_path.exists():
            raise RuntimeError(
                "CNN2SNN completed conversion, but the expected .fbz "
                f"file was not found at: {save_path}"
            )

        print(
            "[convert_to_akida] Akida model saved to: "
            f"{save_path}"
        )

    print("[convert_to_akida] Model information:")
    print(f"  IP version:   {model_akida.ip_version}")
    print(f"  input shape:  {model_akida.input_shape}")
    print(f"  output shape: {model_akida.output_shape}")

    # Optional mapping to hardware (if you have a device connected)
    model_akida.summary()

    if cfg.map_to_device:
        map_akida_model_to_device(
            model=model_akida,
            hw_only=cfg.hw_only,
        )

    return model_akida

def _extract_class_scores(
    model_output: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Convert an Akida output tensor into one score per class.

    Depending on the model, Akida output may have a shape such as:

        (1, num_classes)

    or:

        (1, output_height, output_width, num_classes)

    When spatial dimensions exist, their scores are averaged.
    """
    output = np.asarray(model_output)

    if output.size == 0:
        raise RuntimeError(
            "The Akida model returned an empty output."
        )

    # Remove the batch dimension for the single-image prediction.
    if output.shape[0] == 1:
        output = output[0]

    if output.ndim == 1:
        class_scores = output

    else:
        # Assume the last dimension contains the classes.
        # Combine any remaining spatial dimensions into one dimension,
        # then average their values.
        output = output.reshape(
            -1,
            output.shape[-1],
        )

        class_scores = output.mean(
            axis=0,
        )

    if len(class_scores) < num_classes:
        raise RuntimeError(
            "The Akida model output contains fewer values than the "
            "number of EuroSAT classes. "
            f"Output values: {len(class_scores)}, "
            f"classes: {num_classes}"
        )

    return class_scores[:num_classes]

def _get_akida_sample_shape(
    model: Model,
) -> tuple[int, ...]:
    """
    Return the shape of one Akida input sample.

    Depending on the Akida runtime version, model.input_shape may include
    a leading sample/batch dimension.
    """
    input_shape = tuple(model.input_shape)

    if len(input_shape) == 4:
        return input_shape[1:]

    if len(input_shape) == 3:
        return input_shape

    raise RuntimeError(
        "Unexpected Akida model input shape: "
        f"{input_shape}"
    )


def _resolve_akida_version(
    version_name: str,
) -> AkidaVersion:
    """
    Convert the configuration string into CNN2SNN's AkidaVersion enum.
    """
    normalized_version = version_name.strip().lower()

    if normalized_version == "v1":
        return AkidaVersion.v1

    if normalized_version == "v2":
        return AkidaVersion.v2

    raise ValueError(
        "Akida version must be 'v1' or 'v2'. "
        f"Received: {version_name!r}"
    )


def check_akida_compatibility(
    model: tf.keras.Model,
    akida_version: str,
) -> None:
    """
    Check whether a floating-point Keras model is compatible with the
    selected Akida target.

    This checks quantization and conversion compatibility. Without a
    physical device argument, it does not prove that the complete model
    fits on a particular connected device.
    """
    target_version = _resolve_akida_version(
        akida_version
    )

    print(
        "[check_akida_compatibility] "
        f"Checking compatibility with {target_version.value}..."
    )

    try:
        with set_akida_version(target_version):
            print(
                "[check_akida_compatibility] "
                f"Active version context: {get_akida_version()}"
            )

            check_model_compatibility(
                model=model,
                input_dtype="uint8",
            )

    except Exception as error:
        raise RuntimeError(
            "The Keras model is not compatible with the selected "
            f"Akida target: {target_version.value}."
        ) from error

    print(
        "[check_akida_compatibility] "
        "Compatibility check passed."
    )

def map_akida_model_to_device(
    model: Model,
    hw_only: bool = True,
) -> bool:
    """
    Map an Akida model to the first detected hardware device.

    Returns:
        True if the model was mapped successfully.
        False if no Akida device was detected.

    Raises:
        RuntimeError:
            If a device exists but mapping fails.
    """
    available_devices = devices()

    if not available_devices:
        print(
            "[map_akida_model_to_device] "
            "No Akida hardware device detected."
        )
        print(
            "[map_akida_model_to_device] "
            "The model will continue using the software backend."
        )

        return False

    device = available_devices[0]

    print(
        "[map_akida_model_to_device] Mapping model to device:"
    )
    print(f"  {device}")

    try:
        model.map(
            device,
            hw_only=hw_only,
        )

    except Exception as error:
        raise RuntimeError(
            "An Akida device was detected, but the model could not "
            "be mapped to it."
        ) from error

    print(
        "[map_akida_model_to_device] "
        "Model mapped successfully."
    )

    model.summary()

    return True

def load_akida_model(
    model_path: Path,
) -> Model:
    """
    Load an existing Akida .fbz model.

    This function does not require an Akida hardware device. Without
    explicit hardware mapping, inference uses Akida's software backend.

    Args:
        model_path:
            Path to the saved .fbz file.

    Returns:
        Loaded Akida Model object.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Akida model not found: {model_path}"
        )

    if not model_path.is_file():
        raise ValueError(
            f"The Akida model path is not a file: {model_path}"
        )

    if model_path.suffix.lower() != ".fbz":
        raise ValueError(
            "Expected an Akida model with the .fbz extension. "
            f"Received: {model_path.name}"
        )

    print(
        "[load_akida_model] Loading model from: "
        f"{model_path}"
    )

    try:
        # The Akida Model constructor loads the architecture and weights
        # stored inside the .fbz file.
        model = Model(
            str(model_path)
        )

    except Exception as error:
        raise RuntimeError(
            f"Could not load the Akida model: {model_path}"
        ) from error

    print(
        "[load_akida_model] Model loaded successfully."
    )
    print(f"  IP version:   {model.ip_version}")
    print(f"  input shape:  {model.input_shape}")
    print(f"  output shape: {model.output_shape}")

    model.summary()

    return model

def evaluate_akida_model(
    model: Model,
    dataset: tf.data.Dataset,
    class_names: list[str] | tuple[str, ...],
    split_name: str = "test_akida",
    max_samples: int | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """
    Evaluate an Akida model on a TensorFlow dataset.

    The TensorFlow dataset is first converted to NumPy because Akida's
    evaluate() and predict_classes() methods receive NumPy arrays.

    Args:
        model:
            Converted or loaded Akida model.

        dataset:
            Batched EuroSAT validation or test dataset.

        class_names:
            EuroSAT class names ordered by class ID.

        split_name:
            Name used when printing and returning the results.

        max_samples:
            Optional limit for quicker testing.
            Use None to evaluate the entire dataset.

        batch_size:
            Maximum number of inputs processed by Akida at one time.

    Returns:
        Dictionary containing:
            - Akida evaluate() accuracy;
            - manually calculated overall accuracy;
            - confusion matrix;
            - per-class precision, recall, and F1;
            - macro metrics.
    """
    if not class_names:
        raise ValueError(
            "class_names cannot be empty."
        )

    if batch_size < 0:
        raise ValueError(
            "batch_size must be 0 or greater."
        )

    print(
        f"\n[evaluate_akida_model] Evaluating split: "
        f"{split_name}"
    )

    images, true_labels = _collect_numpy_from_tfdata(
        ds=dataset,
        max_samples=max_samples,
    )

    expected_input_shape = _get_akida_sample_shape(model)

    actual_input_shape = tuple(
        images.shape[1:]
    )

    if actual_input_shape != expected_input_shape:
        raise ValueError(
            "The dataset images do not match the Akida model input.\n"
            f"Expected: {expected_input_shape}\n"
            f"Received: {actual_input_shape}"
        )

    num_classes = len(class_names)

    print("[evaluate_akida_model] Collected data:")
    print(f"  images:       {images.shape}")
    print(f"  image dtype:  {images.dtype}")
    print(
        f"  image range:  "
        f"{images.min()} to {images.max()}"
    )
    print(f"  labels:       {true_labels.shape}")
    print(f"  classes:      {num_classes}")
    print(f"  batch size:   {batch_size}")

    # Akida's evaluate method calculates classification accuracy directly.
    akida_accuracy = float(
        model.evaluate(
            images,
            true_labels,
            num_classes=num_classes,
            batch_size=batch_size,
        )
    )

    # Collect class IDs separately so we can calculate the confusion matrix
    # and per-class metrics using evaluation.py.
    predicted_labels = model.predict_classes(
        images,
        num_classes=num_classes,
        batch_size=batch_size,
    ).astype(np.int64)

    classification_metrics = calculate_classification_metrics(
        y_true=true_labels,
        y_pred=predicted_labels,
        class_names=class_names,
    )

    print(
        "[evaluate_akida_model] "
        f"Akida evaluate accuracy: {akida_accuracy:.4f}"
    )

    print_classification_metrics(
        classification_metrics
    )

    return {
        "split_name": split_name,
        "akida_evaluate_accuracy": akida_accuracy,
        **classification_metrics,
    }
    
def show_single_akida_prediction(
    model: Model,
    dataset: tf.data.Dataset,
    class_names: list[str] | tuple[str, ...],
    index: int | None = None,
    top_k: int = 5,
    show_image: bool = True,
    save_path: Path | None = None,
    batch_size: int = 0,
) -> dict[str, Any]:
    """
    Run Akida inference on one image from a TensorFlow dataset.

    Selection behavior:
        - If index is an integer, select that exact image.
        - If index is None, randomly select an image from the entire dataset.

    The displayed probabilities are softmax values calculated from the
    Akida output scores. They are useful for comparing classes, but they
    should not be treated as calibrated confidence values.

    Args:
        model:
            Loaded or converted Akida model.

        dataset:
            Batched EuroSAT dataset containing (image, label) pairs.

        class_names:
            EuroSAT class names ordered by class ID.

        index:
            Index of the individual image after unbatching the dataset.
            Use None to select a random image.

        top_k:
            Number of top predictions to print.

        show_image:
            Display the selected image using Matplotlib.

        save_path:
            Optional path where the prediction image should be saved.

        batch_size:
            Akida inference batch size.
            Zero uses the Akida default.

    Returns:
        Dictionary containing the selected image index, true class,
        predicted class, and top-k prediction results.
    """
    if index is not None and index < 0:
        raise ValueError(
            f"index must be 0 or greater, or None. Received: {index}"
        )

    if top_k <= 0:
        raise ValueError(
            f"top_k must be greater than 0. Received: {top_k}"
        )

    if batch_size < 0:
        raise ValueError(
            f"batch_size must be 0 or greater. Received: {batch_size}"
        )

    if not class_names:
        raise ValueError(
            "class_names cannot be empty."
        )

    # Unbatch the dataset so each element represents one image rather
    # than one complete batch.
    unbatched_dataset = dataset.unbatch()

    image: tf.Tensor | None = None
    label: tf.Tensor | None = None
    selected_index: int | None = None

    if index is not None:
        # Select a specific image. This is useful when comparing the Keras
        # and Akida models on exactly the same test sample.
        selected_dataset = (
            unbatched_dataset
            .skip(index)
            .take(1)
        )

        for selected_image, selected_label in selected_dataset:
            image = selected_image
            label = selected_label
            selected_index = index

        if image is None or label is None:
            raise IndexError(
                f"Could not find an image at dataset index {index}."
            )

        print(
            "[show_single_akida_prediction] "
            f"Selected fixed image index: {selected_index}"
        )

    else:
        # Randomly select one image from the entire dataset.
        #
        # Reservoir sampling is used because TensorFlow may not know the
        # exact number of elements after dataset.unbatch().
        #
        # Each image has an equal probability of being selected without
        # loading the complete dataset into memory.
        random_generator = np.random.default_rng()

        number_seen = 0

        for current_index, (
            current_image,
            current_label,
        ) in enumerate(unbatched_dataset):
            number_seen += 1

            # Replace the currently selected image with probability 1/n.
            # After the full pass, every image has equal selection chance.
            should_select = (
                random_generator.integers(
                    low=0,
                    high=number_seen,
                )
                == 0
            )

            if should_select:
                image = current_image
                label = current_label
                selected_index = current_index

        if image is None or label is None or selected_index is None:
            raise RuntimeError(
                "Could not select an image because the dataset is empty."
            )

        print(
            "[show_single_akida_prediction] "
            f"Randomly selected image index: {selected_index}"
        )

    # Convert the image to the uint8 [0, 255] format expected by Akida.
    image_numpy = _ensure_uint8_images(
        image.numpy()
    )

    expected_input_shape = _get_akida_sample_shape(model)

    actual_input_shape = tuple(
        image_numpy.shape
    )

    if actual_input_shape != expected_input_shape:
        raise ValueError(
            "The selected image does not match the Akida model input.\n"
            f"Expected: {expected_input_shape}\n"
            f"Received: {actual_input_shape}"
        )

    # Add the batch dimension required by the Akida inference methods:
    #
    # (height, width, channels)
    # becomes
    # (1, height, width, channels)
    input_batch = np.expand_dims(
        image_numpy,
        axis=0,
    )

    try:
        # predict() returns output values intended to reproduce the
        # converted Keras model's output.
        model_output = model.predict(
            input_batch,
            batch_size=batch_size,
        )

    except RuntimeError as error:
        print(
            "[show_single_akida_prediction] "
            "predict() is not available for this model."
        )
        print(
            "[show_single_akida_prediction] "
            "Using forward() instead."
        )
        print(f"  reason: {error}")

        # forward() returns the raw Akida output tensor.
        model_output = model.forward(
            input_batch,
            batch_size=batch_size,
        )

    num_classes = len(class_names)

    # Convert the model output into one score for each class.
    class_scores = _extract_class_scores(
        model_output=model_output,
        num_classes=num_classes,
    )

    # Convert output scores into approximate probabilities for display.
    probabilities = _softmax_np(
        class_scores
    )

    selected_top_k = min(
        top_k,
        num_classes,
    )

    # argsort returns indexes from smallest to largest.
    # Negating the probabilities gives the highest values first.
    top_class_ids = np.argsort(
        -probabilities
    )[:selected_top_k]

    top_probabilities = probabilities[
        top_class_ids
    ]

    true_class_id = int(
        label.numpy()
    )

    if not 0 <= true_class_id < num_classes:
        raise ValueError(
            f"True class ID {true_class_id} is outside the valid "
            f"range 0 to {num_classes - 1}."
        )

    predicted_class_id = int(
        top_class_ids[0]
    )

    true_class_name = str(
        class_names[true_class_id]
    )

    predicted_class_name = str(
        class_names[predicted_class_id]
    )

    print(
        "\n[show_single_akida_prediction] Correct class:"
    )
    print(
        f"  {true_class_name} "
        f"(class ID {true_class_id})"
    )

    print(
        "[show_single_akida_prediction] Top predictions:"
    )

    for rank, (
        class_id,
        probability,
    ) in enumerate(
        zip(
            top_class_ids,
            top_probabilities,
        ),
        start=1,
    ):
        class_name = str(
            class_names[class_id]
        )

        print(
            f"  {rank}. "
            f"{class_name:<22} "
            f"probability~{probability:.4f} "
            f"(class ID {class_id})"
        )

    if show_image or save_path is not None:
        plt.figure()
        plt.imshow(image_numpy)
        plt.axis("off")
        plt.title(
            f"Image index: {selected_index}\n"
            f"True: {true_class_name}\n"
            f"Predicted: {predicted_class_name} "
            f"({top_probabilities[0]:.3f})"
        )

        if save_path is not None:
            save_path = Path(save_path)

            # Create the output folder if it does not already exist.
            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                bbox_inches="tight",
                dpi=150,
            )

            print(
                "[show_single_akida_prediction] "
                f"Figure saved to: {save_path}"
            )

        if show_image:
            plt.show()

        plt.close()

    return {
        "index": selected_index,
        "true_class_id": true_class_id,
        "true_class_name": true_class_name,
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": predicted_class_name,
        "top_class_ids": top_class_ids.tolist(),
        "top_class_names": [
            str(class_names[class_id])
            for class_id in top_class_ids
        ],
        "top_probabilities": (
            top_probabilities
            .astype(float)
            .tolist()
        ),
    }