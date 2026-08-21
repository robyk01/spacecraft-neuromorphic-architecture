from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def _prepare_dataset_for_model_input(
    ds: tf.data.Dataset,
    model: tf.keras.Model,
) -> tf.data.Dataset:
    """
    Ensure dataset image dtype matches model input dtype.

    The normal Keras model expects uint8 images, but a QuantizeML model may
    declare a float32 input. In that case, the images are cast to float32
    while keeping their numeric range [0, 255].

    The model's internal Rescaling layer still performs normalization.
    """
    expected_dtype = tf.as_dtype(model.inputs[0].dtype)
    ds_dtype = ds.element_spec[0].dtype
    
    if ds_dtype == expected_dtype:
        return ds
    
    print(
        "[_prepare_dataset_for_model_input] Casting images from "
        f"{ds_dtype.name} to {expected_dtype.name}."
    )

    def cast_image(
        image: tf.Tensor,
        label: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, expected_dtype)
        return image, label

    return ds.map(
        cast_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
def evaluate_keras_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str] | tuple[str, ...],
    split_name: str = "test",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a Keras or QuantizeML classification model.

    This function performs two evaluations:

    1. model.evaluate()
       Calculates the compiled Keras loss and metrics.

    2. Manual prediction collection
       Calculates the confusion matrix, precision, recall, F1 score, and per-class metrics.

    Returns:
        A dictionary containing Keras metrics and classification metrics.
    """
    if not class_names:
        raise ValueError(
            "class_names cannot be empty."
        )

    if max_batches is not None and max_batches <= 0:
        raise ValueError(
            "max_batches must be greater than 0 or None."
        )

    print(
        f"\n[evaluate_keras_model] Evaluating split: "
        f"{split_name}"
    )
    print(
        f"[evaluate_keras_model] Number of classes: "
        f"{len(class_names)}"
    )

    # This is especially important for the future quantized model, whose
    # declared input dtype may differ from the normal Keras model.
    evaluation_dataset = _prepare_dataset_for_model_input(
        ds=dataset,
        model=model,
    )
    
    if max_batches is not None:
        evaluation_dataset = evaluation_dataset.take(
            max_batches
        )

    print(
        "[evaluate_keras_model] Running model.evaluate()..."
    )

    try:
        keras_metrics = model.evaluate(
            evaluation_dataset,
            verbose=1,
            return_dict=True,
        )

    except TypeError:
        # Compatibility fallback for older TensorFlow/Keras versions.
        metric_values = model.evaluate(
            evaluation_dataset,
            verbose=1,
        )

        keras_metrics = dict(
            zip(
                model.metrics_names,
                metric_values,
            )
        )

    # Convert NumPy/TensorFlow scalar values to regular Python floats.
    keras_metrics = {
        metric_name: float(metric_value)
        for metric_name, metric_value in keras_metrics.items()
    }

    print(
        "[evaluate_keras_model] Collecting predictions..."
    )

    true_labels: list[np.ndarray] = []
    predicted_labels: list[np.ndarray] = []

    for batch_images, batch_labels in evaluation_dataset:
        # The model outputs logits. Softmax is unnecessary when selecting
        # the largest output because argmax(logits) and argmax(softmax)
        # produce the same class ID.
        batch_outputs = model(
            batch_images,
            training=False,
        )

        batch_predictions = tf.argmax(
            batch_outputs,
            axis=-1,
            output_type=tf.int64,
        )

        true_labels.append(
            batch_labels.numpy().astype(np.int64)
        )

        predicted_labels.append(
            batch_predictions.numpy()
        )

    if not true_labels:
        raise RuntimeError(
            "No samples were collected from the evaluation dataset."
        )

    y_true = np.concatenate(
        true_labels,
        axis=0,
    )

    y_pred = np.concatenate(
        predicted_labels,
        axis=0,
    )

    classification_metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    print("\nKeras metrics:")

    for metric_name, metric_value in keras_metrics.items():
        print(
            f"  {metric_name}: "
            f"{metric_value:.4f}"
        )

    print_classification_metrics(
        classification_metrics
    )

    return {
        "split_name": split_name,
        "keras_metrics": keras_metrics,
        **classification_metrics,
    }
    
def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """
    Calculate the confusion matrix and classification metrics.

    Args:
        y_true:
            Array containing the correct class IDs.

        y_pred:
            Array containing the predicted class IDs.

        class_names:
            Class names in the same order as their numerical class IDs.

    Returns:
        A dictionary containing:
            - overall accuracy;
            - macro precision;
            - macro recall;
            - macro F1 score;
            - per-class metrics;
            - confusion matrix.

    The returned dictionary contains only normal Python values and lists,
    making it suitable for later JSON saving.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) == 0:
        raise ValueError(
            "Cannot calculate metrics because no labels were provided."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number of elements. "
            f"Received {len(y_true)} true labels and "
            f"{len(y_pred)} predictions."
        )

    num_classes = len(class_names)

    if num_classes <= 1:
        raise ValueError(
            "class_names must contain at least two classes."
        )

    # Make sure class IDs can be used as confusion-matrix indexes.
    if np.any(y_true < 0) or np.any(y_true >= num_classes):
        raise ValueError(
            "y_true contains a class ID outside the valid range "
            f"0 to {num_classes - 1}."
        )

    if np.any(y_pred < 0) or np.any(y_pred >= num_classes):
        raise ValueError(
            "y_pred contains a class ID outside the valid range "
            f"0 to {num_classes - 1}."
        )

    confusion_matrix = tf.math.confusion_matrix(
        labels=y_true,
        predictions=y_pred,
        num_classes=num_classes,
        dtype=tf.int32,
    ).numpy()

    overall_accuracy = float(
        np.mean(y_true == y_pred)
    )

    per_class_metrics: list[dict[str, Any]] = []

    for class_id, class_name in enumerate(class_names):
        true_positive = float(
            confusion_matrix[class_id, class_id]
        )

        false_negative = float(
            confusion_matrix[class_id, :].sum() - true_positive
        )

        false_positive = float(
            confusion_matrix[:, class_id].sum() - true_positive
        )

        support = int(
            confusion_matrix[class_id, :].sum()
        )

        # Explicit checks avoid division by zero when a class has no
        # predicted or true samples.
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        if precision_denominator == 0:
            precision = 0.0
        else:
            precision = true_positive / precision_denominator

        if recall_denominator == 0:
            recall = 0.0
        else:
            recall = true_positive / recall_denominator

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (
                2.0
                * precision
                * recall
                / (precision + recall)
            )


        per_class_metrics.append(
            {
                "class_id": class_id,
                "class_name": str(class_name),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1_score),
                "support": support,
            }
        )

    macro_precision = float(
        np.mean(
            [
                class_result["precision"]
                for class_result in per_class_metrics
            ]
        )
    )

    macro_recall = float(
        np.mean(
            [
                class_result["recall"]
                for class_result in per_class_metrics
            ]
        )
    )

    macro_f1 = float(
        np.mean(
            [
                class_result["f1"]
                for class_result in per_class_metrics
            ]
        )
    )

    return {
        "overall_accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class_metrics,

        # Convert the NumPy matrix to a regular list so the result can
        # later be saved directly in a JSON file.
        "confusion_matrix": confusion_matrix.tolist(),
    }

def print_classification_metrics(
    metrics: dict[str, Any],
) -> None:
    """
    Print classification metrics in a readable table.
    """
    print("\nConfusion matrix:")
    print("(rows = true classes, columns = predicted classes)")

    confusion_matrix = np.asarray(
        metrics["confusion_matrix"]
    )

    print(confusion_matrix)

    print("\nPer-class metrics:")

    header = (
        f"{'ID':>3}  "
        f"{'Class':<22}  "
        f"{'Precision':>9}  "
        f"{'Recall':>9}  "
        f"{'F1':>9}  "
        f"{'Support':>8}"
    )

    print(header)
    print("-" * len(header))

    for class_result in metrics["per_class_metrics"]:
        print(
            f"{class_result['class_id']:>3}  "
            f"{class_result['class_name'][:22]:<22}  "
            f"{class_result['precision']:>9.3f}  "
            f"{class_result['recall']:>9.3f}  "
            f"{class_result['f1']:>9.3f}  "
            f"{class_result['support']:>8}"
        )

    print("\nOverall metrics:")
    print(
        f"  accuracy:        "
        f"{metrics['overall_accuracy']:.4f}"
    )
    print(
        f"  macro precision: "
        f"{metrics['macro_precision']:.4f}"
    )
    print(
        f"  macro recall:    "
        f"{metrics['macro_recall']:.4f}"
    )
    print(
        f"  macro F1:        "
        f"{metrics['macro_f1']:.4f}"
    )

def show_single_keras_prediction(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str] | tuple[str, ...],
    index: int = 0,
    top_k: int = 5,
    from_logits: bool = True,
    show_image: bool = True,
    save_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run inference on one image from a batched TensorFlow dataset.

    The function prints:
        - the correct class;
        - the predicted class;
        - the top-k predictions and their probabilities.

    It can also display or save the selected image.

    Args:
        model:
            Keras or QuantizeML model.

        dataset:
            Batched TensorFlow dataset.

        class_names:
            Class names ordered by their numerical class IDs.

        index:
            Index of the image after the dataset is unbatched.

        top_k:
            Number of highest-probability classes to print.

        from_logits:
            True when the model output contains logits.
            The current EuroSAT model uses logits, so this should normally
            remain True.

        show_image:
            Display the image using Matplotlib.

        save_path:
            Optional path where the displayed figure will be saved.
    """
    if index < 0:
        raise ValueError(
            f"index must be 0 or greater. Received: {index}"
        )

    if top_k <= 0:
        raise ValueError(
            f"top_k must be greater than 0. Received: {top_k}"
        )

    if not class_names:
        raise ValueError(
            "class_names cannot be empty."
        )

    print(
        "[show_single_keras_prediction] Fetching image "
        f"at index {index}..."
    )

    # Unbatch the dataset so index refers to an individual image rather
    # than to an entire batch.
    selected_dataset = (
        dataset
        .unbatch()
        .skip(index)
        .take(1)
    )

    image: tf.Tensor | None = None
    label: tf.Tensor | None = None

    for selected_image, selected_label in selected_dataset:
        image = selected_image
        label = selected_label

    if image is None or label is None:
        raise IndexError(
            f"Could not find an image at dataset index {index}."
        )

    expected_dtype = tf.as_dtype(
        model.inputs[0].dtype
    )

    # Keep the original image for display, but cast the inference copy to
    # the dtype expected by the model.
    inference_image = image

    if inference_image.dtype != expected_dtype:
        inference_image = tf.cast(
            inference_image,
            expected_dtype,
        )

    # Add a batch dimension:
    # (height, width, channels) -> (1, height, width, channels)
    inference_batch = tf.expand_dims(
        inference_image,
        axis=0,
    )

    model_output = model(
        inference_batch,
        training=False,
    )

    if from_logits:
        probabilities = tf.nn.softmax(
            model_output,
            axis=-1,
        )
    else:
        probabilities = model_output

    probabilities = tf.squeeze(
        probabilities,
        axis=0,
    )

    number_of_classes = len(class_names)
    selected_top_k = min(
        top_k,
        number_of_classes,
    )

    top_results = tf.math.top_k(
        probabilities,
        k=selected_top_k,
    )

    top_probabilities = (
        top_results.values
        .numpy()
        .astype(float)
    )

    top_class_ids = (
        top_results.indices
        .numpy()
        .astype(int)
    )

    true_class_id = int(
        label.numpy()
    )

    if not 0 <= true_class_id < number_of_classes:
        raise ValueError(
            f"True class ID {true_class_id} is outside the valid "
            f"range 0 to {number_of_classes - 1}."
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
        "[show_single_keras_prediction] Correct class:"
    )
    print(
        f"  {true_class_name} "
        f"(class ID {true_class_id})"
    )

    print(
        "[show_single_keras_prediction] Top predictions:"
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
        class_name = class_names[class_id]

        print(
            f"  {rank}. "
            f"{class_name:<22} "
            f"probability={probability:.4f} "
            f"(class ID {class_id})"
        )

    if show_image or save_path is not None:
        display_image = image.numpy()

        # Matplotlib expects floating-point RGB images to be in [0, 1].
        # Our normal dataset contains uint8 images in [0, 255], but this
        # also keeps the function safe if a float image is provided.
        if np.issubdtype(
            display_image.dtype,
            np.floating,
        ):
            if float(np.max(display_image)) > 1.5:
                display_image = np.clip(
                    display_image,
                    0.0,
                    255.0,
                ).astype(np.uint8)

        plt.figure()
        plt.imshow(display_image)
        plt.axis("off")
        plt.title(
            f"True: {true_class_name}\n"
            f"Predicted: {predicted_class_name} "
            f"({top_probabilities[0]:.3f})"
        )

        if save_path is not None:
            save_path = Path(save_path)

            # Create the parent directory if it does not already exist.
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
                "[show_single_keras_prediction] Figure saved to: "
                f"{save_path}"
            )

        if show_image:
            plt.show()

        plt.close()

    return {
        "index": index,
        "true_class_id": true_class_id,
        "true_class_name": true_class_name,
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": predicted_class_name,
        "top_class_ids": top_class_ids.tolist(),
        "top_class_names": [
            str(class_names[class_id])
            for class_id in top_class_ids
        ],
        "top_probabilities": top_probabilities.tolist(),
    }