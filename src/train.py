from __future__ import annotations

from landClassification.core import paths

import argparse
import json

from dataclasses import asdict
from pathlib import Path
from typing import Any

# Import dataset.py before the other TensorFlow-dependent modules.
# dataset.py sets the TensorFlow environment variables before importing TF.
from .core.dataset import (
    build_tfdata_pipelines,
    load_eurosat_tfds,
    set_seeds,
)

from .core.akida_model import (
    check_akida_compatibility,
    convert_to_akida,
    evaluate_akida_model,
    show_single_akida_prediction,
)
from .core.config import (
    AkidaConvertConfig,
    DatasetConfig,
    ModelConfig,
    QuantizeConfig,
    TrainConfig,
)
from .core.evaluation import evaluate_keras_model
from .core.keras_model import (
    build_baseline_cnn,
    load_keras_model,
    validate_model_compatibility,
)
from .core.paths import ModelPaths
from .core.quantization import (
    fine_tune_quantized_model,
    quantize_and_calibrate,
    clone_without_regularizers,
)
from .core.training import (
    compile_classifier,
    train_cnn_model,
)

from .core.dataset import eurosat_8_classes


def parse_arguments() -> argparse.Namespace:
    """
    Read command-line arguments.

    Examples:

        Start a new training run:

            python -m landClassification.train ^
                --run-name baseline_v1

        Continue training an existing Keras model:

            python -m landClassification.train ^
                --run-name baseline_v2 ^
                --resume models/baseline_v1/model.keras ^
                --epochs 10 ^
                --learning-rate 0.0001
                
        Skip training and continue with quantization and Akida conversion:
        
            python -m landClassification.train ^
                --run-name baseline_v3 ^
                --resume models/baseline_v1/model.keras ^
                --skip-training
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train, quantize, convert, and evaluate an "
            "EuroSAT land-cover classification model."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="eurosat_baseline",
        help=(
            "Name of the output folder inside models/. "
            "Default: eurosat_baseline"
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Optional path to an existing .keras model. "
            "When provided, training continues from that model."
        ),
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Skip model training and continue directly with evaluation, "
            "quantization, and Akida conversion. This option requires "
            "--resume with an existing .keras model."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of Keras training epochs. Default: 30",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Keras training learning rate. Default: 0.001",
    )

    parser.add_argument(
        "--optimizer",
        choices=("adam", "sgd"),
        default="adam",
        help="Training optimizer. Default: adam",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="EuroSAT pipeline batch size. Default: 64",
    )

    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help=(
            "Optionally resize EuroSAT images. "
            "Example: --resize 128 128"
        ),
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-image augmentation.",
    )

    parser.add_argument(
        "--akida-version",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "Target Akida version. Use v1 for the AKD1000. "
            "Default: v1"
        ),
    )

    parser.add_argument(
        "--max-akida-samples",
        type=int,
        default=None,
        help=(
            "Optionally limit the number of samples used for final "
            "Akida evaluation. By default, the full test set is used."
        ),
    )

    parser.add_argument(
        "--show-prediction",
        action="store_true",
        help="Display one Akida prediction after evaluation.",
    )

    parser.add_argument(
        "--prediction-index",
        type=int,
        default=0,
        help=(
            "Test-set image index used with --show-prediction. "
            "Default: 0"
        ),
    )

    args = parser.parse_args()

    # Skipping training only makes sense when an existing Keras model
    # is supplied.
    if args.skip_training and args.resume is None:
        parser.error(
            "--skip-training requires --resume with the path "
            "to an existing .keras model."
        )

    return args


def save_json(
    file_path: Path,
    content: dict[str, Any],
) -> None:
    """
    Save a dictionary as formatted JSON.

    Path and NumPy scalar values are converted into JSON-compatible values.
    """

    def convert_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)

        # NumPy scalars normally provide .item().
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable."
        )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            content,
            output_file,
            indent=4,
            default=convert_value,
        )

    print(f"[save_json] Saved: {file_path}")


def determine_input_shape(
    dataset_info,
    resize_to: tuple[int, int] | None,
) -> tuple[int, int, int]:
    """
    Determine the input shape used by the Keras model.
    """
    if resize_to is None:
        return tuple(
            dataset_info.features["image"].shape
        )

    return (
        resize_to[0],
        resize_to[1],
        3,
    )


def main() -> None:
    args = parse_arguments()

    resize_to = (
        tuple(args.resize)
        if args.resize is not None
        else None
    )

    # All files generated by this run will be stored inside:
    #
    # models/<run-name>/
    paths = ModelPaths(
        run_name=args.run_name,
    )

    paths.create()

    dataset_cfg = DatasetConfig(
        batch_size=args.batch_size,
        resize_to=resize_to,
        augment=not args.no_augment,
    )

    model_cfg = ModelConfig()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
    )

    quantize_cfg = QuantizeConfig(
        # Calibration preprocessing must match the dataset preprocessing.
        resize_to=dataset_cfg.resize_to,
    )

    akida_cfg = AkidaConvertConfig(
        save_path=paths.fbz,
        akida_version=args.akida_version,

        # Keep this False while running on the PC software backend.
        map_to_device=False,
        hw_only=True,
    )

    print("=" * 70)
    print("EuroSAT land-cover training")
    print("=" * 70)
    print(f"Run name:       {args.run_name}")
    print(f"Output folder:  {paths.folder}")
    print(f"Resume model:   {args.resume}")
    print(f"Skip training:  {args.skip_training}")
    print(f"Akida version:  {args.akida_version}")

    # Set Python, NumPy, and TensorFlow random seeds.
    set_seeds(
        dataset_cfg.seed
    )

    # Load the raw EuroSAT train, validation, and test splits.
    raw_datasets, dataset_info = load_eurosat_tfds(
        dataset_cfg
    )

    # Create the batched training, validation, and test pipelines.
    pipelines = build_tfdata_pipelines(
        datasets=raw_datasets,
        cfg=dataset_cfg,
    )

    class_names = eurosat_8_classes

    num_classes = len(class_names)

    input_shape = determine_input_shape(
        dataset_info=dataset_info,
        resize_to=dataset_cfg.resize_to,
    )

    print("[main] Dataset ready:")
    print(f"  input shape: {input_shape}")
    print(f"  classes:     {num_classes}")
    print(f"  class names: {class_names}")

    # A resumed run loads its existing architecture and trained weights.
    # A new run creates the baseline CNN from ModelConfig.
    if args.resume is not None:
        print(
            "[main] Loading an existing Keras model "
            "for continued training."
        )

        model = load_keras_model(
            model_path=args.resume,
        )

        validate_model_compatibility(
            model=model,
            expected_input_shape=input_shape,
            expected_num_classes=num_classes,
        )

    else:
        print(
            "[main] Building a new Keras model."
        )

        model = build_baseline_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            cfg=model_cfg,
        )
        
    check_akida_compatibility(
        model=model,
        akida_version=akida_cfg.akida_version,
    )

    # Save the configuration before training starts.
    # This also allows test.py to recreate the same dataset settings.
    configuration = {
        "run_name": args.run_name,
        "resume_from": (
            str(args.resume)
            if args.resume is not None
            else None
        ),
        "skip_training": args.skip_training,
        "dataset": asdict(dataset_cfg),
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "quantization": asdict(quantize_cfg),
        "akida": asdict(akida_cfg),
        "input_shape": input_shape,
        "num_classes": num_classes,
        "class_names": class_names,
    }

    save_json(
        file_path=paths.config,
        content=configuration,
    )

    # ---------------------------------------------------------
    # 1. Train the Keras model, unless training was skipped
    # ---------------------------------------------------------

    if args.skip_training:
        print("\n" + "=" * 70)
        print("Training skipped")
        print("=" * 70)
        print(
            "[main] Using the loaded Keras model without additional training."
        )

        # load_keras_model() loads the model with compile=False.
        # Compilation is still required because evaluation.py calls
        # model.evaluate().
        compile_classifier(
            model=model,
            cfg=train_cfg,
        )

        # Save a copy of the loaded model in the current run folder.
        #
        # This is useful when the source model belongs to a different run:
        #
        # source:
        #   models/baseline_v1/model.keras
        #
        # current output:
        #   models/baseline_v1_conversion/model.keras
        model.save(
            str(paths.keras)
        )

        print(
            "[main] Loaded Keras model saved to the current run folder:"
        )
        print(f"  {paths.keras}")

        # There are no training epochs or training history in this mode.
        history_data = {}
        training_validation_metrics = None

    else:
        model, history, training_validation_metrics = train_cnn_model(
            model=model,
            train_ds=pipelines["train"],
            val_ds=pipelines["val"],
            cfg=train_cfg,
            paths=paths,
        )

        history_data = history.history

    # ---------------------------------------------------------
    # 2. Evaluate the trained Keras model
    # ---------------------------------------------------------

    validation_results = evaluate_keras_model(
        model=model,
        dataset=pipelines["val"],
        class_names=class_names,
        split_name="validation",
    )
    
    # train_cnn_model() normally provides these values.
    # In skip-training mode, take them from the evaluation performed above.
    if training_validation_metrics is None:
        training_validation_metrics = validation_results["keras_metrics"]

    test_results = evaluate_keras_model(
        model=model,
        dataset=pipelines["test"],
        class_names=class_names,
        split_name="test_keras",
    )
    
    metrics = {
        "training_skipped": args.skip_training,
        "training_validation_metrics": training_validation_metrics,

        # Empty when --skip-training is used.
        "training_history": history_data,

        "validation_keras": validation_results,
        "test_keras": test_results,
    }
    
    save_json(paths.metrics, metrics)

    # ---------------------------------------------------------
    # 3. Quantize and calibrate the Keras model
    # ---------------------------------------------------------

    model_for_quantization = clone_without_regularizers(
        model=model,
    )

    quantized_model = quantize_and_calibrate(
        model=model_for_quantization,

        # Quantization must use the raw, unaugmented training data.
        raw_train_ds=raw_datasets["train"],
        cfg=quantize_cfg,
        seed=dataset_cfg.seed,
    )

    pre_qat_test_results = evaluate_keras_model(
        model=quantized_model,
        dataset=pipelines["test"],
        class_names=class_names,
        split_name="test_quantized",
    )
    
    # ---------------------------------------------------------
    # 4. Quantization-aware fine-tuning
    # ---------------------------------------------------------

    quantized_model, qat_history, qat_validation_metrics = (
        fine_tune_quantized_model(
            quantized_model=quantized_model,
            train_dataset=pipelines["train"],
            val_dataset=pipelines["val"],
            cfg=quantize_cfg,
        )
    )

    # Evaluate the final QAT model.
    post_qat_test_results = evaluate_keras_model(
        model=quantized_model,
        dataset=pipelines["test"],
        class_names=class_names,
        split_name="test_quantized_after_qat",
    )
    
    metrics["test_quantized_before_qat"] = pre_qat_test_results
    metrics["qat_validation_metrics"] = qat_validation_metrics
    metrics["test_quantized_after_qat"] = post_qat_test_results

    if qat_history is not None:
        metrics["qat_history"] = qat_history.history
        
    save_json(paths.metrics, metrics)

    # ---------------------------------------------------------
    # 5. Convert the quantized model to Akida
    # ---------------------------------------------------------

    akida_model = convert_to_akida(
        model_quantized=quantized_model,
        cfg=akida_cfg,
    )

    akida_test_results = evaluate_akida_model(
        model=akida_model,
        dataset=pipelines["test"],
        class_names=class_names,
        split_name="test_akida",
        max_samples=args.max_akida_samples,
        batch_size=256,
    )
    
    metrics["test_akida"] = akida_test_results
    save_json(paths.metrics, metrics)

    # Optionally display one test-set prediction.
    if args.show_prediction:
        show_single_akida_prediction(
            model=akida_model,
            dataset=pipelines["test"],
            class_names=class_names,
            # index=args.prediction_index,
            top_k=5,
            show_image=True,
        )

    print("\n" + "=" * 70)
    print("Training pipeline completed successfully")
    print("=" * 70)
    print(f"Keras model: {paths.keras}")
    print(f"Akida model: {paths.fbz}")
    print(f"Configuration: {paths.config}")
    print(f"Metrics: {paths.metrics}")


if __name__ == "__main__":
    main()