from __future__ import annotations

import argparse
import json

from dataclasses import fields
from pathlib import Path
from typing import Any

# Import dataset.py before the other TensorFlow-dependent modules.
from .core.dataset import (
    build_tfdata_pipelines,
    load_eurosat_tfds,
    set_seeds,
)

from .core.akida_model import (
    evaluate_akida_model,
    load_akida_model,
    map_akida_model_to_device,
    show_single_akida_prediction,
)
from .core.config import DatasetConfig
from .core.paths import ModelPaths


def parse_arguments() -> argparse.Namespace:
    """
    Read command-line options for .fbz evaluation.

    Examples:

        Test the model belonging to one run:

            python -m landClassification.test_fbz ^
                --run-name baseline_v1

        Test a directly specified .fbz file:

            python -m landClassification.test_fbz ^
                --model models/baseline_v1/model.fbz
    """
    parser = argparse.ArgumentParser(
        description=(
            "Load an Akida .fbz model and evaluate it "
            "on the EuroSAT test dataset."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="eurosat_baseline",
        help=(
            "Model run folder inside models/. "
            "Ignored as a model location when --model is supplied."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Optional direct path to a .fbz model. "
            "When omitted, models/<run-name>/model.fbz is used."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional override for the TensorFlow dataset batch size. "
            "Otherwise, the saved training configuration is used."
        ),
    )

    parser.add_argument(
        "--akida-batch-size",
        type=int,
        default=256,
        help="Akida inference batch size. Default: 256",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optionally evaluate only the first N test samples. "
            "By default, the complete test set is used."
        ),
    )

    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help=(
            "Optional image-size override. This is mainly useful when "
            "testing an external .fbz without a saved config.json."
        ),
    )

    parser.add_argument(
        "--show-prediction",
        action="store_true",
        help="Display one prediction after evaluation.",
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

    parser.add_argument(
        "--map-device",
        action="store_true",
        help=(
            "Map the model to the first connected Akida device. "
            "Without this option, the PC software backend is used."
        ),
    )

    return parser.parse_args()


def load_saved_dataset_config(
    config_path: Path,
) -> DatasetConfig:
    """
    Load the DatasetConfig previously saved by train.py.

    If the configuration file does not exist, use the standard defaults.
    """
    if not config_path.exists():
        print(
            "[load_saved_dataset_config] No config.json found."
        )
        print(
            "[load_saved_dataset_config] "
            "Using the default DatasetConfig."
        )

        return DatasetConfig()

    print(
        "[load_saved_dataset_config] Loading configuration from: "
        f"{config_path}"
    )

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as input_file:
        saved_configuration: dict[str, Any] = json.load(
            input_file
        )

    saved_dataset_values = saved_configuration.get(
        "dataset",
        {},
    )

    # Only pass known DatasetConfig fields to the constructor.
    # This prevents unrelated JSON fields from causing an error.
    valid_field_names = {
        field.name
        for field in fields(DatasetConfig)
    }

    saved_dataset_values = {
        name: value
        for name, value in saved_dataset_values.items()
        if name in valid_field_names
    }

    # JSON stores tuples as lists.
    if saved_dataset_values.get("resize_to") is not None:
        saved_dataset_values["resize_to"] = tuple(
            saved_dataset_values["resize_to"]
        )

    # JSON stores pathlib.Path objects as strings.
    if saved_dataset_values.get("data_dir") is not None:
        saved_dataset_values["data_dir"] = Path(
            saved_dataset_values["data_dir"]
        )

    return DatasetConfig(
        **saved_dataset_values
    )


def main() -> None:
    args = parse_arguments()

    paths = ModelPaths(
        run_name=args.run_name,
    )

    if args.model is None:
        model_path = paths.fbz
        config_path = paths.config
    else:
        model_path = args.model

        # For a directly specified model, look for config.json in the
        # same folder as the .fbz file.
        config_path = model_path.parent / "config.json"

    dataset_cfg = load_saved_dataset_config(
        config_path=config_path,
    )

    # The test program does not need training augmentation.
    dataset_cfg.augment = False

    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError(
                "--batch-size must be greater than 0."
            )

        dataset_cfg.batch_size = args.batch_size

    if args.resize is not None:
        dataset_cfg.resize_to = tuple(
            args.resize
        )

    print("=" * 70)
    print("EuroSAT Akida model evaluation")
    print("=" * 70)
    print(f"Model:        {model_path}")
    print(f"Config:       {config_path}")
    print(f"Image size:   {dataset_cfg.resize_to}")
    print(f"Max samples:  {args.max_samples}")
    print(f"Map device:   {args.map_device}")

    set_seeds(
        dataset_cfg.seed
    )

    raw_datasets, dataset_info = load_eurosat_tfds(
        dataset_cfg
    )

    pipelines = build_tfdata_pipelines(
        datasets=raw_datasets,
        cfg=dataset_cfg,
    )

    class_names = list(
        dataset_info.features["label"].names
    )

    akida_model = load_akida_model(
        model_path=model_path,
    )

    # Mapping is optional. When it is not requested, the loaded .fbz runs
    # through the Akida software backend on the PC.
    if args.map_device:
        map_akida_model_to_device(
            model=akida_model,
            hw_only=True,
        )

    results = evaluate_akida_model(
        model=akida_model,
        dataset=pipelines["test"],
        class_names=class_names,
        split_name="test_akida",
        max_samples=args.max_samples,
        batch_size=args.akida_batch_size,
    )

    show_single_akida_prediction(
        model=akida_model,
        dataset=pipelines["test"],
        class_names=class_names,
        # index=args.prediction_index,
        top_k=5,
        show_image=True,
    )

    print("\n" + "=" * 70)
    print("Akida evaluation completed")
    print("=" * 70)
    print(
        "Accuracy: "
        f"{results['overall_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()