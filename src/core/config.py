from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class DatasetConfig:
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
    resize_to: Optional[tuple[int, int]] = None

    # Training pipeline params
    batch_size: int = 64
    shuffle_buffer: int = 10_000
    cache: bool = True
    augment: bool = True

    # Dataset download directory (optional)
    data_dir: Optional[Path] = None


@dataclass
class ModelConfig:
    base_filters: int = 32
    dense_units: int = 128
    weight_decay: float = 1e-4
    
    # Akida 1.0 requires a bounded ReLU.
    # ReLU6 is supported and is commonly used in Akida-compatible models.
    relu_max_value: float = 6.0

@dataclass
class TrainConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # "adam" or "sgd"
    label_smoothing: float = 0.0

    early_stop_patience: int = 7
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

    use_tensorboard: bool = False
    
@dataclass
class QuantizeConfig:
    # Quantization scheme
    input_weight_bits: int = 8
    weight_bits: int = 4
    activation_bits: int = 4
    
    # Akida 1.0 supports per-tensor activation quantization.
    per_tensor_activations: bool = True

    # Calibration (BrainChip recommended defaults)
    num_samples: int = 1024
    batch_size: int = 100
    epochs: int = 2

    # Data preprocessing must match what the model expects
    resize_to: Optional[tuple[int, int]] = None  # must match your pipeline if used
    
    # Used when compiling the quantized model before QAT/evaluation.
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # "adam" or "sgd"
    
    # ---------------------------------------------------------
    # Quantization-aware fine-tuning
    # ---------------------------------------------------------

    # Set False to perform post-training quantization without QAT.
    qat_enabled: bool = True

    # QAT should normally be shorter than the original float training.
    qat_epochs: int = 5

    # QAT should use a lower learning rate than float training.
    qat_learning_rate: float = 1e-4

    # Stop when quantized validation accuracy no longer improves.
    qat_early_stop_patience: int = 2

    # Reduce the QAT learning rate when validation loss stagnates.
    qat_reduce_lr_patience: int = 1
    qat_reduce_lr_factor: float = 0.5
    qat_min_lr: float = 1e-6
    
@dataclass
class AkidaConvertConfig:
    # Where to save the converted Akida model (".fbz" will be added if missing).
    save_path: Optional[Path] = None

    # Target Akida version context for conversion ("v2" default).
    # CNN2SNN supports setting this context via set_akida_version. :contentReference[oaicite:1]{index=1}
    akida_version: str = "v1"  # "v1" or "v2"

    # Optional: map the converted model to a device (if present).
    map_to_device: bool = False
    hw_only: bool = True  # map only hardware-compatible sequences (if mapping)
