High-level analysis of existing code and functions. 
The goal of this repository is to take standard EuroSAT RGB satellite tiles (64x64x3), train a hardware-efficient neural network, quantize it to integers, and convert it into a deployable Akida .fbz hardware file.

### Table of Contents
1. [[#Configuration]]
2. [[#Reproducibility]]
3. [[#Load EuroSAT from TFDS]]
4. [[#Build tf.data pipelines]]
5. [[#Build baseline CNN model]]
6. [[#Train CNN model]]
7. [[#Evaluate model]]
8. [[#Quantize + calibrate with QuantizeML]]
9. [[#Convert to Akida with CNN2SNN]]
10. [[#Evaluate Akida model]]
11. [[#main()]]

### Configuration
This part creates dataclasses for configs (like PipelineConfig, TrainConfig, QuantizeConfig, AkidaConvertConfig).

**Pipeline**:
- The training split ratio is 75-15-15 (Train-Validation-Test)
- Parameters count is 64
**Train**:
- Epochs count is 30
- Learning rate is 1e-3
- For optimizer we will use Adam
**Quantize**:
- The size of quantization is 8x8x8
- Number of samples is 1024
- Batch size is 100
- Epochs count is 2
**Akida Converter**:
- Akida version is v1
- Mapping to device is set to False
- Hardware only is set to true

### Reproducibility
Set seeds for Python, Numpy, and TensorFlow

### Load EuroSAT from TFDS
Load EuroSAT from TensorFlow Datasets and create deterministic splits from the 'train' split.

Because TFDS doesn't organize images in folders, it dumps all of them into a massive folder "train", so we need to slice it for the split.

```python
train_split = f"train[:70%]"
val_split   = f"train[70%:85%]" 
test_split  = f"train[85%:]"
```

We load the data with `tfds.load()`.
### Build tf.data pipelines
Build efficient `tf.data` pipelines.

```python
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
```

**`.map(preprocess)`**
Runs the `preprocess` function on every image individually across multiple CPU cores simultaneously (`num_parallel_calls=autotune`). This is the first function to be called because it casts images to uint8 and resizes them if requested, ensuring every image entering the rest of the pipeline has the exact same spatial dimensions.

**`.map(augment_train)`**
Applies random transformations (flips, rotations, brightness shifts) to generate synthetic variations of the images. Only works in training because we never augment the validation or test sets.

**`.cache()`**
The first time epoch 1 runs, TensorFlow reads the images from disk, preprocesses them, and saves them directly into the computer RAM. For epochs 2 through 30, the computer never touches the hard disk again. It reads data instantly from the RAM, making training run exponetially faster.

**`.shuffle(buffer_size=10_000)`**
Pulls 10,000 images into a memory pool, randomly grabs one, passes it forward, and refills the empty spot with a new image from the disk. This is an important step because neural networks learn patterns based on the order the data arrives. If the dataset presented 1,000 images of a forest, then 1,000 images of an highway, the network would forget what a forest looks like while learning highways. Shuffling breaks up sequence bias.

**`.batch(batch_size=64`**
Collects 64 individual 3D image arrays (64x64x3) and stacks them into a single 4D tensor block (63, 64, 64, 3). This block is what gets pushed into the matrix multipliers of the network.

**`prefetch(autotune)`**
Decouples data loading from the hardware inference. While the Akida chip is busy doing math on Batch #1, the CPU is working simultaneously in the background to prepare, augment, and packaget Batch #2. This is also important for a low hardware latency.

The `preprocess()` and `augment_train()` joggles with converting the image from uint8 to fp and back because the photometric math requires decimals and akida hardware requires integers.

### Build baseline CNN model
This section defines the layout of the network before it gets trained or converted into hardware.

###### 1. Input & Rescaling Layer

```python
inputs = tf.keras.Input(shape=input_shape, dtype=tf.uint8, name="input")
x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_u8_to_f32")(inputs)
```

Accepts raw integer images (uint8 from 0 to 255) and immediately divides them by 255.0 inside the model graph to produce floating-point numbers between 0.0 and 1.0.
By keeping the scaling inside the Keras model rather than in the external dataset pipeline, the network explicitely advertises that its input is raw uint8 integers. When QuantizeML later calibrates this model, it can interface directly with the raw pixel integers emitted by the Raspberry Pi camera.

###### 2. Stem Layer (Initial Downsampling)

```python
x = tf.keras.layers.Conv2D(
    filters=base_filters, kernel_size=3, strides=2, padding="same", use_bias=False, ...
)(x)
x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
x = tf.keras.layers.ReLU(name="stem_relu")(x)
```

A standard 3x3 convolution with 32 filters. Strides=2 parameter cuts the spatial height and width of the image in half - from 64x64 to 32x32.
BrainChip's QuantizeML and cnn2snn tools require clear visibility of the activation boundaries so they can replace them with specialized QuantizedReLU layers. Separate layers make quantization converion seamless, thats why we have a dedicated ReLU layer rather than passing `activation="relu"` inside the Conv2D layer.

###### 3. Depthwise-Separable Blocks
Instead of standard convolutions, the rest of the network is built entirely out of a custom helper block called `dw_pw_block`. This block splits standard 3D convolutions into two distinct steps.

```python
def dw_pw_block(x, filters: int, stride: int, block_name: str):
    # Step 1: Depthwise Conv (Spatial patterns only, 1 filter per channel)
    x = tf.keras.layers.DepthwiseConv2D(...)(x)
    ...
    # Step 2: Pointwise Conv (1x1 Conv, mixes channels together)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, ...)(x)
    ...
```

A standard 3x3 convolution mixes spatial edges and channel colors at the same time, requiring massive matrix multiplication. A Depthwise-Separable convolution splits the math:
- Depthwise: Looks only at 3x3 spatial patterns inside each channel individually.
- Pointwise: Looks across the channels to combine them into deeper features. It looks at a single pixels across all channels and multiplies them by own weights, outputting a single value.
This split drastically reduces computational complexity and parameter counts because Akida's NPUs are heavily optimized to execute DW and Separable convolutions natively at ultra-low power levels.

###### 4. Progressive Downsampling

```python
    x = dw_pw_block(x, filters=base_filters * 2, stride=1, block_name="b1")  
    # 32x32
    x = dw_pw_block(x, filters=base_filters * 2, stride=2, block_name="b2")
    # 16x16
    x = dw_pw_block(x, filters=base_filters * 4, stride=1, block_name="b3")
    # 16x16
    x = dw_pw_block(x, filters=base_filters * 4, stride=2, block_name="b4") 
    # 8x8
    x = dw_pw_block(x, filters=base_filters * 8, stride=1, block_name="b5")
    # 8x8
```

By using `strides=2` inside the depthwise layers, the network compresses spatial dimensions while actively learning weights at the same time. We need multiple blocks for the network to not lose critical information and accuracy.
- For the first block (32x32), looks at raw pixels and finds simple edges.
- The second block (16x16), takes those edges and combines them into simple shapes
- At 8x8, it takes the shapes and combines them into different concepts

###### 5. The Classification Head

```python
x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
x = tf.keras.layers.Dense(dense_units, use_bias=False, ...)(x)
...
outputs = tf.keras.layers.Dense(num_classes, name="logits")(x)
```

**Global Average Pooling**: 
Takes the final 8x8 spatial feature grid across all 256 channels and averages each channel into a single number, collapsing the tensor into a flat 256-element vector. At the end of the network, pooling destroys the spatial flow, stripping away useless spatial coordinates and leaving only the pure summary numbers.

**Dense Layer**:
First dense layer is a synthesizer. It looks at the summary numbers and mixes them together to form high-level deductions. For example, if Feature 14 (grey color) and Feature 200 (straight lines) are both high, that strongly indicates a road. It reduces the 256 raw clues into 128 refined logical deductions.
The second layer acts like a jury. It takes the 128 deductions and casts the final vote. It has exactly 10 output nodes. Whichever node gets the highest score from the 128 deductions wins the classification.




### Train CNN model
Compile and train the basine cnn (non-quantized) TF-Keras model.
#### Loss Function

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    label_smoothing=cfg.label_smoothing,
)
```

Mathematical ruler that measures exactly how wrong the model is after every single guess.
In some datasets, labels are one-hot encoded (e.g. `0, 0, 1, 0`). EuroSAT provides labels as single integers (e.g. 2 for Forest). Sparse tells TF to expect raw integers, saving massive amounts of RAM.
Because the last `Dense()` layer outputs logits and not percentages, we use `from_logits=True` to calculate the Softmax function internally (convert network score -> predictions from 0 to 1).

#### Optimizer
The Loss Function calculates the error, but the Optimizer is the engine that actually drives backward through the network (backprop) to turn the weight knobs.
The script defaults to Adam. Adam is the industry standard because it has an adaptive learning rate. It takes big, aggressive steps at the beginning of training, and automatically takes smaller steps as it gets closer to the right answer.

#### Callbacks
###### 1. Early Stopping

```python
tf.keras.callbacks.EarlyStopping(
    monitor="val_acc", patience=7, restore_best_weights=True
)
```

It watches the model's accuracy on then unseen Validation dataset. If the validation accuracy stops improving for 7 epochs in a row (`patience=7`), it kills the training immediately to prevent overfitting. When it aborts, it rolls back the model to the exact epoch where it hit its highest score, throwing away the degraded weights from the final epochs (`restore_best_weights=True`).

###### 2. ReduceLROnPlateau

```python
tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3
)
```

If the model gets stuck in a rut and the validation loss stops dropping for 3 epochs, this callbacks cuts the Learning Rate (the size of the weight updates) strictly in half (`factor=0.5`).

###### ModelCheckpoint

```python
tf.keras.callbacks.ModelCheckpoint(
    monitor="val_acc", mode="max", save_best_only=True,
    save_weights_only=False,   # keep full model for convenience
)
```

Every time the models hits a new high score on the valdiation set, it physically saves a `.keras` file to the hard drive. If the system crashes at epoch 29, we don't lose our work.


### Evaluate model
This section is mostly standard ML reporting (Confusion Matrices, Precision, Recall, F1-scores).
Because the model outputs raw logits, the evaluation script manually applies `tf.nn.softmax(out, axis=-1)` to convert the unbounded scores into clean probabilities before picking the winner. It also runs a strict check (`_prepare_dataset_for_model_input`) to guarantee the evaluation data is formatted exactly the same way as the training data (`uint8`).


### Quantize + calibrate with QuantizeML
This is where the model stops being a standard CNN and starts becoming hardware.

#### Setting the Bit-Width Limits

```python
qparams = QuantizationParams(
    input_weight_bits=cfg.input_weight_bits, # 8
    weight_bits=cfg.weight_bits,             # 8
    activation_bits=cfg.activation_bits,     # 8
)
```

This configuration forces the model into an 8/8/8 quantization scheme. It tells BrainChip's QuantizeML engine to round every single decimal weight, input and activation into a rigid grid of 8-bit integers (values between -128 and +127).

Because we can't just blindly round decimals to integers (the network's accuracy will collapse), we tell the engine the normal operating range of network.

```python
samples = _extract_calibration_samples(raw_train_ds, num_samples=1024)
model_quantized = quantize(model_cnn, qparams=qparams, samples=samples, ...)
```

The script grabs 1,024 real, unaugmented satellite photos and passes them through the floating-point network. As these images flow through the layers, the QuantizeML engine watches the math. It records the minimum and maximum values that naturally occur at every layer. By knowing these exact ranges, it can intelligently scale and clip the floating-point numbers before rounding them to integers, preserving almost 100% of the model's accuracy.

### Convert to Akida with CNN2SNN
Now we need to translate the quantized Keras model into a Spiking Neural Network for the physical silicon.

```python
model_akida = convert(
    model_quantized,
    file_path=cfg.save_path, 
    input_scaling=None
)
```

This single `convert()` function is the bridge between software and hardware. It takes the Depthwise-Separable blocks and maps them to the exact configuration of the Akida Neural Processing Units (NPUs). The compiler looks at the integer weights and calculates the exact firing thresholds needed for the hardware's integrate-and-fire buckets to perfectly mimic the Keras model's outputs. It packages the entire architecture, the integer weights, and the firing thresholds into a single, highly compressed binary file.


### Evaluate Akida model
Throughout the entire training process, the data was flowing through a high-speed `tf.data.Dataset` pipeline. However, Akida runtime is a C++ hardware driver under the hood, it has no idea what a TF Dataset is. Before running hardware inference, the `_collect_numpy_from_tfdata` script manually loops through the TF dataset, unpacks the images, and stitches them into a giant standard Numpy array in the system's RAM.

### main()
In the final `evaluate_akida_model` call:

```python
batch_size=256 # Akida processes inputs in chunks; tune for your RAM
```

When predicting on a Keras software model, batch sizes are mostly about GPU speed. On Akida, it is about physical SRAM (Static RAM) on the chip. The chip can only hold a certain number of images in its input buffer at one time. If we try to pass 10,000 Numpy images to the chip at once without setting batch_size, it will physically overflow the board's memory and crash.

**The Verification Strategy**:
- `evaluate_model(model)` $\rightarrow$ Baseline floating-point accuracy.
- `evaluate_model(model_quant)` $\rightarrow$ 8-bit integer accuracy.
- `evaluate_akida_model(model_akida)` $\rightarrow$ Hardware neuromorphic accuracy.
If we launch the payload and the accuracy is terrible, we won't know whose fault it is (either the CNN, QuantizeML, hardware conversion). By running this waterfall test, we rpove mathematically that the accuracy survived the jump from floats -> integers -> silicon. If Step 1 gets 95%, Step 2 gets 94%, and Step 3 gets 94%, you know the conversion is mathematically flawless.