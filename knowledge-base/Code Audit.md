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
- Pointwise: Looks across the channels to combine them into deeper features.
This split drastically reduces computational complexity and parameter counts because Akida's NPUs are heavily optimized to execute DW and Separable convolutions natively at ultra-low power levels.

### Train CNN model

### Evaluate model

### Quantize + calibrate with QuantizeML

### Convert to Akida with CNN2SNN

### Evaluate Akida model

### main()