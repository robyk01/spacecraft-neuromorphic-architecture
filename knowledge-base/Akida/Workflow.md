### Neural Network Layers

==```keras.layers.Input(shape=(28, 28, 1), name="input", dtype=tf.uint8)```==
Defines what can enter the network.
- shape: specifies that the network expects a 2D grid of 28x28 pixels with 1 color channel (grayscale).
- dtype: forces the input to be unsigned 8-bit integers (0 to 255). Akida hardware input requires 8-bit integer formats to map raw pixel signals effectively.

==```keras.layers.Rescaling(1. / 255)```==
Scales down the raw pixel integer values into a floating-point range between 0.0 and 1.1. Normalizing data like this keeps the math stable during initial training.

==```keras.layers.Conv2D(filters=32, kernel_size=3, strides=2)```==
A standard 2D Convolution layer. It slides mathematical weight matrices across the pixel grid to extract fundamental spatial features like edges, lines, or sharp contrasts.
- filters: the layer will create 32 distinct filters. Each filter trains itself to find a unique visual pattern, resulting in 32 separate feature map channels.
- kernel_size: the sliding grids are 3x3 matrices of weights.
- strides: the kernel jumps 2 pixels at a time instead of moving pixel by pixel.

==```keras.layers.BatchNormalization()```==
Standardizes the outputs of the previous layer so that they have a mean close to 0 and a standard deviation close to 1.
Placing BatchNormalization immediately after a convolution layer is a strict rule for the cnn2snn compiler.

==`keras.layers.ReLU()`==
Rectified Linear Unit activation function. It passes any positive value straight through but squashes any negative values down to 0.
By changing negative numbers to zero, it introduces high [[Sparsity|Activation Sparsity]]. Because Akida doesn't process zeros, a high volume of ReLU zeros translates directly to low power draw on the hardware.

#### Separable layers
These two layers form a **Depthwise Separable Convolution**. This architecture splits standard convolution into two cheaper steps, making it perfect for resource-constrained edge systems and [[FPGA]] acceleration.

==``keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', strides=2)``==
Instead of mixing all channels together, this layer filters each channel completely independently using a localized 3x3 window.
- strides: scales down the resolution to cut processing size (same as before)
- padding: pads the borders with zeros so the spatial dimensions scale evenly

==`keras.layers.Conv2D(filters=64, kernel_size=1, padding='same')`==
It takes those independent channels from the depthwise step and use a tiny 1x1 window to blend them together into 64 brand new feature combinations (filters=64)
This combination achieves practically the same feature extraction power as a standard heavy Conv2D layer, but uses a fraction of the parameters and memory.

==`keras.layers.BatchNormalization()` & `keras.layers.ReLU()`==
Stabilizes and introduces activation sparsity to the outputs of our separable convolution block before passing it to the final classifier.

==`keras.layers.Flatten()`==
Unrolls the multi-dimensional spatial grid block (Height x Width x Channels) into a single, flat, continuous 1D vector of numbers.

==`keras.layers.Dense(10)`==
A fully connected (dense) layer. Every single number from our flattened vector connects to every single one of the 10 output nodes. Each node represents a final category score.
- 10: The number of target output categories (MNIST example)

### Training
In traditional ML, when we initialize models they already know their mathematical identity (how to calculate splits, what their internal loss function is, and how its solver updates). In contrast, Neural Networks are just empty architectural shells when we define the layers. They have weights and structure, but they don't know:
- **Loss function**
- **Optimizer**
- **Metrics**

```python
model_keras.compile(

	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

	optimizer=Adam(learning_rate=1e-3),

	metrics=['accuracy']

)
```

Calling `.compile()` is the step where we build the underlying computation graph, binding the network layers to its optimizer and loss math. Without compiling, the `.fit()` function will not know how to run backpropagation.

#### The Loss Function
It measures how far off the network's guess was from the actual answer.
- **Sparse**: this means the ground-truth labels (y_train) are provided as simple integers rather than binary one-hot encoded vectors. It saves massive amounts of memory.
- **Categorical Crossentropy**: the standard formula used when a network has to choose between multiple distinct categories.
- **from_logits**: this tells the loss function that our final Dense(10) layer outputs raw, unscaled real numbers (called logits), rather than formatted probabilities between 0 and 1.

#### The Optimizer
It calculates gradients and actually updates the weights inside your network layers to reduce loss score.
- **Adam** (Adaptive Moment Estimation): is the industry standard because it dynamically adjusts the learning speed for each individual weight as it trains.
- **learning_rate**: this dictates the step size the optimizare takes down the mathematical slope.

#### The Metrics
While the optimizer cares about minimizing a complex cross-entropy loss number, this tells Keras to print out a clean, readable percentage of how many images the model classified correctly.


```python
_ = model_keras.fit(x_train, y_train, epochs=10, validation_split=0.1)
```
Iterates over batches and epoch, running forward and backward passes to find patterns.
- **x_train, y_train**: training data inputs and their corresponding true labels
- **epochs**: one epoch means the network has processed the entire dataset forward and backward exactly once
- **validation_split**: this carves out exactly 10% of the training data and locks it away. The model trains exclusively on the remaining 90%. At the exact end of every epoch, the model runs a test pass on this hidden 10% to compute a validation accuracy


```python
score = model_keras.evaluate(x_test, y_test, verbose=0)
```
A helper function that automatically runs predictions and outputs final loss/accuracy scores.
1. The function takes the unlabeled input data (`x_test`) and runs a forward inference pass through every single layer of the network. This generates a raw prediction vector for every image.
2. It immediately take those predictions and compares them mathematically against the true ground-truth labels (`y_test`) using the exact loss and metrics formulas locked during `.compile()`.
3. It then discards the predictions to save memory and only returns the final calculations: the overall Loss value and the Accuracy percentage.

##### Difference between `.predict()` and `.evaluate()`

| Command                          | Parameters              | Utility                                                                | Returns                    |
| -------------------------------- | ----------------------- | ---------------------------------------------------------------------- | -------------------------- |
| `model.predict(x_test)`          | Only inputs             | Runs the forward pass to generate raw guesses                          | The raw guesses themselves |
| `model.evaluate(x_test, y_test)` | Inputs and truth labels | Runs the forward pass, then scores the results against the true labels | Performance metrics        |

### Quantization
### Convertion
