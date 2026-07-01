The main difference between Akida and other machine learning networks is that inputs and weights are integer tensors (Numpy arrays) and it only performs integer operations, so that it can further reduce the power consumption and memory footprint. Since [[QuantizeML|quantization]] and ReLU activation functions lead to substantial [[Sparsity||sparsity]], Akida takes advantage of this by implementing operations in biologically inspired event-based calculations.

The sequence is:
1. Create and train: Using TensorFlow/Keras or Akida model zoo.
2. Quantize: Using [[QuantizeML]] toolkit.
3. Convert: Using [[CNN2SNN]] toolkit.
### Akida Model

The Model object has basic features such as:
- **summary()**: prints a description of model architecture
- **save()**: needs a path for the model and allows saving to disk for future use
- **forward**: generate the outputs for a specific set of inputs
- **predict**: similar to the forward method, but is specifically designed to replicate the float outputs of a converted CNN
- **statistics**: provides relevant inference statistics

See an example of Akida [[Workflow]].