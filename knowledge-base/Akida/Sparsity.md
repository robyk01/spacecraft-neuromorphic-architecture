BrainChip's Akida is just an accelerator on Convolutional Neural Networks (CNNs). Standard CNNs show high levels of sparsity (many of the values involved in the computations are 0) so Akida takes advantage by skipping unnecessary operations and improve efficiency.

![[Standard CNN vs Akida.png]]

On the left we can see a standard CNN processing all values. On the right, Akida hardware skips values of 0 thus leading to better efficiency (lower latency, the output is unchanged)

![[Figures/Sparsity Representation.png|697]]

In this figure we can see the representation of sparsity on a CNN. The black dots indicate zeros. 

### Types of sparsity

- **Activation sparsity**: outputs of previous layers in a neural network are called activations. Akida exploits this type of sparsity because commonly used activation functions like "ReLU" are bringing negative values to 0, which indicates a high sparsity in models.
- **Input sparsity**: the inputs to the first layer must be considered a special case. Akida does not need this kind of sparsity to do well because the activation sparsity described above will work between layers in the model regardless of the initial input.
- **Weight sparsity**: this refers to zeros in the model weight. These will arise naturally during training and are typically unstructured (have no spatial pattern within the weights matrices)

The main distinction is that input sparsity comes from the data itself, weight sparsity is a built-in characteristic of the model, and activation sparsity (also called event sparsity) is a dynamic trait of the model's activations, varying with the input from one sample to another

More informations about sparsity here: https://brainchip.com/akida-exploits-sparsity-for-low-power-in-neural-networks/

