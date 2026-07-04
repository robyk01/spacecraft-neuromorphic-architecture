Core objective is evaluating neuromorphic architectures implemented on [[FPGA|FPGAs]] for real-time data processing. We will use BrainChip's [[Akida]] MetaTF environment to handle the software side: quantizing and compiling SNNs so they can run efficiently inside the FPGA IP core.

The project will be divided into two distrinc chronological chapters: 
### 1. The Factory
###### The Standard CNN
We train a normal Keras [[CNN]] using floating-point math. It learns the visual features of EuroSAT satellite images (highways, forests, crops)
###### The Quantization
We pass that Keras model to [[QuantizeML]]. It converts the continuous floating-point weights into discrete 8-bit or 4-bit integers (like +5, -2, +1)
###### The SNN Convertion
We pass the quantized model to [[CNN2SNN]] convert function. This compiler looks at the integer weights and calculates the exact firing threshold needed for an integrate-and-fire network to mimic your CNN.
###### The Output
The compiler bundles up the architecture, integer weights, and firing thresholds into a single hardware configuration file (fbz)

### 2. The Flight
###### Flashing the Hardware
When the python script starts on the Raspberry Pi, it loads the .fbz file into the Akida hardware board. This instantly configures the physical transistors: it sets the weights in the silicon synapses (+5, -2) and programs the thresholdlevels in the silicon buckets.
###### The Fragmentation
The camera takes a photo, and the Pi CPU slices out a 64x64 patch.
###### The Inference
That 64x64 patch enters the Akida chip. The chip's input gate converts the pixels into electrical pulses across 4 clock cycles, the silicon registers and integrate the integer weights, the nodes fire when overflowing, and the winning class score is handed right back to the Raspberry Pi.