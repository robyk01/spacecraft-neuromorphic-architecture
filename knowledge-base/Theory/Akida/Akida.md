The main difference between Akida and other machine learning networks is that inputs and weights are integer tensors (Numpy arrays) and it only performs integer operations, so that it can further reduce the power consumption and memory footprint. Since [[QuantizeML|quantization]] and ReLU activation functions lead to substantial [[Sparsity||sparsity]], Akida takes advantage of this by implementing operations in biologically inspired event-based calculations.

Akida is an **ASIC** (Application-Specific Integrated Circuit) - a neuromorphic processor.
BrainChip took the mathematical blueprint of an [[SNN]] (synapse, membrane, integration, thresholds, firing) and physically wired the microchips to work that way.

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

### Inference
###### 1. Pre-Processing
Before anything touches the neuromorphic chip, the Raspberry Pi's processor handles the standard data prep:
- The python script takes the large photo captured by Camera Module 2 and crops out a single 64x64x3 RGB patch
- At this exact moment, that patch is just a standard Numpy array sitiing in the Pi's system RAM. It contains 12,288 individual numbers (ranging from 0 to 255)
- When the script calls `model.predict(patch)`, the Pi packages those 12,288 integer numbers and fires them across the PCIe connection directly into the on-board memory of the Akida hardware board.
###### 2. Pixel-to-Spike Conversion
Once the array lands inside the Akida chip, standard computer arithmetic stops completely. The Akida input hardware converts those static numbers into physical time-based eletrical pulses.
The chip establishes an internal micro-timeline consisting of a fixed number of Clock Steps
Instead of reading the image at all once, the input hardware reads the brightness of the pixels to determine when each pixels fires an electrical pulse alog the silicon wires:
- Pure white / ultra-bright pixels (e.g. 250): The hardware fires an eletrical pulse (1) at Clock Step 0 (immediately)
- Medium gray / mid-tone pixels (e.g. 125): The hardware delays the signal and fires an eletrical pulse (1) at Clock Step 2
- Pure Black / Dark Pixels (e.g. 0): The hardware emits nothing (0) across the entire timeline. The physical wire remains completely silent.
Our static 64x64 image has now been transformed into a 4-frame movie of digital electrical pulses streaming across time.
###### 3.  Integrate and Fire
Now that the image is a stream of pulses moving through time steps, those pulses travel along microscopic physical wires into Akida's neural processing cores (the quantized convolutional layers).

Here is what happens inside the physical transistors during each clock step:
**Clock Step 0**: Early spikes -> Synapse (+5 Weight) -> Membrane -> Fires
**Clock Step 1**: Mid spikes -> Synapse (+2 Weight) -> Membrane
**Clock Step 2**: Late spikes -> Synapse (No event) -> Core powers down

Only the pulses from the brightest pixels arrive at the first layer of hardware neurons.
When a pulse hits a silicon synapse, the hardware reads the stored 4-bit or 8-bit integer weight and adds it directly into that neuron's internal hardware register. If that +5 pushes the register above is pre-programmed threshold, the transistor circuit instantly fires a new electrical pulse deeper into the next layer of the chip and resets its register back to zero. The dimmer pixels finally send their pulses, triggering further additions across the networks.
When dark pixels send no pulses the receiving transistors perform zero math. They do not multiply by zero, the logic gates literally remain unlocked, consuming zero electrical power (Sparsity)
###### 4. The Scoreboard
At the very end of the silicon pipeline sit the output neurons - one for each EuroSAT classification category. The output nodes act as dedicated digital counters standing open across the Clock Steps:
- As features (like straight lines representing a highway or dense green textures representing a forest) are recognized by the earlier layers, cascades of pulses cascade down into the final output nodes.
- If the patch contains a highway the **Highway Output Node** receives a rapid barrage of incoming pulses across the clock steps. Its internal integer counter rapidly ticks upward: +10, +22, +35.
- The **Forest Output Node** receives mostly negative or inhibiting pulses, keeping its score buried near zero or below.
###### 5. Returning the Verdict
Once the final clock step ticks to a close, the Akida hardware halts the simulation. The chips reads the final accumulated scores sitting in the output registers, sees that the HighWay register holds the highest integer score and sends a final array of scores back across the PCIe bus to the Raspberry Pi CPU. The scripts receives the results, tags that specific 64x64 patch as "Highway" and immediately grabs the next patch from the camera image to repeat the cycle.

[Input Buffer] ──> [Pixel-to-Spike Gate] ──> [Silicon Layer 1] ──> [Silicon Layer 2]
 (Raw 64x64x3)          (Clock Steps 0-3)             (32 Filter Sheets)           (64 Filter Sheets)
NO FEATURES              NO FEATURES                    FEATURE MAPS              FEATURE MAPS