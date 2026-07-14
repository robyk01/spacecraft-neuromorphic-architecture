### Architectural Flow:
#### 1. Initialization
The model initialization happens inside a Docker container, which is granted hardware access via the `--device=/dev/akida0` flag. The Akida kernel module itself is handled by host system commands (`sudo insmod ./akida-pcie.ko`) before Docker starts.
The camera code is decoupled from the inference script. The camera runs on the host Raspberry Pi OS, snapping photos and automatically dropping them into `/home/synapse2025/Desktop/akida_sample` folder.
#### 2. Preprocess
Pass the photo to the `prepare_flight_batch()` function. The script will use Pillow to load the image file from `/data/`, convert it to a NumPy array, andpass it directly into `prepare_flight_batch(image)` to yield the 4D tensor stack.
#### 3. Hardware Inference
Pass the 4D batch resulted from preprocessing into `model_akida.predict_classes()`. The inference must be wrapped in high-precision time deltas.
#### 4. Output
Here the scripts `infer.py` and `infer_all.py` are execution runs that calculate and print out final performance reports directly to the terminal console. The script needs to calculate and format these exact metrics:
- Total images processed
- Correctly classified patches vs. Incorrectly classified patches
- Global accuracy percentage
- Total processing time (seconds)
- Average time per image (milliseconds)
- Total frames per second
