### Architectural Flow:
#### 1. Initialization
- Load the Akida model into the physical hardware.
- Initialize the Raspberry Pi camera, set its resolution, and let it warm up for 2 seconds (sensors need time to adapt to light).
- Create a blank csv on the SD card to log flight data.
#### 2. Capture
- The camera snaps a high-resolution photo.
- Save a copy of the raw photo to the SD card.
#### 3. Preprocess
- Pass the photo to the `prepare_flight_batch()` function.
#### 4. Hardware Inference
- Pass the 4D batch resulted from preprocessing into `model_akida.predict_classes()`
#### 5. Aggregation
- Map the integers from the inference (predictions) to string names (e.g. 2 = Forest, 8 = Highway).
- Count them up and make a summary.
- Append the summary in the csv log file.
- Wait X seconds then repeat.