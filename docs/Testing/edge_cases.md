## Edge Cases

### 1. Boot & Environment
Things to consider:
- how the Pi boots at -45/-50 C
- kernel modules loading

#### Overview:
1. ESP32 (OBC) fires first (under 50 milliseconds). It activates the hardware/software watchdogs, initializes sensor loops, and handles telemetry. If the rasp pi dies, the ESP keeps downlinking telemetry to the ground station.
2. Rasp Pi takes around 20-30s too boot. It has to initialize the filesystem and hardware buses of the Linux OS
By default, Linux has no idea what Akida is so we inject the driver `akida-pcie.ko` into the running Linux kernel, creating a hardware interface file in the system `/dev/akida0`. The hardware communicates with the SDK through reading and writing of bytes.

#### Risks
1. **Temperature Bug**
In Synapse\subsystems\obc\esp32-obc\main\tasks\mode_manager_task.c:

```cpp
bool is_critical = (battery < 3.2f || temp.temperature_external_c > 70.0f || temp.temperature_external_c < -10.0f);

if (is_critical) return MODE_SAFE;
```

In the stratosphere, external air temperature drops to -40 to -60 C. During ascent, the ESP32 will think the spacecraft is dying and permanently trigger Safe Mode, disabling `photo_active = false` and aborting camera/inference ops.

**Recommendation**: The threshold should be modified for internal temp instead.

Source: documentation
"Safe Mode - Any temperature sensor > 50◦C, or Vbat < threshold for 10s"

2. **Missing Processing Command**
In Synapse\subsystems\obc\rpi\comm\testint_comm.py and Synapse\subsystems\obc\esp32-obc\main\tasks\mode_manager_task.c:

Right now neither scripts know about the `run_entire_data_processing_mode.py`. When balloon is in active descent `mode_manager_task.c` should launch a CMD|START_PROCESSING over UART to launch the 4-step AI inference pipeline. `testint_comm.py` receives the command in rasp pi and launches the python script.

Also OBC side is missing the radio downlink pipeline to receive and broadcast the classification results.

Source: documentation
"Data Processing Mode- Free fall detection (9.8 m/s2) + GPS + Barometer."


3. **Thermal Risk** 
Unless there is a hardware relay that is missing from the software, the Raspberry Pi is hardwired to main power and remains powered on the entire flight, only the software remaining off/idle.

If the raspberry pi is kept powered off between 50m and 25km, the components in the payload might cold-soak. When the ESP32 switches power on to the Pi at 25km:
- MicroSD flash memory controllers might refuse to initialize or write below -20 C
- The onboard oscillator crystal can drift, corrupting PCIe and UART bus clocks

**Recommendation**: A cold-box freezer boot test (powering pi on after it has sat in a freezer for 1 hour) can be performed.

If only the flight software is kept off/idle Linux will consume around 3 Watts which will act as a steady electrical heater, preventing the bay, batteries, and SD card from reaching -40 C. **In this case the battery capacity needs to be checked** to cover the energy used (3W at 5V is 600mA, over 2.5 hrs of flight that is around 1500 mAH) plus the Akida inference burst.

### 2. In-Flight Image Aquisition
Things to consider:
- auto exposure at high altitude
- motion blur when swinging
- check SD storage

#### Overview:
Hardware: Raspberry Pi Camera Module v2 (Sony IMX219 sensor, 3280 × 2464 resolution)
1. Runs auto_select_shutter() at startup to pick the sharpest shutter speed and analog gain.
2. Enters a while True continuous capture loop, saving a photo every 2 seconds to ~/Desktop/foto_mode/poze/.
3. Appends photo metadata, camera settings, and GPS coordinates to log_pozitii.csv.

#### Risks

1. **Auto Exposure**
In `Synapse\subsystems\payload\src\Code on Rasp Pi\foto_mode\script.py`:
`auto_select_shutter()` is ran once at boot, then never again because of the `AeEnable = false`. At ground the algorithm might pick a slow shutter and boost the gain, but at high altitudes solar irradiance is 15% stronger and cloud decks have an higher albedo.

**Recommendation**: Either re-run exposure calibration at altitude, or enable Auto Exposure

2. **Motion Blur**
If the balloon will swing and oscillate and shutter speed is slower than 1/500s, angular rotation might blur roads, rivers, and vegetation across multiple pixels

**Recommendation**: Hard ceiling on exposure time: shutter must be <= 1/1000s to freeze motion blur

Other things to consider: adding an exit condition in `script.py` alongside CTRL+C, locking white balance to daylight (`"AwbMode": "daylight"`) to stop different tints from payload oscillation, adding a try/except to `picam2.start()` to prevent crashes in case of a voltage drop or cable noise.

### 3. Candidate Selection & Processing
Things to consider:
- popping early
- unhandled corrupted image crash

#### Overview:
1. Candidate Selection (choosing_foto.py):
Filters log_pozitii.csv for photos taken above 25 km (MIN_ALT_M = 25000.0).
Picks up to 20 candidate photos that maximize geographical spread (using Convex Hull and Haversine distance).
Evaluates each candidate through a Clarity Filter (checks extreme exposure, cloud deck, cloud quadrants, blur, and atmospheric haze).
Selects the top 5 clearest photos and copies them into ~/Desktop/data_processing_mode/photos_for_akida/.

2. Preprocessing (pre-processing.py):
Pads each image with black pixels to reach exact multiples of 64 (3280 × 2464 ➔ 3328 × 2496).
Slices each image into a grid of 52 × 39 = 2,028 patches of 64 × 64 pixels.
Stacks all 2,028 patches in RAM into a single 4D NumPy array (2028, 64, 64, 3) and saves it as batch.npy.


#### Risks
1. **Hard Altitude Ceiling Risk**
We need to consider what if the balloon pops earlier than expected (like 24800 meters), or the GPS altitude calibration has a small offset. In that case, `choosing_foto.py` will find 0 photos, log a message and exit without selecting any photos - mission loss of AI results.

```python
if not high_alt:
        print("Nicio poza nu indeplineste pragul de altitudine.")
        return
```

**Recommendation**: A fallback is needed, something like: *if no photos are above 25k km, select top 20 photos with highest recorded altitude*

2. **Unhandled Corrupted Image Crash**
In `pre-processing.py`:

```python
img = Image.open(image_path).convert("RGB")
```

If a photo write was cut short by a power fluctuation or SD write stall, PIL throws an unhandled PIL.UnidentifiedImageError or OSError: image file is truncated. There is zero try/except inside `process_image()`. If one single photo out of the 5 is corrupt, `pre-processing.py` crashes on that photo and aborts the entire preprocessing run, generating zero segments for the remaining 4 perfectly good photos.

**Recommendation**: again we should wrap image opening in try/except (OSError, Image.DecompressionBombError): log the corrupted photo and continue slicing the remaining ones.


### 4. Edge AI Inference
Things to consider:
- pcie stability
- fragile $home paths
- unhandled exception in np.load(batch.npy)
- obsolete else


#### Overview:
1. Host Driver & Container Launch (start_akida.sh):
Verifies the kernel module is loaded (/dev/akida0); injects akida-pcie.ko if missing.
Spins up the isolated Docker container (akida-image) with direct PCIe hardware access (--device=/dev/akida0).
Executes inference.py inside the container.
2. Hardware Inference Pipeline (inference.py):
Loads the trained 8-class SNN model: eurosat_8classes_v1.fbz.
Calls map_to_hardware() to explicitly bind the model to /dev/akida0 .
Loads batch.npy (all 2,028 patches per photo) and runs bulk batch inference via model.predict_classes(flight_batch).
Writes dual outputs:
.csv for complete on-board SD card archiving.
.bin (2,028 uint8 bytes) for LoRa radio downlink.
Logs overall performance (FPS, latency, frame count) to log_performanta.csv.

#### Risks:
1. **PCIe Link Stability**
During balloon burst and high-speed descent, the payload experiences tumbling and vibration. Because of this, thermal contraction on PCIe Gen 2/3 lanes can cause PCIe link resets or DMA timeouts in the `akida_pcie` driver.

**Recommendation**: The ribbon cable should be mechanically secured with tape/foam padding so physical shock during balloon burst doesn't disrupt the bus pins.

2. **Fragile $home paths**
In `start_akida.sh`, if this script is launched automatically at boot via systemd or /etc/rc.local, $HOME resolves to /root, not /home/synapse2025. The script immediately crashes with "Nu gasesc folderul /root/akida-setup/akida_dw_edma".

**Recommendation**: Hardcode absolute paths (e.g. /home/synapse2025/Desktop/...) instead of relying on $HOME, and configure the user in /etc/sudoers with NOPASSWD: ALL so docker and insmod never prompt for a password.

3. **Unhandled Exception on np.load(batch.npy)**
In `inference.py`:
There is zero try/except block around np.load(batch_path) or model.predict_classes(). If the balloon bursts or power fluctuates while pre-processing.py is writing batch.npy to the SD card, the file will be partially written or corrupted. When inference.py hits that corrupted file, np.load throws an uncaught ValueError / EOFError, crashing the entire Python script immediately.

**Recommendation**: We should wrap the batch inference loop in a try/except Exception as e: so a single bad file is logged and skipped, allowing the remaining 4 images to be classified.

4. **The Obsolete Else**
In `inference.py`:
In inference.py, if batch.npy is missing, the code falls back to an else branch that looks for individual Frame_N.png files. However, in pre-processing.py, the code that saves .png files was commented out to save CPU cycles.

We should just remove this else branch.


### 5. Downlink Telemetry
Things to consider:
- lora packet size
- missing packet shift


#### Overview:
1. Onboard Binary Serialization (inference.py):
Directly converts Akida prediction indices into an 8-bit unsigned integer array:
```python
predictions.astype(np.uint8).tofile(bin_path)
```
Exactly 2,028 bytes per image (52 × 39 patches), replacing the 46.6 KB plain text CSV.

2. Ground Station Telemetry Decoding (decode_downlink.py):
Reads the downlinked .bin file, matches each byte index to its corresponding class name, computes class percentage distributions, and outputs a ground-verified CSV table.


#### Risks:
1. **Lora Packets**
A standard LoRa transceiver (Semtech SX1276 / SX1262) cannot transmit 2,028 bytes in one single message.
The maximum physical hardware packet limit (MTU) for LoRa is 255 bytes.
In practice, payload size per packet is typically kept to ~128–200 bytes for radio stability. If the OBC tries to send foto_1.bin (2,028 bytes) as one continuous chunk, the LoRa radio driver will reject it or truncate it to 255 bytes, losing ~88% of the data.

**Recommendation**: The binary file must be packetized into smaller radio packets (2028 bytes / 200 bytes/packet = 11 packets per photo)

2. **Missing Packet Shift**
In `decode_downlink.py`:

```python
raw_bytes = list(bin_path.read_bytes())
decoded_labels = [CLASSES[b] ... for b in raw_bytes]
```

The decoder assumes byte index i strictly corresponds to patch index i on the 52 × 39 spatial grid but if packet #3 (patches 400 to 600) is dropped in the radio link due to interference, and the ground station simply concatenates the surviving packets: byte 601 now sits at index 401
Every subsequent patch shifts backwards. The entire 2D reconstructed map of Earth gets corrupted: forests end up in rivers, and highways end up in crops.

**Recommendation**: Each LoRa packet must include an offset header [PHOTO_ID, PACKET_NUM, OFFSET]. The ground receiver must initialize a fixed 2,028-byte array (pre-filled with 0xFF / Unknown) and place incoming bytes into their exact grid slots.



