## Edge Cases & Risk Assessment

### 1. Boot & Environment
Things to consider:
- how the Pi boots at -45/-50 C
- kernel modules loading

#### Overview:
1. ESP32 (OBC) fires first (under 50 milliseconds). It activates the hardware/software watchdogs, initializes sensor loops, and handles telemetry. If the rasp pi dies, the ESP keeps downlinking telemetry to the ground station.
2. Rasp Pi takes around 20-30s to boot. It has to initialize the filesystem and hardware buses of the Linux OS.
By default, Linux has no idea what Akida is so we inject the driver `akida-pcie.ko` into the running Linux kernel, creating a hardware interface file in the system `/dev/akida0`. The hardware communicates with the SDK through reading and writing of bytes.

#### Risks
1. **Temperature Bug** - Done
In `Synapse\subsystems\obc\esp32-obc\main\tasks\mode_manager_task.c`:

In the stratosphere, external air temperature drops to -40 to -60°C. During ascent, the original ESP32 code checked `temp.temperature_external_c < -10.0f`, which would think the spacecraft is dying and permanently trigger Safe Mode, disabling `photo_active = false` and aborting camera/inference ops.

*Status*: **Done**. Fixed by checking internal board temperature (`temp.temperature_internal_c.devices[0].temperature_c`) with nominal fallback, avoiding premature Safe Mode.

2. **Missing Processing Command**
In `Synapse\subsystems\obc\rpi\comm\testint_comm.py` and `Synapse\subsystems\obc\esp32-obc\main\tasks\mode_manager_task.c`:

Right now neither scripts know about `run_entire_data_processing_mode.py`. When balloon is in active descent, `mode_manager_task.c` should launch a `CMD|START_PROCESSING` over UART to launch the 4-step AI inference pipeline. `testint_comm.py` receives the command on Rasp Pi and launches the python script.
Also OBC side is missing the radio downlink pipeline to receive and broadcast the classification results.
*Status*: Open (OBC subsystem responsibility).

3. **Thermal Risk** - Ignored
Unless there is a hardware relay that is missing from the software, the Raspberry Pi is hardwired to main power and remains powered on the entire flight, only the software remaining off/idle.

If the raspberry pi is kept powered off between 50m and 25km, the components in the payload might cold-soak. When the ESP32 switches power on to the Pi at 25km:
- MicroSD flash memory controllers might refuse to initialize or write below -20°C
- The onboard oscillator crystal can drift, corrupting PCIe and UART bus clocks

If only the flight software is kept off/idle, Linux consumes ~3 Watts which acts as a steady electrical heater, preventing the bay, batteries, and SD card from reaching -40°C. Battery capacity covers ~2.5 hrs of flight plus the Akida inference burst.

---

### 2. In-Flight Image Acquisition
Things to consider:
- auto exposure at high altitude
- motion blur when swinging
- check SD storage

#### Overview:
Hardware: Raspberry Pi Camera Module v2 (Sony IMX219 sensor, 3280 × 2464 resolution)
1. Runs `auto_select_shutter()` at startup to pick the sharpest shutter speed and analog gain.
2. Enters a `while True` continuous capture loop, saving a photo every 2 seconds to `~/Desktop/foto_mode/poze/`.
3. Appends photo metadata, camera settings, and GPS coordinates to `log_pozitii.csv`.

#### Risks
1. **Auto Exposure** - Done
In `Synapse\subsystems\payload\src\Code on Rasp Pi\foto_mode\script.py`:
`auto_select_shutter()` previously ran once at boot. At altitude, albedo and light change dramatically.
*Status*: **Done**. Script now recalibrates shutter and analog gain automatically every 5 minutes (`recalib_interval_sec=300`) using `time.monotonic()`.

2. **Motion Blur** - Done
If the balloon swings and oscillates with shutter slower than 1/500s, angular rotation blurs roads, rivers, and vegetation across multiple pixels.
*Status*: **Done**. Exposure times ceiling prioritized $\le 1/1000$s (1000 µs) to freeze oscillation motion blur.

---

### 3. Candidate Selection & Processing
Things to consider:
- popping early
- unhandled corrupted image crash

#### Overview:
1. Candidate Selection (`choosing_foto.py`):
Filters `log_pozitii.csv` for photos taken above 25 km (`MIN_ALT_M = 25000.0`).
Picks up to 20 candidate photos that maximize geographical spread (Convex Hull and Haversine distance).
Evaluates each candidate through a Clarity Filter (extreme exposure, cloud deck, cloud quadrants, blur, haze).
Selects the top 5 clearest photos and copies them into `~/Desktop/data_processing_mode/photos_for_akida/`.

2. Preprocessing (`pre-processing.py`):
Pads each image with black pixels to reach exact multiples of 64 (3280 × 2464 -> 3328 × 2496).
Slices each image into a grid of 52 × 39 = 2,028 patches of 64 × 64 pixels.
Stacks all 2,028 patches in RAM into a single 4D NumPy array `(2028, 64, 64, 3)` and saves it as `batch.npy`.

#### Risks
1. **Hard Altitude Ceiling Risk** - Done
If the balloon bursts early (e.g. 24,800 m), `choosing_foto.py` previously aborted with 0 photos.
*Status*: **Done**. Fallback added to pick the highest altitude photos that physically exist on disk, with `ZeroDivisionError` guards.

2. **Unhandled Corrupted Image Crash** - Done
In `pre-processing.py`, an unhandled `PIL.UnidentifiedImageError` or `OSError` on a single bad photo would crash the entire batch.
*Status*: **Done**. `Image.open` wrapped in `try/except` with directory creation only upon valid read, skipping bad images without aborting the rest.

---

### 4. Edge AI Inference
Things to consider:
- pcie stability
- fragile $home paths
- unhandled exception in np.load(batch.npy)
- obsolete else

#### Overview:
1. Host Driver & Container Launch (`start_akida.sh`):
Verifies the kernel module is loaded (`/dev/akida0`); injects `akida-pcie.ko` if missing.
Spins up the Docker container (`akida-image`) with direct PCIe hardware access (`--device=/dev/akida0`).
Executes `inference.py` inside the container.
2. Hardware Inference Pipeline (`inference.py`):
Loads the trained 8-class SNN model: `eurosat_8classes_v1.fbz`.
Calls `map_to_hardware()` to explicitly bind the model to `/dev/akida0`.
Loads `batch.npy` (all 2,028 patches per photo) and runs bulk batch inference via `model.predict_classes(flight_batch)`.
Writes dual outputs: `.csv` for SD card and `.bin` + `_packets.bin` for LoRa radio downlink.
Logs overall performance (FPS, latency, frame count) to `log_performanta.csv`.

#### Risks
1. **PCIe Link Stability**
Ribbon cable secured mechanically with tape/foam padding so physical shock during balloon burst doesn't disrupt bus pins.

2. **Fragile $home paths**
Ensure absolute paths are used in systemd scripts.

3. **Unhandled Exception on np.load(batch.npy)** - Done
If a single `batch.npy` is corrupt or interrupted, previously `inference.py` would crash or exit.
*Status*: **Done**. Inference loop wrapped in `try/except` with `continue`, skipping damaged files while processing remaining candidate photos.

4. **The Obsolete Else** - Done
In `inference.py`, the old fallback looking for individual `.png` files (which was causing CPU thermal throttling) was still present.
*Status*: **Done**. Cleanly deleted the obsolete else branch.

---

### 5. Downlink Telemetry
Things to consider:
- lora packet size
- missing packet shift

#### Overview:
1. **Patch Grid & Data Size**:
   - Camera resolution: 3280 × 2464 padded to 3328 × 2496.
   - Patch size: 64 × 64 pixels.
   - Spatial Grid: **52 columns × 39 rows = 2,028 patches per image**.
   - Each patch is classified into 1 of 8 EuroSAT classes (integer 0–7, 1 byte).
   - Full image raw predictions = **2,028 bytes**.

2. **Onboard Binary Serialization & LoRa Packetization** (`inference.py`):
   - Rather than sending 2,028 bytes at once (which exceeds the 255-byte hardware MTU of the SX1262 LoRa radio), the array is sliced into **11 radio packets**:
     - 10 packets of 200 bytes
     - 1 packet of 28 bytes
   - Each packet includes a 6-byte Big-Endian offset header:
     `[photo_id (1B), packet_idx (1B), total_packets (1B), offset (2B uint16), chunk_len (1B)]`
   - Maximum packet size = **206 bytes** ($\le 255$ bytes MTU).

3. **Ground Station Telemetry Decoding** (`decode_downlink.py`):
   - Initializes a fixed 2,028-element grid.
   - Places each packet's predictions into `grid[offset : offset + chunk_len]`.
   - Missing packets are flagged as `DROPPED` / `MISSING` without shifting subsequent coordinates.

#### Risks
1. **Lora Packets** - Done
*Status*: **Done**. Implemented in `inference.py` to auto-generate `[image_name]_packets.bin` (11 packets of $\le 206$ bytes).

2. **Missing Packet Shift** - Done
*Status*: **Done**. Implemented in `decode_downlink.py`. Dropped radio packets no longer scramble spatial coordinates.

---

### Downlink Verification & ASCII Grid Test Output

Below are two sample test executions of `decode_downlink.py` demonstrating ground reception:

#### Case A: Nominal Flight (100% Reception, 0 Dropped Packets)
```
============================================================
SYNAPSE TELEMETRY DECODER - foto_0001_packets.bin
============================================================
Format Mode           : LoRa Packet Stream (Photo ID: 0)
Packets Received      : 11 / 11 (100.0%)
Total Patches Target  : 2028
Patches Decoded       : 2028 (100.0% spatial coverage)
Patches Dropped       : 0 (0.0% lost)

CLASS DISTRIBUTION (Received Patches):
  [AnnualCrop            ] :  254 patches ( 12.5%)
  [Forest                ] :  254 patches ( 12.5%)
  [HerbaceousVegetation  ] :  254 patches ( 12.5%)
  [Highway               ] :  254 patches ( 12.5%)
  [Industrial            ] :  253 patches ( 12.5%)
  [PermanentCrop         ] :  253 patches ( 12.5%)
  [Residential           ] :  253 patches ( 12.5%)
  [River                 ] :  253 patches ( 12.5%)

RECEPTION GRID (52 columns x 39 rows) [. = Received, X = Dropped]:
------------------------------------------------------
....................................................
....................................................
....................................................
....................................................
[All 39 rows fully received with zero gaps]
------------------------------------------------------
Ground CSV saved to: foto_0001_decoded.csv
============================================================
```

#### Case B: RF Packet Drop (Packet #3 Lost During Transmission)
*Note: Packet #3 covers patches 600 to 799 (200 patches). Because of the offset header, patches 800+ remain in their exact spatial positions, preserving the geometry of roads, rivers, and forests.*

```
============================================================
SYNAPSE TELEMETRY DECODER - test_flight_packets.bin
============================================================
Format Mode           : LoRa Packet Stream (Photo ID: 0)
Packets Received      : 10 / 11 (90.9%)
Missing Packets       : [3] (dropped over radio)
Total Patches Target  : 2028
Patches Decoded       : 1828 (90.1% spatial coverage)
Patches Dropped       : 200 (9.9% lost)

CLASS DISTRIBUTION (Received Patches):
  [AnnualCrop            ] :  229 patches ( 12.5%)
  [Forest                ] :  229 patches ( 12.5%)
  [Highway               ] :  229 patches ( 12.5%)
  ...

RECEPTION GRID (52 columns x 39 rows) [. = Received, X = Dropped]:
------------------------------------------------------
....................................................
....................................................
....................................................
....................................................
....................................................
....................................................
....................................................
....................................................
....................................................
....................................................
....................................................
............................XXXXXXXXXXXXXXXXXXXXXXXX  <-- Patch 600 starts drop
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXX................................  <-- Patch 800 resumes accurately!
....................................................
....................................................
....................................................
------------------------------------------------------

Ground CSV saved to: test_flight_packets_decoded.csv (Missing patches logged as DROPPED)
============================================================
```
