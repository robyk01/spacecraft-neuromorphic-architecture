# Flight Software Pipeline

### Architectural Flow

```
[Camera (script.py)]
        │  Captures high-res photos & logs GPS (alt > 25 km)
        ▼
[Photo Selection (choosing_foto.py)]
        │  Filters by altitude, max geographic spread, and runs Clarity Filter
        ▼
[Preprocessing (pre-processing.py)]
        │  Pads & slices into 64x64 patches; stacks into (N, 64, 64, 3) tensor
        │  Saves atomic binary: /data/segmente/<photo>/batch.npy (0.05s, 0% CPU burn)
        ▼
[Hardware Inference (start_akida.sh -> Docker -> inference.py)]
        │  Loads batch.npy; dispatches full batch in ONE call to /dev/akida0
        │  Writes telemetry CSV: poza_XXXX.csv [frame, eticheta]
        ▼
[Downlink & Summary (inference_performances.py)]
        │  Logs summary to log_performanta.csv
        ▼
[Radio Downlink to Ground Station (CSV only)] & [SD Card Storage (Raw Images)]
```

#### 1. Photo Acquisition & Candidate Selection
- The camera runs on the host Raspberry Pi OS (`script.py`), continuously capturing raw JPEG images and saving GPS coordinates, altitude, and timestamps into `log_pozitii.csv`.
- `choosing_foto.py` filters photos taken above **25 km altitude**, selects candidate photos maximizing geographical coverage, and evaluates them with the **Clarity Filter** (checks for atmospheric haze, cloud deck, quadrant cloud coverage, and blur).
- The top 5 clearest images are staged into `photos_for_akida/`.

#### 2. Preprocessing & Batch Serialization (`pre-processing.py`)
- Reads the selected images, calculates edge padding to multiples of 64, and slices the images into $64 \times 64$ patches.
- **Optimization:** Rather than saving thousands of individual `.png` files (which causes extreme CPU thermal runaway due to `zlib` compression), all patches are stacked in memory into a 4D NumPy array `(N, 64, 64, 3)` and saved as a single raw binary file (`batch.npy`).
- **Execution Time:** Reduced from ~90–120 seconds of 100% CPU burn to **< 0.1 seconds**.

#### 3. Hardware Inference in Docker (`start_akida.sh` & `inference.py`)
- The BrainChip Akida PCIe driver (`akida-pcie.ko`) is verified on the host (`/dev/akida0`).
- Docker container (`akida-image`) starts with device access (`--device=/dev/akida0`), mounting `data_processing_mode/` as `/data`.
- `inference.py` loads `batch.npy` in milliseconds and passes the **entire flight batch directly to Akida in one single call** (`model.predict_classes(flight_batch)`).
- **Throughput:** ~200+ FPS in batch mode vs ~20–40 FPS in patch-by-patch loops.

#### 4. Output & Telemetry Downlink
- `inference.py` writes classification results into `rezultate_inferenta/<image_name>.csv` with columns:
  - `frame`: `Frame_0`, `Frame_1`, ...
  - `eticheta`: Predicted EuroSAT class (`Forest`, `AnnualCrop`, `Highway`, etc.)
- Overall execution metrics (FPS, total frames, total time) are appended to `log_performanta.csv`.
- Downlink telemetry only transmits the lightweight CSV classifications to the Ground Station, while raw full-resolution photos remain safely on the SD card for post-mission physical retrieval.

---

### Thermal Testing & Mitigation Summary

During ground thermal testing in the sealed payload enclosure:
- **Baseline Idle Temperature:** ~65°C–67°C.
- **Problem Observed:** In legacy testing, the `pre-processing` stage caused temperatures to spike to **82°C+**, triggering Broadcom BCM2712 SoC thermal throttling.
- **Investigation Finding:** The BrainChip Akida chip is ultra-low power (milliwatts) and actually cools down the system during inference. The thermal runaway was entirely caused by the software saving 3,000–5,000 individual PNGs per image via CPU `zlib` compression.
- **Solution:** Replaced PNG disk serialization with in-memory `batch.npy` transfer and bulk batch inference.
- **Result:** TBD

---
