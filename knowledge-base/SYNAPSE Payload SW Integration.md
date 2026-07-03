

**SYNAPSE**

*Stratospheric Balloon Mission*

**Launch: Mid-September**

**TASK DOCUMENT**

**Payload SW Integration**

**Task Deadline:** 5 July

**Responsible:** Mirel, Violeta

**Mission:** SYNAPSE

**Document Version:** 1.0

**Status:** DRAFT

CLASSIFICATION: INTERNAL

# **Table of Contents**

# **1\. Scope**

This document covers the software integration of the Raspberry Pi 5 and the Akida BrainChip neuromorphic chip for the SYNAPSE payload. It includes the physical connection setup, environment configuration, inference pipeline development, camera integration, and performance benchmarking.

## **1.1 Objectives**

* Set up the Raspberry Pi — Akida hardware and software stack.

* Run inference on the Akida chip using EuroSAT satellite images.

* Develop and validate a full pipeline: image capture → segmentation → inference.

* Benchmark performance under realistic conditions.

* Iterate the model based on real conditions.

## **1.2 Out of Scope**

* Payload hardware assembly (covered by hardware task).

# **2\. Requirements**

## **2.1 Functional Requirements**

* The system shall run inference on the Akida chip.

* The inference pipeline shall classify 64x64 image patches from segmented satellite images.

* The system shall segment large images into 64x64 patches before inference.

* The full pipeline (capture → segment → infer) shall be implemented.

## **2.2 Non-Functional Requirements**

* Inference time per image and per full batch shall be measured and documented.

* The pipeline shall be tested with real satellite images photographed from a screen (realistic conditions) and various satellite images.


# **3\. Implementation**

## **3.1 Implementation Steps**

1. Test inference on Akida: classify 100 EuroSAT images saved locally on the Raspberry Pi. Measure classification accuracy and time. Then test on all images and record total inference time.

2. Download a large satellite image from the internet. Run the segmentation script on it. Classify the resulting patches. Measure time and performance.

3. Integrate the camera: configure it to capture images and save them to the Raspberry Pi automatically.

4. Develop the segmentation script: split large images into 64x64 patches.

5. Photograph satellite images from a screen and run the full pipeline (segmentation → inference). Record performance parameters.

6. Iterate the model taking into account the most realistic operating conditions.

## **3.2 Physical Setup — Raspberry Pi & Akida Connection**

Connect the Raspberry Pi to a laptop via UART using the red USB communication module. Wire connections as shown in the images below.

![Physical wiring — Raspberry Pi to UART module (1/3)][image1]

*Physical wiring — Raspberry Pi to UART module (1/3)*

![Physical wiring — Raspberry Pi to UART module (2/3)][image2]

*Physical wiring — Raspberry Pi to UART module (2/3)*

![Physical wiring — Raspberry Pi to UART module (3/3)][image3]

*Physical wiring — Raspberry Pi to UART module (3/3)*

![Raspberry Pi and Akida board assembled][image4]

*Raspberry Pi and Akida board assembled*

## **3.3 Driver Installation & Serial Connection (UART)**

### **Step 1 — Install UART driver**

Download and install the VCP driver for the red USB communication module:

FTDI VCP Drivers: [https://ftdichip.com/drivers/vcp-drivers/](https://ftdichip.com/drivers/vcp-drivers/)

Run the setup executable, install, then restart the laptop.

![Driver installation — setup executable][image5]

*Driver installation — setup executable*

### **Step 2 — Identify COM port**

Connect the red module via USB. Open Device Manager and note the COM port number under Ports:

![Device Manager — COM port identification][image6]

*Device Manager — COM port identification*

### **Step 3 — Connect via PuTTY**

Download PuTTY: [https://putty.org/index.html](https://putty.org/index.html)

Open PuTTY and configure as follows: Connection type: Serial | Speed (Baud rate): 115200 | Serial line: COM5 (or your identified port).

![PuTTY configuration][image7]

*PuTTY configuration*

Connect the wires from the Raspberry Pi to the red module while the Pi is powered on. You should now have a serial terminal.

Login credentials:

Hostname: synapse2025

Password: synapse2025

### **Step 4 — WiFi connection**

Connect to the WiFi hotspot using the following command:

sudo nmcli dev wifi connect "My" password "violetamihai"

WiFi hotspot credentials: SSID: My | Password: violetamihai (rename phone hotspot accordingly if needed).

## **3.4 Akida Driver & Docker Environment**

### **Step 1 — Load Akida kernel module**

Run the following each time the Raspberry Pi boots:

cd \~/akida-setup/akida\_dw\_edma

sudo insmod ./akida-pcie.ko

### **Step 2 — Verify Akida device**

ls \-l /dev/akida\*

Expected output:

crw------- 1 root root 10, 260 Jun  9 00:57 /dev/akida0

### **Step 3 — Dockerfile**

Create the Dockerfile on the Raspberry Pi:

nano \~/akida-setup/test-simple-program/Dockerfile

Content:

FROM ubuntu:22.04

RUN apt update && apt install \-y python3 python3-pip

RUN pip3 install akida==2.16.1 numpy==1.26.4 Pillow

WORKDIR /data

Save with Ctrl+X → Y → Enter, then build the image:

sudo docker build \-t akida-image \~/akida-setup/test-simple-program/

### **Step 4 — Start the container**

sudo docker run \-it \--device=/dev/akida0 \-v /home/synapse2025/Desktop/akida\_sample:/data akida-image

### **Step 5 — Transfer images to Pi**

From Windows laptop, copy images to the Raspberry Pi:

scp \-r "C:\\Users\\Violeta\\Desktop\\Synapse\\subsystems\\payload\\images" synapse2025@192.168.137.189:/home/synapse2025/Desktop/akida\_sample/

## **3.5 Startup Checklist (Every Session)**

Follow these steps each time the Raspberry Pi is powered on:

7. Load the Akida kernel module:

cd \~/akida-setup/akida\_dw\_edma

sudo insmod ./akida-pcie.ko

8. Verify the device exists:

ls /dev/akida\*

9. Start the Docker container:

sudo docker run \-it \--device=/dev/akida0 \-v /home/synapse2025/Desktop/akida\_sample:/data akida-image

### **Inside the container**

Run inference on a single image:

python3 /data/infer.py

Run inference on all images:

python3 /data/infer\_all.py

Exit the container:

exit

### **Key files**

* /home/synapse2025/Desktop/akida\_sample/infer.py — inference on a single image

* /home/synapse2025/Desktop/akida\_sample/infer\_all.py — inference on all images across 10 classes

* \~/akida-setup/test-simple-program/Dockerfile — Docker container recipe

Note: files in /data/ inside the container map directly to /home/synapse2025/Desktop/akida\_sample/ on the Pi. Scripts edited on the Pi are immediately visible in the container. Docker only needs to be rebuilt if the Dockerfile changes.

![Docker container summary — file structure and workflow][image8]

*Performances* 

## **3.6 Performance Results**

*\[Fill in inference results as tests are completed.\]*

| Test | Images | Time (total) | Notes |
| ----- | ----- | ----- | ----- |
| EuroSAT — 100 images | 100 | \[TBD\] |  |
| EuroSAT — full dataset | \[TBD\] | \[TBD\] |  |
| Large image — segmented | \[TBD\] | \[TBD\] |  |
| Full pipeline (screen photo) | \[TBD\] | \[TBD\] | Real conditions |

# **4\. Risks and Mitigation**

| ID | Risk | Probability | Impact | Mitigation |
| ----- | ----- | ----- | ----- | ----- |
| **RSK-01** | Akida kernel module fails to load after reboot | Medium | High | Load kernel module manually each session (sudo insmod ./akida-pcie.ko). Document startup checklist clearly. |
| **RSK-02** | Docker container loses access to /dev/akida0 | Medium | High | Always pass \--device=/dev/akida0 flag when starting container. Verify device exists before running container. |
| **RSK-03** | Inference performance insufficient for real-time pipeline | Medium | Critical | Benchmark early (Section 3.1). Iterate model and segmentation parameters if needed (Section 3.5). |
| **RSK-04** | Camera integration fails or produces incompatible image format | Low | Medium | Test camera save pipeline independently before integrating with segmentation \+ inference scripts. |
| **RSK-05** | WiFi credentials change or hotspot unavailable during setup | Low | Medium | Document current credentials. Fallback: use UART/USB connection as alternative access method. |

## **4.1 Additional Notes**

*\[Any additional risk observations or mitigation actions not captured in the table above.\]*