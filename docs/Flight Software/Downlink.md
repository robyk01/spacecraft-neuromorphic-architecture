### Current State
Flight Camera: Raspberry Pi Camera Module v2 (Sony IMX219 sensor)
Sensor Resolution: 3280 x 2464 pixels
Patch Grid: 64 x 64 patches
=> Horizontal Patches: 3280 / 64 = 52 patches
Vertical Patches: 2464 / 64 = 39 patches
Patches per Image = 52 * 39 = 2028 patches

| Format | Desc | Size per Image | Transmission Time @ 9.6 kbps | Reduction |
| - | - | - | - | - |
| Raw JPEG | Quality 95-100% | ~3.5 MB | ~48.6 min | - |
| Current CSV | Plain ASCII | ~46.6 KB | 38.88 seconds | Baseline |
| Bytes Pack | 1 byte per class ID | ~2.03 KB | 1.69 seconds | 95.65% |
| 4-bit Pack | using 4 digits per prediction (10 classes) | ~1.01 kb | 0.84 seconds | 97.83% |
| 3-bit Pack | using 3 digits per prediction (8 classes) | ~0.76 kb | 0.63 seconds | 98.37% |

**Formula**:
Average LoRA downlink speed: 9.6 kbps
1 CSV entry - 46.6 KB for one image, 46.6 KB * 8 = 372.8 kilobits, 372 kb / 9.6 kbps = 38.8 seconds

For top 5 clearest images: 
Current CSV - 46.6 KB per image * 5 = 233 KB, 38.8 seconds * 5 = 194 s = 3.2 minutes
4-bit Pack - 5070 bytes, 4.22 seconds, 23 LoRA packets
3-bit Pack - 3805 bytes, 3.17 seconds, 18 LoRA packets

Conclusion:
in inference.py, current function creates a CSV for each image and writes a row in the CSV for each path with naming "Frame_{n}.png,{class}\n" which takes 20 bytes, while the computer only cares that Patch n is class. So we can write that in binary, each position in the binary file is a frame number and the number represents the class. Example:

Byte at index 0 = 0x01 (frame 0: forest)
Byte at index 1 = 0x01 (frame 1: forest)
Byte at index 2 = 0x07 (frame 2: river)
etc
