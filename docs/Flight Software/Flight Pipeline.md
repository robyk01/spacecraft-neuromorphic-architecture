### Architectural Flow:
#### 1. Initialization
The model initialization happens inside a Docker container, which is granted hardware access via the `--device=/dev/akida0` flag. The Akida kernel module itself is handled by host system commands (`sudo insmod ./akida-pcie.ko`) before Docker starts.
The camera code is decoupled from the inference script. The camera runs on the host Raspberry Pi OS, snapping photos and automatically dropping them into `/home/synapse2025/Desktop/akida_sample` folder.
#### 2. Preprocess
Pass the photo to the `prepare_flight_batch()` function. The script will use Pillow to load the image file from `/data/`, convert it to a NumPy array, and pass it directly into `prepare_flight_batch(image)` to yield the 4D tensor stack.
#### 3. Hardware InferenceFlight
Pass the 4D batch resulted from preprocessing into `model_akida.predict_classes()`. The inference must be wrapped in high-precision time deltas.
#### 4. Output
Here the scripts `infer.py` and `infer_all.py` are execution runs that calculate and print out final performance reports directly to the terminal console. The script needs to calculate and format these exact metrics:
- Total images processed
- Correctly classified patches vs. Incorrectly classified patches
- Global accuracy percentage
- Total processing time (seconds)
- Average time per image (milliseconds)
- Total frames per second

### Testing
#### Results on infer.py for baseline_model.fbz

                Model Summary                 
______________________________________________
Input shape: [64, 64, 3]
Output shape: [1, 1, 10]  
Sequences: 1
Layers: 14
____________________________________________________________
Layer (type)               Output shape   Kernel shape    

============ SW/stem_conv-dequantizer (Software) ===========

stem_conv (InputConv2D)    [32, 32, 32]   (3, 3, 3, 32)   
____________________________________________________________
b1_dw (DepthwiseConv2D)    [32, 32, 32]   (3, 3, 32, 1)   
____________________________________________________________
b1_pw (Conv2D)             [32, 32, 64]   (1, 1, 32, 64)  
____________________________________________________________
b2_dw (DepthwiseConv2D)    [16, 16, 64]   (3, 3, 64, 1)   
____________________________________________________________
b2_pw (Conv2D)             [16, 16, 64]   (1, 1, 64, 64)  
____________________________________________________________
b3_dw (DepthwiseConv2D)    [16, 16, 64]   (3, 3, 64, 1)   
____________________________________________________________
b3_pw (Conv2D)             [16, 16, 128]  (1, 1, 64, 128) 
____________________________________________________________
b4_dw (DepthwiseConv2D)    [8, 8, 128]    (3, 3, 128, 1)  
____________________________________________________________
b4_pw (Conv2D)             [8, 8, 128]    (1, 1, 128, 128)
____________________________________________________________
b5_dw (DepthwiseConv2D)    [8, 8, 128]    (3, 3, 128, 1)  
____________________________________________________________
b5_pw (Conv2D)             [1, 1, 256]    (1, 1, 128, 256)
____________________________________________________________
fc1 (Dense1D)              [1, 1, 128]    (256, 128)      
____________________________________________________________
logits (Dense1D)           [1, 1, 10]     (128, 10)       
____________________________________________________________
dequantizer (Dequantizer)  [1, 1, 10]     N/A             
____________________________________________________________
Total Patches: 510 | FPS: 174.55 | Avg Latency: 5.73ms
--------------------------------------------------
[AnnualCrop] -> Count: 4 (0.8%)
[HerbaceousVegetation] -> Count: 78 (15.3%)
[Highway] -> Count: 35 (6.9%)
[Industrial] -> Count: 154 (30.2%)
[PermanentCrop] -> Count: 4 (0.8%)
[Residential] -> Count: 229 (44.9%)
[River] -> Count: 5 (1.0%)
[SeaLake] -> Count: 1 (0.2%)

#### Results on infer.py for attuned_model.py:

                Model Summary                 
______________________________________________
Input shape: [64, 64, 3]
Output shape:  [1, 1, 10]
Sequences: 1
Layers: 8
_________________________________________________________
Layer (type)            Output shape   Kernel shape    

============= SW/stem_conv-logits (Software) ============

stem_conv (InputConv.)  [32, 32, 32]   (3, 3, 3, 32)   
_________________________________________________________
b1_sep (Sep.Conv.)      [32, 32, 64]   (3, 3, 32, 1)   
_________________________________________________________
						   (1, 1, 32, 64)  
_________________________________________________________
b2_sep (Sep.Conv.)      [16, 16, 64]   (3, 3, 64, 1)   
_________________________________________________________
						   (1, 1, 64, 64)  
_________________________________________________________
b3_sep (Sep.Conv.)      [16, 16, 128]  (3, 3, 64, 1)   
_________________________________________________________
						   (1, 1, 64, 128) 
_________________________________________________________
b4_sep (Sep.Conv.)      [8, 8, 128]    (3, 3, 128, 1)  
_________________________________________________________
						   (1, 1, 128, 128)
_________________________________________________________
b5_sep (Sep.Conv.)      [1, 1, 256]    (3, 3, 128, 1)  
_________________________________________________________
						   (1, 1, 128, 256)
_________________________________________________________
fc1 (Fully.)            [1, 1, 128]    (1, 1, 256, 128)
_________________________________________________________
logits (Fully.)         [1, 1, 10]     (1, 1, 128, 10) 
_________________________________________________________
Total Patches: 510 | FPS: 286.47 | Avg Latency: 3.49ms
--------------------------------------------------
[AnnualCrop] -> Count: 8 (1.6%)
[HerbaceousVegetation] -> Count: 19 (3.7%)
[Highway] -> Count: 44 (8.6%)
[Industrial] -> Count: 104 (20.4%)
[PermanentCrop] -> Count: 33 (6.5%)
[Residential] -> Count: 289 (56.7%)
[River] -> Count: 12 (2.4%)
[SeaLake] -> Count: 1 (0.2%)


#### Results on infer_all.py for baseline_model.fbz:
Se ruleaza inferenta...

[AnnualCrop] 2901/3000 =  96.70%  | timp: 132.06s
[Forest] 2973/3000 =  99.10%  | timp: 125.68s
[HerbaceousVegetation] 2886/3000 =  96.20%  | timp: 122.45s
[Highway] 2438/2500 =  97.52%  | timp: 100.73s
[Industrial] 2464/2500 =  98.56%  | timp: 102.64s
[Pasture] 1881/2000 =  94.05%  | timp: 80.25s
[PermanentCrop] 2264/2500 =  90.56%  | timp: 99.83s
[Residential] 2985/3000 =  99.50%  | timp: 121.54s
[River] 2372/2500 =  94.88%  | timp: 99.73s
[SeaLake] 2985/3000 =  99.50%  | timp: 137.93s

REZULTATE FINALE
Total imagini procesate : 27000
Clasificate corect: 26149
Clasificate gresit: 851
Acuratete globala: 96.85%
Timp total: 1123.37s
Timp mediu per imagine: 41.61ms
Imagini per secunda: 24.0 FPS


#### Results on infer_all.py for attuned_model.fbz:
Se ruleaza inferenta...

[AnnualCrop] 2822/3000 =  94.07%  | timp: 60.40s
[Forest] 2606/3000 =  86.87%  | timp: 62.67s
[HerbaceousVegetation] 2759/3000 =  91.97%  | timp: 59.20s
[Highway] 2452/2500 =  98.08%  | timp: 50.80s
[Industrial] 2424/2500 =  96.96%  | timp: 58.09s
[Pasture] 1884/2000 =  94.20%  | timp: 43.08s
[PermanentCrop] 2381/2500 =  95.24%  | timp: 51.13s
[Residential] 2989/3000 =  99.63%  | timp: 64.97s
[River] 2410/2500 =  96.40%  | timp: 60.90s
[SeaLake] 2968/3000 =  98.93%  | timp: 62.59s


REZULTATE FINALE
Total imagini procesate : 27000
Clasificate corect: 25695
Clasificate gresit: 1305
Acuratete globala: 95.17%
Timp total: 574.10s
Timp mediu per imagine: 21.26ms
Imagini per secunda: 47.0 FPS


#### Results on infer_all.py for attuned_model.fbz after data augmentation:
Se ruleaza inferenta...

[AnnualCrop]   66/100  =  66.00%  | timp: 2.50s
[Forest]    4/100  =   4.00%  | timp: 2.16s
[HerbaceousVegetation]   50/100  =  50.00%  | timp: 2.00s
[Highway]   47/100  =  47.00%  | timp: 1.99s
[Industrial]   70/100  =  70.00%  | timp: 2.00s
[Pasture]   21/100  =  21.00%  | timp: 1.98s
[PermanentCrop]   25/100  =  25.00%  | timp: 2.07s
[Residential]   41/100  =  41.00%  | timp: 1.99s
[River]   48/100  =  48.00%  | timp: 1.97s
[SeaLake]   89/100  =  89.00%  | timp: 2.02s


REZULTATE FINALE
Total imagini procesate : 1000
Clasificate corect      : 461
Clasificate gresit      : 539
Acuratete globala       : 46.10%
Timp total              : 20.77s
Timp mediu per imagine  : 20.77ms
Imagini per secunda     : 48.2 FPS

**Conclusion**:
After noise injection the model is practically blind to forests now. In the clean dataset, forests are easily indentified by deep, dark green colors and textures (leaves). Here, Rayleight Scaterring washed away the deep green, replacing it with blue/gray, the blur decreased texture quality and once these are gone, the model has no geometric shapes to latch onto.
It seems like Pasture and PermanentCrop both suffered like forest. These classes rely heavily on distinct color shades and ground patterns. When introduced to sun glares and clouds, pastures look exactly like HerbaceousVegetation or AnnualCrop.
This proves that the original model didn't actually learn what a forest is. It learned what a perfect low-altitude photo of a forest looks like.