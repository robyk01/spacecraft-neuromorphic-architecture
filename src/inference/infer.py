# infer.py - Inferenta Akida pe O SINGURA imagine
#
# Ce face:
# 1. Incarca modelul antrenat (.fbz) pe chip-ul Akida
# 2. Incarca o singura imagine si o redimensioneaza la 64x64
# 3. Ruleaza inferenta pe chip si afiseaza output-ul brut + clasa prezisa
#
# Trebuie rulat DIN INTERIORUL containerului Docker (acces la /dev/akida0)
# Comanda: python3 /data/infer.py

import akida
import numpy as np
from PIL import Image
import math
import time
from typing import List

# fragments image in 64x64 parts, returns a list of batches
def slice_image(image: np.ndarray) -> List[np.ndarray]:
    patches = []
    width = image.shape[1]
    height = image.shape[0]
    
    for y in range(0, height, 64):
        for x in range(0, width, 64):
            patch = image[y : y + 64, x : x + 64]
            patches.append(patch)
    return patches

# calculates the ceiling bounds, pads any mismatched outer edges with zeros, cuts the entire high-res grid into independent 64x64 patches, and stacks them into a single 4-dimensional matrix block
def prepare_flight_batch(image: np.ndarray) -> np.ndarray:
    width = image.shape[1]
    height = image.shape[0]
    channels = image.shape[2]

    new_h = math.ceil(height / 64) * 64
    new_w = math.ceil(width / 64) * 64

    res = np.zeros((new_h, new_w, channels), dtype=np.uint8)
    res[0 : height, 0 : width] = image

    patches = slice_image(res)

    flight_batch = np.stack(patches, axis=0)
    return flight_batch

# counts up the results and dynamically map those numbers to a custom string array, it outputs a composition overview
def compute_metrics(predictions, start_time, end_time):
    # total duration
    time = end_time - start_time

    # total patches processed
    total_patches = len(predictions)

    # frames per second
    fps = total_patches / time

    # avg time per image (milliseconds)
    time_per_image = (time / total_patches) * 1000

    class_ids, counts = np.unique(predictions, return_counts=True)

    classes = [
        "AnunalCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SaltLake"
    ]

    print(f"Total Patches: {total_patches} | FPS: {fps:.2f} | Avg Latency: {time_per_image:.2f}ms")
    print("--------------------------------------------------")

    for class_id, count in zip(class_ids, counts):
        class_name = classes[class_id]
        percentage = (count / total_patches) * 100
        print(f"[{class_name}] -> Count: {count} ({percentage:.1f}%)")


# Incarca modelul
model = akida.Model("/data/eurosat_akida_model.fbz")
model.summary()

# Incarca imaginea
img = Image.open("/data/image.png")
img_array = np.array(img, dtype=np.uint8)
flight_batch = prepare_flight_batch(img_array)

# Inferenta
start_time = time.perf_counter()
predictions = model.predict_classes(flight_batch)
end_time = time.perf_counter()

compute_metrics(predictions, start_time, end_time)
