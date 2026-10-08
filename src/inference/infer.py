# infer.py - Inferenta Akida pe O SINGURA imagine
#
# Ce face:
# 1. Incarca modelul antrenat (.fbz) pe chip-ul Akida
# 2. Sparge imaginea mare in patch-uri de 64x64
# 3. Aplica filter (Extreme Values, Cloud Deck, Blur, Haze) pe fiecare patch
# 4. Ruleaza inferenta doar pe patch-urile curate pe chip-ul Akida
# 5. Afiseaza metricile si numarul de patch-uri ignorate (neclare)

# Trebuie rulat DIN INTERIORUL containerului Docker (acces la /dev/akida0)
# Comanda: python3 /data/infer.py

import akida
import numpy as np
from PIL import Image
import math
import time
from typing import List
import os

OUTPUT_DIR = "src/data"
image_name = "satellite_image"

# fragments image in 64x64 parts, returns a list of batches
def slice_image(image: np.ndarray) -> List[np.ndarray]:
    patches = []
    unclear_cnt = 0
    width = image.shape[1]
    height = image.shape[0]
    
    for y in range(0, height, 64):
        for x in range(0, width, 64):
            patch = image[y : y + 64, x : x + 64]
            if is_image_clear(patch):
                patches.append(patch)
            else:
                unclear_cnt += 1
    return patches, unclear_cnt


# calculates the ceiling bounds, pads any mismatched outer edges with zeros, cuts the entire high-res grid into independent 64x64 patches, and stacks them into a single 4-dimensional matrix block
def prepare_flight_batch(image: np.ndarray) -> np.ndarray:
    width = image.shape[1]
    height = image.shape[0]
    channels = image.shape[2]

    new_h = math.ceil(height / 64) * 64
    new_w = math.ceil(width / 64) * 64

    res = np.zeros((new_h, new_w, channels), dtype=np.uint8)
    res[0 : height, 0 : width] = image

    patches, unclear_cnt = slice_image(res)

    if not patches:
        return np.empty((0, 64, 64, channels), dtype=np.uint8), unclear_cnt

    flight_batch = np.stack(patches, axis=0)
    return flight_batch, unclear_cnt


# evaluates 4 physical clarity criteria on an image patch: extreme values, cloud coverage, blur and haze
def is_image_clear(image) -> bool:
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image
    # grayscale representation
    gray_img = np.array(pil_img.convert("L"))

    # Extreme Values
    pct = np.mean((gray_img < 20) | (gray_img > 230))
    pass1 = pct <= 0.30

    # Cloud Deck
    gray_mean = np.mean(gray_img)
    gray_std = np.std(gray_img)
    pass2 = gray_mean <= 170 or gray_std >= 15

    # Quadrant Coverage
    h, w = gray_img.shape[:2]
    mid_y, mid_x = h // 2, w // 2
    quadrants = [
        gray_img[:mid_y, :mid_x], gray_img[:mid_y, mid_x:],
        gray_img[mid_y:, :mid_x], gray_img[mid_y:, mid_x:]
    ]
    pass5 = True
    for quad in quadrants:
        if quad.size > 0 and np.mean(quad) > 175.0 and np.std(quad) < 18.0:
            pass5 = False
            break

    # Laplacian Variance
    if h > 2 and w > 2:
        laplacian = (
            gray_img[:-2, 1:-1] + gray_img[2:, 1:-1]
            + gray_img[1:-1, :-2] + gray_img[1:-1, 2:]
            - 4 * gray_img[1:-1, 1:-1]
        )
        pass3 = True if gray_mean < 60 else (np.var(laplacian) > 30.0)
    else:
        pass3 = True

    # Haze (HSV)
    hsv_img = np.array(pil_img.convert("HSV"))
    sat_mean, val_mean = np.mean(hsv_img[:, :, 1]), np.mean(hsv_img[:, :, 2])
    pass4 = not (val_mean > 140 and sat_mean < 35)
    return pass1 and pass2 and pass3 and pass4 and pass5


# counts up the results and dynamically map those numbers to a custom string array, it outputs a composition overview
def compute_metrics(predictions, start_time, end_time, unclear_cnt):
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
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "PermanentCrop", "Residential", "River"
    ]

    print(f"Total Patches: {total_patches} | FPS: {fps:.2f} | Avg Latency: {time_per_image:.2f}ms")
    print("--------------------------------------------------")

    for class_id, count in zip(class_ids, counts):
        class_name = classes[class_id]
        percentage = (count / total_patches) * 100
        print(f"[{class_name}] -> Count: {count} ({percentage:.1f}%)")

    print(f"Unclear Image Count: {unclear_cnt}")


# Incarca modelul
model = akida.Model("src/data/models/eurosat_8classes_v1/model.fbz")
model.summary()

# Incarca imaginea
img = Image.open("src/data/satellite_image.jpg").convert("RGB")
img_array = np.array(img, dtype=np.uint8)
flight_batch, unclear_cnt = prepare_flight_batch(img_array)

# Inferenta
if flight_batch.shape[0] > 0:
    start_time = time.perf_counter()
    predictions = model.predict_classes(flight_batch)
    end_time = time.perf_counter()

    csv_path = os.path.join(OUTPUT_DIR, f"{image_name}.csv")
    bin_path = os.path.join(OUTPUT_DIR, f"{image_name}.bin")

    # binary file with predictions for each patch as uint8 
    predictions.astype(np.uint8).tofile(bin_path)

    compute_metrics(predictions, start_time, end_time, unclear_cnt)
else:
    print(f"Niciun patch nu a trecut de filtru. Imagini neclare: {unclear_cnt}")