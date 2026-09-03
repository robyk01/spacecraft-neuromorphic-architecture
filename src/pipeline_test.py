import os, csv, time
import numpy as np
import akida
from PIL import Image

# Paths
BASE_DIR = r"d:\Projects\synapse\Synapse\subsystems\payload\src\Code on Rasp Pi"
PHOTO = os.path.join(BASE_DIR, r"foto_mode\poze\real_image.jpg")
MODEL = os.path.join(BASE_DIR, r"data_processing_mode\models\new_attuned.fbz")

# Slice
img = Image.open(PHOTO).convert("RGB")
arr = np.array(img)
patches = [arr[y:y+64, x:x+64] for y in range(0, (arr.shape[0]//64)*64, 64) for x in range(0, (arr.shape[1]//64)*64, 64)]
batch = np.stack(patches, axis=0)

# Save & Load batch.npy
np.save("test_batch.npy", batch)
loaded_batch = np.load("test_batch.npy")

# Infer with real model
model = akida.Model(MODEL)
predictions = model.predict_classes(loaded_batch)
print(f"Success! Processed {len(predictions)} patches with Akida in batch mode.")

if os.path.exists("test_batch.npy"):
    os.remove("test_batch.npy")