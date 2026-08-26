# infer_all.py - Inferenta Akida pe TOATE imaginile din toate clasele
#
# Ce face:
# 1. Incarca modelul antrenat (.fbz) pe chip-ul Akida
# 2. Parcurge cele 10 foldere de clase (AnnualCrop, Forest, etc.)
#    din /data/images/ si proceseaza fiecare imagine
# 3. Compara predictia cu eticheta reala (numele folderului) si calculeaza:
#    - acuratete per clasa si globala
#    - timp total si timp mediu per imagine
#    - viteza in FPS (imagini procesate pe secunda)
# 4. Afiseaza si info despre hardware-ul Akida folosit
#
# Trebuie rulat DIN INTERIORUL containerului Docker (acces la /dev/akida0)
# Comanda: python3 /data/infer_all.py

import akida
import numpy as np
from PIL import Image
import os
import time

def is_image_clear(image):
    np_image = np.array(image.convert("L"))

    # Extreme Values
    pct = np.mean((np_image < 20) | (np_image > 230))
    pass1 = pct <= 0.30

    # Cloud Deck
    gray_mean = np.mean(np_image)
    gray_std = np.std(np_image)
    pass2 = gray_mean <= 170 or gray_std >= 15

    np_image = np.array(image.convert("L"))
    h, w = np_image.shape[:2]
    mid_y, mid_x = h // 2, w // 2

    ## Cloud Quadrant Check
    quadrants = [
        np_image[:mid_y, :mid_x],
        np_image[:mid_y, mid_x:],
        np_image[mid_y:, :mid_x],
        np_image[mid_y:, mid_x:]
    ]

    pass5 = True
    for quad in quadrants:
        q_mean = np.mean(quad)
        q_std = np.std(quad)

        if q_mean > 175.0 and q_std < 18.0:
            pass5 = False

    # Blur check 
    laplacian = (
        np_image[:-2, 1:-1] 
        + np_image[2:, 1:-1] 
        + np_image[1:-1, :-2] 
        + np_image[1:-1, 2:] 
        - 4 * np_image[1:-1, 1:-1] 
    )

    laplacian_var = np.var(laplacian)

    if gray_mean < 60:
        pass3 = True
    else:
        pass3 = laplacian_var > 30.0

    # Haze
    np_image = np.array(image.convert("HSV"))

    sat_mean = np.mean(np_image[:, :, 1])
    val_mean = np.mean(np_image[:, :, 2])

    is_hazy = (val_mean > 140) and (sat_mean < 35)
    pass4 = not is_hazy

    # Dark Channel
    # np_image = np.array(image.convert("RGB"))
    # dark_channel = np.min(np_image, axis=2)
    # pass5 =  np.mean(dark_channel) < 60

    
    
    return pass1 and pass2 and pass3 and pass4 and pass5

classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

print("=" * 60)
print("AKIDA INFERENCE - EuroSAT Dataset")
print("=" * 60)

# Incarca modelul
print("\nSe incarca modelul...")
model = akida.Model("src/data/models/joint_quality_land_v1/model.fbz")
model.summary()

# Statistici per clasa
correct = 0
total = 0
per_class_correct = {c: 0 for c in classes}
per_class_total = {c: 0 for c in classes}

print("\nSe ruleaza inferenta...\n")
start_total = time.perf_counter()

unclear_images = 0

for class_idx, class_name in enumerate(classes):
    folder = f"src/data/test_images_mixed/{class_name}"
    if not os.path.exists(folder):
        print(f"[SKIP] Folder negasit: {folder}")
        continue

    images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = images[:100]

    start_class = time.perf_counter()
    for img_file in images:
        img_path = os.path.join(folder, img_file)
        img = Image.open(img_path).convert("RGB").resize((64, 64))
        img_array = np.array(img, dtype=np.uint8)
        img_array = np.expand_dims(img_array, axis=0)

        if not is_image_clear(img):
            unclear_images += 1
            continue

        predicted = model.predict_classes(img_array)[0]

        per_class_total[class_name] += 1
        total += 1
        if predicted == class_idx:
            per_class_correct[class_name] += 1
            correct += 1

    end_class = time.perf_counter()
    class_acc = per_class_correct[class_name] / per_class_total[class_name] * 100 if per_class_total[class_name] > 0 else 0
    print(f"[{class_name:<25}] {per_class_correct[class_name]:>4}/{per_class_total[class_name]:<4} = {class_acc:>6.2f}%  | timp: {end_class-start_class:.2f}s")

end_total = time.perf_counter()
total_time = end_total - start_total

print("\n" + "=" * 60)
print("REZULTATE FINALE")
print("=" * 60)
print(f"Total imagini procesate : {total}")
print(f"Clasificate corect      : {correct}")
print(f"Clasificate gresit      : {total - correct}")
print(f"Acuratete globala       : {correct/total*100:.2f}%")
print(f"Timp total              : {total_time:.2f}s")
print(f"Timp mediu per imagine  : {total_time/total*1000:.2f}ms")
print(f"Imagini per secunda     : {total/total_time:.1f} FPS")
print(f"Imagini neclare         : {unclear_images}")
# Info hardware Akida
print("\n" + "=" * 60)
print("INFO HARDWARE AKIDA")
print("=" * 60)
try:
    device = akida.devices()[0]
    print(f"Device                  : {device}")
    print(f"Version                 : {device.version}")
    print(f"Spiking ops             : {model.statistics.get('total_sparsity', 'N/A')}")
except Exception as e:
    print(f"Info hardware: {e}")

print("\nDone!")
