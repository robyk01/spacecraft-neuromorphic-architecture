from pathlib import Path

bin_path = Path(__file__).parent / "data" / "satellite_image.bin"
with open(bin_path, "rb") as f:
    class_ids = list(f.read())

classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "PermanentCrop", "Residential", "River"
]

results = [classes[cid] for cid in class_ids]
print(f"Successfully decoded {len(results)} predictions from binary.")
print(f"Sample (first 10): {results[:10]}")

output_csv = bin_path.parent / "ground_decoded.csv"
with open(output_csv, "w") as f:
    f.write("patch_id,class_id,class_name\n")
    for i, (cid, name) in enumerate(zip(class_ids, results)):
        f.write(f"{i},{cid},{name}\n")
print(f"Saved ground CSV to: {output_csv.name}")