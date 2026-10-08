### Clarity Filter

#### Clean Images
|Baseline Model|Attuned Model (no filter)|Attuned Model + Extreme Values| Attuned Model + Cloud Deck| Attuned Model + Spatial Gradient and Haze| New Attuned Model after Retraining | New Model after Class Changes |
|---|---|---|---|---|---|---|
|96%|95%|95%|95%|95%|92%|94.88%|

#### Distorted Images
|Baseline Model|Attuned Model (no filter)|Attuned Model + Extreme Values| Attuned Model + Cloud Deck| Attuned Model + Spatial Gradient and Haze| New Attuned Model after Retraining |
|---|---|---|---|---|---|
|44.4%|46.1%|47.4%|47%|49.77%|67.13%|

#### Mixed Images
|Baseline Model|Attuned Model (no filter)|Attuned Model + Extreme Values| Attuned Model + Cloud Deck| Attuned Model + Spatial Gradient and Haze| New Attuned Model after Retraining | New Model after Class Changes
|---|---|---|---|---|---|
|70.2%|69.9%|71.3%|71.75%|75.72%|81.26%|83.60%|

After merging Pasture and HerbaceousVegetation and removing SeaLake classes:
Overall metrics:
  accuracy:        0.9404
  macro precision: 0.9428
  macro recall:    0.9414
  macro F1:        0.9418

  ### 10-Class vs 8-Class Architecture Comparison

| Metric | 10-Class Baseline | 8-Class Retrained (Our Model) | Delta |
| :--- | :--- | :--- | :--- |
| **Number of Classes** | 10 | 8 | -2 classes |
| **Validation / Test Acc** | ~81.3% (mixed) | **94.04%** | **+12.7%** |
| **Macro F1 Score** | ~80.5% | **94.18%** | **+13.7%** |
| **Downlink Telemetry (1 frame)** | 46.6 KB (CSV) | **0.76 KB (3-bit) / 2.03 KB (uint8)** | **-95.6% to -98.4%** |
| **Akida SNN Test Acc (.fbz)** | ~80% | **93.39%** | **+13.4%** |