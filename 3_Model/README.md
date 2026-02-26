# 3_Model

This folder contains the final model definitions, training notebooks, evaluation results, and supporting scripts for the Valeo Microelectronics Defect Classification project.

## Contents

| File | Description |
|---|---|
| `model_definition_evaluation.ipynb` | Main training notebook — trains Models 1–3 (custom CNNs) and InceptionV3 (transfer learning), evaluates all on the held-out test set |
| `FinalModel_andDrift.ipynb` | Script to combine a well performing model with drift class detection to fully label a test dataset |
| `ManualDriftClassLabeling.ipynb` | Script for manually labeling a subset of public test images as the `Drift` class (out-of-distribution detection) |
| `InspectManuallyLabelledData.ipynb` | Script to inspect drift class images label with ManualDriftClassLabeling.ipynb |
| `Model_Performance_overview.csv` | Model performance comparison table (CSV format) |
| `Model_Performance_overview.xlsx` | Model performance comparison table (Excel format) |
| `INSTRUCTIONS.md` | Setup and usage instructions for running the notebooks |

## Models Trained

| Model | Architecture | Weighted F1 |
|---|---|---|
| Model 1 | 1× Conv + 1× Dense | 0.766 |
| Model 2 | 2× Conv + 1× Dense | 0.850 |
| **Model 3** | **3× Conv + 2× Dense** | **0.883** ✅ Best |
| InceptionV3 | Feature Extraction (frozen) | 0.827 |

All models were trained on a **balanced dataset** (undersampled to ~71 samples/class), evaluated at **299×299 px**, with **no data augmentation**.

## How to Run

1. Set `base_file_path` in `model_definition_evaluation.ipynb` to your local data directory
2. Configure hyperparameters at the top of the notebook (`balanced_flag`, `aug_flag`, `target_size`, etc.)
3. Run all cells — models, evaluation reports, and saved weights are automatically organized into subfolders under `model_evaluation/`

## Output Structure

```
model_evaluation/
└── ImgSz_299_NoAug_balanced/
    ├── model_1/
    │   ├── classification_report.csv
    │   └── model.keras
    ├── model_2/
    ├── model_3/
    └── InceptionV3_feat_extract/
```

## Notes

- The `Drift` class (out-of-distribution) is **not included** in supervised training labels. Use `ManualDriftClassLabeling.ipynb` to build a labeled subset for future work.
- Data augmentation was tested but found to **decrease** performance — the industrial images are highly structured and orientation-invariant transforms introduce misleading variation.
- See the root `README.md` for full project context, methodology, and results.
