# Synthetic Industrial Metal Surface Defect Dataset

## Quick Start

This dataset contains **15,000 grayscale images** of metal surfaces with 5 classes:
- `normal` - Defect-free surface
- `scratch` - Linear abrasion marks
- `crack` - Branching fracture patterns
- `rust` - Corrosion patches
- `hole` - Puncture defects

### Structure

```
├── images/
│   ├── train/          # 12,000 images (80%)
│   │   ├── normal/     # 2,400
│   │   ├── scratch/    # 2,400
│   │   ├── crack/      # 2,400
│   │   ├── rust/       # 2,400
│   │   └── hole/       # 2,400
│   └── val/            # 3,000 images (20%)
│       └── {class}/    # 600 each
├── metadata.csv        # Per-image metadata
├── config.json         # Generation configuration
└── README.md           # This file
```

### Image Specs
- **Resolution:** 256 × 256 px
- **Format:** PNG (grayscale)
- **Bits:** 8-bit (0-255)

---

## Loading the Data

### PyTorch
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.ImageFolder('images/train', transform)
val_data = datasets.ImageFolder('images/val', transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
```

### TensorFlow
```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    'images/train',
    image_size=(256, 256),
    color_mode='grayscale',
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'images/val',
    image_size=(256, 256),
    color_mode='grayscale',
    batch_size=32
)
```

### Metadata
```python
import pandas as pd
metadata = pd.read_csv('metadata.csv')
print(metadata['class'].value_counts())
```

---

## Metadata Columns

| Column | Description |
|--------|-------------|
| `filename` | Image filename |
| `class` | Defect type |
| `split` | `train` or `val` |
| `texture_type` | `brushed`, `matte`, or `grain` |
| `base_intensity` | Average brightness |
| `lighting_angle` | Light direction (degrees) |
| `defect_count` | Number of defects |
| `defect_coverage_pct` | % of image affected |
| `generation_seed` | For reproducibility |

---

## License

**CC BY 4.0** - Use freely with attribution.

---

## Citation

```
@dataset{synthetic_metal_defects_2026,
    title = {Synthetic Industrial Metal Surface Defect Dataset},
    author = {Tatheer Abbas},
    year = {2026}
}
```
