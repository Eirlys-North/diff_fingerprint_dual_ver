This is our work submitted to the 33rd ACM International Conference on Multimedia (ACM MM 2025).

# Components & Usage

## Source Model
A pre-trained LeNet model is provided as the source model:
```python
from models import get_model
import torch

model = get_model("lenet", "mnist", False)
model.load_state_dict(torch.load("./pretrained_models/teacher_model.pt"))
model.eval()
```

## Pirated Model & Irrelevant Models

```
# Fine-tune attack (Finetune-L & Finetuen-A)
python Finetune.py

# Transfer learning attack
python TL.py

# Fine-pruning attack
python FP.py

# Model extraction via labeling
python MEL.py

# Model extraction via probing
python MEP.py

# Train irrelevant model
python irrelevant.py

```

## Diffusion Model

The original DDPM implementation is in script.py.

To fine-tune the diffusion model, run finetuning_diff.py, 
it also provides multiple ablation settings.
You can also directly use the fine-tuned model from [here.](https://drive.google.com/file/d/1vy8LEFfr_Agsl8qxBmkqua9enw0qvnLA/view?usp=share_link)

## Fingerprint Sample Selection

Run sample selection using the fine-tuned diffusion model:
```
python selection.py
```

## Fingerprint Set
We provide the fingerprint set used in the paper:
```
./data/generate_DiffFP_boundary_100.h5
```
There are total 200 samples: 
* Even indices (0, 2, 4, ...) are the final fingerprint samples x_w
* Odd indices are the intermediate sample x_{w-1}, not used in evaluation

Load the fingerprint set:
```python
import h5py, numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

class FingerprintDataset(Dataset):
    def __init__(self, h5_file_path):
        with h5py.File(h5_file_path, 'r') as f:
            self.images = np.array(f['data'])
            self.labels = np.array(f['label'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        image = self.images[item]
        image = np.transpose(image, (1, 2, 0))  # (C, H, W) → (H, W, C)
        image = transform_test(image)
        return image, label

# Load x_w (even indices only)
dataset = FingerprintDataset('./data/generate_DiffFP_boundary_100.h5')
subset_indices = list(range(0, 200, 2))  # Select x_w only
subset = Subset(dataset, subset_indices)
loader = DataLoader(subset, batch_size=128, shuffle=False)
```
