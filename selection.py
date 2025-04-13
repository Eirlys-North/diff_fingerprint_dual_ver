import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import h5py

from script import DDPM, ContextUnet
from models import get_model

device = "cuda:0"
n_feat = 256
n_T = 500
n_classes = 10
lrate = 1e-4
num_samples_needed = 100
save_path = "./data/generate_DiffFP_boundary_100.h5"
teacher_ckpt = "./pretrained_models/teacher_model.pth"
ddpm_ckpt = "./diffusion/model_19half.pth"

teacher = get_model("lenet", "mnist", False)
teacher.load_state_dict(torch.load(teacher_ckpt))
teacher = teacher.to(device)
teacher.eval()


ddpm = DDPM(
    nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    drop_prob=0.1
)
ddpm.load_state_dict(torch.load(ddpm_ckpt))
ddpm.eval()


transform_test = transforms.Normalize(mean=(0.5,), std=(0.5,))


all_images = []
all_labels = []
total = 0
num = 0

while num < num_samples_needed:
    with torch.no_grad():
        x, x_p = ddpm.sample(10, (1, 28, 28), device, progressive=True, guide_w=1.0)
        output = teacher(transform_test(x).to(device))
        pred = torch.argmax(output, dim=1)

        for i in range(10):
            total += 1
            xx = x[i].unsqueeze(0)
            prob = F.softmax(teacher(transform_test(xx).to(device)), dim=1)
            top2_probs, _ = torch.topk(prob, 2, dim=1)
            margin = top2_probs[:, 0] - top2_probs[:, 1]

            if i == pred[i].item() and margin.item() <= 0.1:
                for j in range(n_T - 1, -1, -1):
                    tmp_x = x_p[j, i].unsqueeze(0)
                    tmp_output = teacher(transform_test(tmp_x).to(device))
                    tmp_pred = torch.argmax(tmp_output, dim=1)
                    if tmp_pred.item() != pred[i].item():
                        all_images.append(x_p[j, i].unsqueeze(0).cpu())
                        all_images.append(x_p[j + 1, i].unsqueeze(0).cpu())
                        all_labels.extend([tmp_pred.item(), pred[i].item()])
                        num += 1
                        break
            if num == num_samples_needed:
                break
        if num == num_samples_needed:
            break

print(f"Total samples evaluated: {total}")

all_images_np = torch.cat(all_images, dim=0).numpy()
all_labels_np = np.array(all_labels)

os.makedirs(os.path.dirname(save_path), exist_ok=True)
with h5py.File(save_path, 'w') as hf:
    hf.create_dataset('data', data=all_images_np)
    hf.create_dataset('label', data=all_labels_np)
print(f"Saved {num*2} images to {save_path}")
