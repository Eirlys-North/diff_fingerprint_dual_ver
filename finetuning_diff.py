import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Set your desired GPU index
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from models import get_model
from script import DDPM, ContextUnet  # Using the same script as previous


def load_teacher_model(path, device):
    model = get_model("lenet", "mnist", False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_half_mnist_dataset():
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    targets = dataset.targets
    indices_per_class = {i: [] for i in range(10)}
    for idx, label in enumerate(targets):
        indices_per_class[label.item()].append(idx)

    subset_indices = []
    for class_indices in indices_per_class.values():
        mid_point = len(class_indices) // 2
        subset_indices.extend(class_indices[:mid_point])

    return Subset(dataset, subset_indices)


def evaluate_and_save(ddpm, teacher, x_gen, save_dir, ep, guide_w):
    transform_test = transforms.Normalize(mean=(0.5,), std=(0.5,))
    x_real = torch.Tensor(x_gen.shape).to(x_gen.device)
    x_all = torch.cat([x_gen, x_real])
    grid = make_grid(x_all * -1 + 1, nrow=10)
    save_path = os.path.join(save_dir, f"image_ep{ep}_w{guide_w}.png")
    save_image(grid, save_path)
    print('Saved image at', save_path)

    teacher_outputs = teacher(transform_test(x_gen))
    predicted_labels = torch.argmax(teacher_outputs, dim=1)
    print("Teacher predictions:", predicted_labels)
    prob = F.softmax(teacher_outputs, dim=1)
    top2_probs, _ = torch.topk(prob, 2, dim=1)
    margin = top2_probs[:, 0] - top2_probs[:, 1]
    print("Margins:", margin)


def run_ablation():
    ablation_configs = {
        "no_pro_no_grad_no_feature": {
            'lambda_mse': 1.0, 'lambda_rec': 1.0, 'lambda_entropy': 0, 'lambda_margin': 0,
            'lambda_feature': 0.0, 'lambda_grad': 0.0, 'lambda_cls': 0,
        },
        "no_grad_no_feature": {
            'lambda_mse': 1.0, 'lambda_rec': 1.0, 'lambda_entropy': 0.1, 'lambda_margin': 0.15,
            'lambda_feature': 0.0, 'lambda_grad': 0.0, 'lambda_cls': 0.001
        },
        "no_feature": {
            'lambda_mse': 1.0, 'lambda_rec': 1.0, 'lambda_entropy': 0.1, 'lambda_margin': 0.15,
            'lambda_feature': 0.0, 'lambda_grad': 0.01, 'lambda_cls': 0.001
        },
        "no_grad": {
            'lambda_mse': 1.0, 'lambda_rec': 1.0, 'lambda_entropy': 0.1, 'lambda_margin': 0.15,
            'lambda_feature': 0.05, 'lambda_grad': 0.0, 'lambda_cls': 0.001
        },
        "full": {
            'lambda_mse': 1.0, 'lambda_rec': 1.0, 'lambda_entropy': 0.1, 'lambda_margin': 0.15,
            'lambda_feature': 0.05, 'lambda_grad': 0.01, 'lambda_cls': 0.001
        }
    }

    for name, lambda_dict in ablation_configs.items():
        print(f"\n===> Running ablation setting: {name}")
        save_dir = os.path.join('./ablation_results', name)
        os.makedirs(save_dir, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        teacher = get_model("lenet", "mnist", False)
        teacher.load_state_dict(torch.load(PRETRAINED_PATH)
        teacher_model.eval()
        teacher = teacher.to(device)

        ddpm = DDPM(
            nn_model=ContextUnet(in_channels=1, n_feat=256, n_classes=10),
            betas=(1e-4, 0.02), n_T=500, device=device, drop_prob=0.1
        ).to(device)
        ddpm.load_state_dict(torch.load("./diffusion_outputs/model_99.pth", map_location=device))

        dataset = get_half_mnist_dataset()
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

        for ep in range(20):
            ddpm.train()
            pbar = tqdm(dataloader, desc=f"Epoch {ep}")
            for x, c in pbar:
                optimizer.zero_grad()
                x, c = x.to(device), c.to(device)
                loss = fine_tune_loss(ddpm, x, c, teacher, lambda_dict)
                loss.backward()
                optimizer.step()
                pbar.set_description(f"[{name}] Loss: {loss.item():.4f}")

            ddpm.eval()
            with torch.no_grad():
                x_gen, _ = ddpm.sample(10, (1, 28, 28), device, guide_w=1.0)
                evaluate_and_save(ddpm, teacher, x_gen, save_dir, ep, guide_w=1.0)

            if ep == 19:
                model_path = os.path.join(save_dir, f"model_{name}.pth")
                torch.save(ddpm.state_dict(), model_path)
                print("Saved model at", model_path)


if __name__ == "__main__":
    run_ablation()
