import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from copy import deepcopy
import logging

from models import get_model
from datasets import get_dataset

# ==== Config ====
BATCH_SIZE = 256
PRUNE_RATIO = 0.90
SAVE_DIR = './trained_models/fine_pruning'
LOG_DIR = './logs/fine_pruning'
PRETRAINED_PATH = './pretrained_models/teacher_model.pt'

# ==== Logging ====
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger('FinePruning')
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.addHandler(logging.StreamHandler())
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(os.path.join(LOG_DIR, f'{timestamp}.log'))
logger.addHandler(file_handler)

# ==== Hook Class ====
class FeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output
    def close(self):
        self.hook.remove()

# ==== Helper Functions ====
def idx_change(idx, neuron_num):
    total = 0
    for i in range(neuron_num.shape[0]):
        total += neuron_num[i]
        if idx < total:
            layer_num = i
            layer_idx = idx - (total - neuron_num[i])
            return layer_num, layer_idx

def prune_neuron(mask_list, idx, neuron_num):
    layer_num, layer_idx = idx_change(idx, neuron_num)
    mask_list[layer_num].weight_mask[layer_idx] = 0

def find_smallest_neuron(hook_list, prune_list):
    activation_list = []
    for hook in hook_list:
        activation = hook.output
        for i in range(activation.shape[1]):
            act_value = torch.mean(torch.abs(activation[:, i, :, :]))
            activation_list.append(act_value)

    remaining_idxs = [i for i in range(len(activation_list)) if i not in prune_list]
    activation_values = torch.tensor([activation_list[i] for i in remaining_idxs])
    min_idx = remaining_idxs[torch.argmin(activation_values)]
    return min_idx

def finetune_step(model, dataloader, criterion):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.long().cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i + 1) * inputs.size(0) >= 2056:
            break

def value(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.cuda(), y.long().cuda()
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def run_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.cuda()
            _ = model(x)

def fine_pruning(model, train_loader, test_loader):
    model = model.cuda()
    module_list = []
    hook_list = []
    neuron_num = []

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module_list.append(module)
            neuron_num.append(module.out_channels)
            hook_list.append(FeatureHook(module))
    neuron_num = np.array(neuron_num)

    # Setup pruning masks
    mask_list = [prune.identity(m, "weight") for m in module_list]

    total_neurons = neuron_num.sum()
    logger.info(f"Total neurons: {total_neurons}")
    print(f"Total neurons: {total_neurons}")

    prune_list = []
    acc_log = []
    initial_acc = value(model, test_loader)

    for i in range(int(np.floor(0.8 * total_neurons))):
        if i % 20 == 0:
            run_model(model, train_loader)
        prune_idx = find_smallest_neuron(hook_list, prune_list)
        prune_list.append(prune_idx)
        prune_neuron(mask_list, prune_idx, neuron_num)

        if i % 50 == 0:
            finetune_step(model, train_loader, criterion=nn.CrossEntropyLoss())
            current_acc = value(model, test_loader)
            acc_log.append([i, current_acc])
            logger.info(f"Step {i}: Init Acc: {initial_acc:.4f}, Current Acc: {current_acc:.4f}")
            print(f"Step {i}: Init Acc: {initial_acc:.4f}, Current Acc: {current_acc:.4f}")

        if (np.floor(20 * i / total_neurons) - np.floor(20 * (i - 1) / total_neurons)) == 1:
            iter_idx = int(np.floor(20 * i / total_neurons))
            model_path = os.path.join(SAVE_DIR, f"prune_model_{iter_idx}.pth")
            torch.save(model, model_path)
            logger.info(f"Model saved at pruning step {i} (iter {iter_idx})")

if __name__ == "__main__":
    model = get_model("lenet", "mnist", pretrained=False)
    model.load_state_dict(torch.load(PRETRAINED_PATH))
    model.eval()
    model.cuda()

    train_loader, test_loader = get_dataset("mnist", BATCH_SIZE, augment=True, role="attacker")
    fine_pruning(model, train_loader, test_loader)
