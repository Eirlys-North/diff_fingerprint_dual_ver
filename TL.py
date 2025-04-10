import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from math import isnan
import logging
import time
from models import get_model
from datasets import get_dataset

BATCH_SIZE = 256
EPOCH = 50

# Change this to your preferred model saving directory
SAVE_DIR = './trained_models'
LOG_DIR = './logs'

def train_student_model(iter_idx, teacher_model, train_loader, log_dir):
    teacher_model = teacher_model.cuda()
    model = teacher_model
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    logger = logging.getLogger(f'TL_{iter_idx}')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear any existing handlers
    console_handler = logging.StreamHandler()
    timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_handler = logging.FileHandler(os.path.join(log_dir, f'train_log_{iter_idx}_{timeticks}.log'))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    model.train()
    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                log_msg = f"Epoch: {epoch + 1}, iteration: {i}, loss: {loss.item():.4f}"
                print(log_msg)
                logger.info(log_msg)

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"trained_model_iter_{iter_idx}.pth"))

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Change model and dataset name as needed
    model_name = "lenet"
    dataset_name = "kmnist"

    teacher_model = get_model(model_name, dataset_name, pretrained=False)
    # Load pretrained weights from a user-specified path
    pretrained_path = "./pretrained_models/teacher_model.pth"
    teacher_model.load_state_dict(torch.load(pretrained_path))
    teacher_model.eval()

    train_loader, test_loader = get_dataset(dataset_name, BATCH_SIZE, augment=False)

    for iter_idx in range(5):
        print(f"Begin training model: {iter_idx}")
        train_student_model(iter_idx, teacher_model, train_loader, LOG_DIR)
