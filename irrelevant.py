import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import logging
import time

from models import get_model
from datasets import get_dataset

# Training Configuration
BATCH_SIZE = 128
EPOCH = 50
SAVE_DIR = './trained_models/irrelevant'
LOG_DIR = './logs/irrelevant'
PRETRAINED_PATH = './pretrained_models/teacher_model.pth'

# Model reset function
def reset(model_type):
    if model_type == 'resnet':
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_type == "vgg":
        model = models.vgg13(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    elif model_type == 'dense':
        model = models.densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, 10)
    elif model_type == 'mobile':
        model = models.mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model.cuda()

# Training function
def train_irrelevant_model(iter_idx, model_type):
    accu_best = 0
    train_loader, test_loader = get_dataset("mnist", BATCH_SIZE, augment=True, role="defender")
    
    model = reset(model_type)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCH):
        model.train()
        train_loss = []

        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = loss_func(output, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                logger.info(f"Epoch:{epoch + 1}, Iteration:{i}, Loss:{loss.item():.4f}")
                print(f"Epoch: {epoch + 1}, Iteration: {i}, Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%")
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%")

        # Save model with best accuracy
        if accuracy > accu_best:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"clean_model_{iter_idx}.pth"))
            accu_best = accuracy

    return accu_best

# Logger setup
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logger = logging.getLogger('Source Model')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(os.path.join(LOG_DIR, f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Training multiple models
for iter_idx in range(20):
    if iter_idx < 5:
        model_type = 'vgg'
    elif 5 <= iter_idx < 10:
        model_type = 'resnet'
    elif 10 <= iter_idx < 15:
        model_type = 'dense'
    else:
        model_type = 'mobile'

    logger.info(f"Begin training model {iter_idx}, student model: {model_type}")
    print(f"Begin training model: {iter_idx}, student model: {model_type}")
    
    accuracy = train_irrelevant_model(iter_idx, model_type)
    logger.info(f"Model {iter_idx} trained with accuracy: {accuracy:.2f}%")
    print(f"Model {iter_idx} trained with accuracy: {accuracy:.2f}%")
