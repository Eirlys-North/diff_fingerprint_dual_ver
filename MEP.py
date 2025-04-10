import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import logging
import time
from math import isnan

from models import get_model
from datasets import get_dataset

BATCH_SIZE = 128
EPOCH = 10
SAVE_DIR = './trained_models'
LOG_DIR = './logs'
PRETRAINED_PATH = './pretrained_models/teacher_model.pth'

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

def train_student_model(iter_idx, teacher_model, model_type, train_loader, test_loader, logger):
    teacher_model.eval()
    model = reset(model_type)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_accuracy = 0.0
    restart_count = 0
    alpha = 0.9
    T = 20

    for epoch in range(EPOCH):
        model.train()
        for i, (x, _) in enumerate(train_loader):
            x = x.type(torch.FloatTensor).cuda()
            with torch.no_grad():
                teacher_output = teacher_model(x)
            pred_labels = torch.max(teacher_output, 1)[1]
            student_output = model(x)

            loss_kd = nn.KLDivLoss()(F.log_softmax(student_output / T, dim=1),
                                     F.softmax(teacher_output / T, dim=1)) * (alpha * T * T)
            loss_ce = F.cross_entropy(student_output, pred_labels) * (1. - alpha)
            loss = loss_kd + loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                logger.info(f"Epoch {epoch+1}, Iteration {i}, Loss: {loss.item():.4f}")
                print(f"Epoch: {epoch+1}, Iteration: {i}, Loss: {loss.item():.4f}")

        if isnan(loss.item()):
            logger.warning("Loss is NaN. Resetting model.")
            model = reset(model_type)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.type(torch.FloatTensor).cuda(), y.cuda()
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        logger.info(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")
        print(f"Epoch: {epoch+1}, Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"student_kd_{model_type}_{iter_idx}.pth"))
            best_accuracy = accuracy

        if accuracy < 0.12:
            restart_count += 1
        if restart_count > 20:
            logger.warning("Low accuracy for too long. Resetting model.")
            model = reset(model_type)
            restart_count = 0

    return best_accuracy

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = os.path.join(LOG_DIR, f'kd_training_{timeticks}.log')
    logger = logging.getLogger('KnowledgeDistillation')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_file))

    teacher_model = get_model("lenet", "mnist", pretrained=False)
    teacher_model.load_state_dict(torch.load(PRETRAINED_PATH))
    teacher_model.cuda()

    train_loader, test_loader = get_dataset("mnist", BATCH_SIZE, augment=True, role="attacker")

    accuracies = []

    for iter_idx in range(20):
        if iter_idx < 5:
            model_type = 'vgg'
        elif iter_idx < 10:
            model_type = 'resnet'
        elif iter_idx < 15:
            model_type = 'dense'
        else:
            model_type = 'mobile'

        logger.info(f"Start training: Iter {iter_idx}, Model: {model_type}")
        print(f"Begin training model: {iter_idx}, student model: {model_type}")

        acc = train_student_model(iter_idx, teacher_model, model_type, train_loader, test_loader, logger)
        accuracies.append(acc)

        logger.info(f"Completed model {iter_idx}, Accuracy: {acc:.4f}")
        print(f"Model {iter_idx} finished with accuracy: {acc:.4f}")

    print("Final Accuracies:", accuracies)
