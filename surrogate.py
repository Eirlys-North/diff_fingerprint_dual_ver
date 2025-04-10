import os
import time
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import get_model
from datasets import get_dataset

# Training configuration
BATCH_SIZE = 256
EPOCH = 10
MODEL_NAME = "lenet"
DATASET_NAME = "mnist"
SAVE_DIR = "./trained_models/surrogate"
LOG_DIR = "./logs/surrogate"
PRETRAINED_PATH = "./pretrained_models/teacher_model.pth"

def train_student_model(iter_idx, teacher_model, device="cuda"):
    teacher_model.to(device)
    teacher_model.eval()

    train_loader, test_loader = get_dataset(DATASET_NAME, BATCH_SIZE, augment=True, role="defender")

    student_model = get_model(MODEL_NAME, DATASET_NAME, pretrained=False)
    student_model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=5e-4)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_accuracy = 0.0

    for epoch in tqdm(range(EPOCH), desc=f"Training Iter {iter_idx}"):
        student_model.train()
        train_losses = []

        for x, _ in train_loader:
            x = x.type(torch.FloatTensor).to(device)
            with torch.no_grad():
                teacher_output = teacher_model(x)
            pseudo_labels = teacher_output.argmax(dim=1).detach()
            student_output = student_model(x)

            loss = loss_func(student_output, pseudo_labels)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = sum(train_losses) / len(train_loader)
        logger.info(f"[Iter {iter_idx}] Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # Evaluation
        student_model.eval()
        correct, total = 0, 0
        test_losses = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.type(torch.FloatTensor).to(device), y.to(device)
                output = student_model(x)
                loss = loss_func(output, output.argmax(dim=1).detach())
                test_losses.append(loss.item())

                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        accuracy = correct / total
        avg_test_loss = sum(test_losses) / len(test_loader)
        logger.info(f"[Iter {iter_idx}] Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            save_path = os.path.join(SAVE_DIR, f"surrogate_{iter_idx}.pth")
            torch.save(student_model.state_dict(), save_path)
            best_accuracy = accuracy

    return best_accuracy


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("SurrogateTraining")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = os.path.join(LOG_DIR, f"surrogate_train_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)

    teacher_model = get_model(MODEL_NAME, DATASET_NAME, pretrained=False)
    teacher_model.load_state_dict(torch.load(PRETRAINED_PATH))

    for i in range(3):
        logger.info(f"Begin training surrogate model {i}")
        print(f"Begin training surrogate model {i}")
        final_acc = train_student_model(i, teacher_model)
        logger.info(f"Surrogate model {i} final accuracy: {final_acc:.4f}")
        print(f"Model {i} finished with accuracy: {final_acc:.4f}")
