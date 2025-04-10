import os
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import get_model
from datasets import get_dataset

# ==== Config ====
BATCH_SIZE = 256
EPOCH_MULTI = 50    # fragile 模型训练轮数
EPOCH_SINGLE = 100  # pirated 模型训练轮数
SAVE_DIR = './trained_models/finetune'
LOG_DIR = './logs/finetune'
PRETRAINED_PATH = './pretrained_models/teacher_model.pth'

# ==== Setup logging ====
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger('FineTune')
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.addHandler(logging.StreamHandler())
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
log_file = os.path.join(LOG_DIR, f'finetune_{timestamp}.log')
logger.addHandler(logging.FileHandler(log_file))

# ==== Fine-tune function ====
def finetune_model(teacher, train_loader, test_loader, mode="single", finetune_scope="all", iter_idx=None):
    teacher = teacher.cuda()
    teacher.train()
    best_acc = 0.0
    loss_func = nn.CrossEntropyLoss()

    # 设置学习率：fragile 用 1e-4，pirated 用 5e-4
    if finetune_scope == 'all':
        lr = 1e-4 if mode == "multi" else 5e-4
        optimizer = optim.Adam(teacher.parameters(), lr=lr)
    elif finetune_scope == 'last':
        lr = 1e-4 if mode == "multi" else 5e-4
        optimizer = optim.Adam(teacher.classifier.parameters(), lr=lr)
    else:
        raise ValueError("finetune_scope must be 'all' or 'last'")

    logger.info(f"[{mode.upper()}] Finetune Scope: {finetune_scope}, Learning Rate: {lr}")

    total_epochs = EPOCH_MULTI if mode == "multi" else EPOCH_SINGLE

    for epoch in range(total_epochs):
        teacher.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.type(torch.FloatTensor).cuda(), y.long().cuda()
            outputs = teacher(x)
            loss = loss_func(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                logger.info(f"[{mode}] Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item():.4f}")
                print(f"Epoch: {epoch + 1}, Iter: {i}, Loss: {loss.item():.4f}")

        # Evaluation
        teacher.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.type(torch.FloatTensor).cuda(), y.long().cuda()
                preds = teacher(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        logger.info(f"[{mode}] Epoch {epoch+1}, Accuracy: {acc:.2f}%")
        print(f"Epoch {epoch+1}, Accuracy: {acc:.2f}%")

        # Save model
        if mode == "multi":
            if acc > best_acc and iter_idx is not None:
                torch.save(teacher.state_dict(), os.path.join(SAVE_DIR, f"finetune_fragile_{iter_idx}.pth"))
                best_acc = acc
        elif mode == "single":
            if (epoch + 1) % 5 == 0:
                torch.save(teacher.state_dict(), os.path.join(SAVE_DIR, f"finetune_pirated_epoch{epoch + 1}.pth"))

    return best_acc


if __name__ == "__main__":
    # Load dataset
    train_loader, test_loader = get_dataset("mnist", BATCH_SIZE, augment=True, role="attacker")

    # === Mode 1: Fragile Finetune - 20个模型，前10个训练全部层，后10个只微调最后层 ===
    for i in range(20):
        teacher = get_model("lenet", "mnist", False)
        teacher.load_state_dict(torch.load(PRETRAINED_PATH))
        finetune_scope = "all" if i < 10 else "last"

        logger.info(f"[FRAGILE] Begin training model {i}, Finetune Scope: {finetune_scope}")
        print(f"Begin fragile training model {i}, Finetune Scope: {finetune_scope}")

        acc = finetune_model(teacher, train_loader, test_loader, mode="multi", finetune_scope=finetune_scope, iter_idx=i)

        logger.info(f"[FRAGILE] Model {i} completed, Best Accuracy: {acc:.2f}%")
        print(f"Fragile model {i} best accuracy: {acc:.2f}%")

    # === Mode 2: Pirated Finetune - 单个模型，训练所有层，保存每 5 epoch ===
    pirated_teacher = get_model("lenet", "mnist", False)
    pirated_teacher.load_state_dict(torch.load(PRETRAINED_PATH))

    logger.info("[PIRATED] Begin long-term finetune")
    print("Begin pirated model finetune...")

    _ = finetune_model(pirated_teacher, train_loader, test_loader, mode="single", finetune_scope="last")
