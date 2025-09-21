# coding:utf-8
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_one_batch(images, labels, device, model, criterion, optimizer, epoch, batch_idx):
    """训练一个 batch 并返回日志"""
    images, labels = images.to(device), labels.to(device)

    # 前向传播与计算损失
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 反向传播与更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 转 numpy 计算指标
    preds = torch.argmax(outputs, 1).cpu().numpy()
    loss = loss.item()
    labels = labels.cpu().numpy()

    return {
        "epoch": epoch,
        "batch": batch_idx,
        "train_loss": loss,
        "train_accuracy": accuracy_score(labels, preds),
    }


def evaluate_testset(test_loader, device, model, criterion, epoch):
    """在整个测试集上评估并返回日志"""
    model.eval()
    loss_list, labels_list, preds_list = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels).item()

            preds = torch.argmax(outputs, 1).cpu().numpy()
            labels = labels.cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

    return {
        "epoch": epoch,
        "test_loss": np.mean(loss_list),
        "test_accuracy": accuracy_score(labels_list, preds_list),
        "test_precision": precision_score(labels_list, preds_list, average='macro', zero_division=0),
        "test_recall": recall_score(labels_list, preds_list, average='macro', zero_division=0),
        "test_f1": f1_score(labels_list, preds_list, average='macro', zero_division=0),
    }
