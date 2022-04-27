import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from feature.build_dataloader import build_dataloader
from model.model import CommaModel
from visualization.visualize import plot_loss


def train(epoch: int,
          model: nn.Module,
          training_data_loader: DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: str):
    """
    функция для обучения на одной эпохе
    :param epoch: номер эпохи
    :param model: модель для обучения
    :param training_data_loader: тренировочный DataLoader
    :param criterion: функция потерь
    :param optimizer: оптимизатор
    :param device: cuda или cpu
    :return:
    """
    train_loss = 0.0
    # train_accuracy = 0.0
    train_iteration = 0
    correct = 0.0
    total = 0.0

    model.train()
    for batch in tqdm(training_data_loader):
        x, y, y_mask, att_mask = batch['feature'], batch['target'], batch['target_mask'], batch['attention_mask']
        x = x.to(device)
        y = y.view(-1).to(device)
        y_mask = y_mask.view(-1).to(device)
        att_mask = att_mask.to(device)

        try:
            y_predict = model(x, att_mask)
        except:
            print(x.shape)
            print(att_mask.shape)
            continue
            raise NotImplementedError
        # print(y_predict.shape)

        y_predict = y_predict.view(-1, y_predict.shape[2])
        loss = criterion(y_predict, y)

        y_predict = torch.argmax(y_predict, dim=1).view(-1)
        correct += torch.sum(y_mask * (y_predict == y)).item()

        optimizer.zero_grad()
        train_loss += loss.item()
        train_iteration += 1
        loss.backward()

        optimizer.step()
        total += torch.sum(y_mask.view(-1)).item()

    train_loss /= train_iteration
    train_accuracy = correct / total

    return train_loss, train_accuracy


def fit(dataset: Dataset, epochs: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 32

    training_data_loader = build_dataloader(dataset, batch_size)

    # testing_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=batch_size,
    #                                  shuffle=False)

    model = CommaModel(num_class=3).to(device)

    for param in model.pretrained_transformer.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracy = []
    # val_accuracy = []

    for epoch in range(1, epochs):
        train_loss, train_acc = train(epoch, model, training_data_loader, criterion, optimizer, device)
        # val_loss, val_acc = test(model, testing_data_loader, criterion, device)
        # checkpoint(epoch, model, 'models')

        train_losses.append(train_loss)
        # val_losses.append(val_loss)
        train_accuracy.append(train_acc)
        # val_psnrs.append(val_acc)

    torch.save(model, 'models/model.pth')

    # plot_psnr(train_psnrs, val_psnrs)
    plot_loss(train_losses, val_losses)
