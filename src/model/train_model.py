import argparse
import logging
import os
import pickle
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.model.model import CommaModel
from src.visualization.visualize import plot_loss, plot_acc


fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def collate_fn(batch) -> dict:
    max_len = max(len(row["feature"]) for row in batch)

    input_ids = torch.empty((len(batch), max_len), dtype=torch.long)
    input_target = torch.empty((len(batch), max_len), dtype=torch.long)
    target_mask = torch.empty((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.empty((len(batch), max_len), dtype=torch.long)

    for idx, row in enumerate(batch):
        to_pad = max_len - len(row["feature"])
        input_ids[idx] = torch.cat((row["feature"], torch.zeros(to_pad)))
        input_target[idx] = torch.cat((row["target"], torch.zeros(to_pad)))
        target_mask[idx] = torch.cat((row["target_mask"], torch.zeros(to_pad)))
        attention_mask[idx] = torch.cat((row["attention_mask"], torch.zeros(to_pad)))

    return {
        'feature': input_ids,
        'target': input_target,
        'target_mask': target_mask,
        'attention_mask': attention_mask
    }


def train(model: nn.Module,
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

        y_predict = model(x, att_mask)

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


def fit(model: nn.Module,
        training_data_loader: DataLoader,
        validating_data_loader: DataLoader,
        epochs: int
        ) -> (list, list):
    """
    Основной цикл обучения по эпохам
    :param model: модель
    :param training_data_loader: набор для обучения
    :param validating_data_loader: набор для валидации
    :param epochs: число эпох обучения
    :return:
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, training_data_loader, criterion, optimizer, device)
        # val_loss, val_acc = test(model, testing_data_loader, criterion, device)
        # checkpoint(epoch, model, 'models')

        train_losses.append(train_loss)
        # val_losses.append(val_loss)
        train_accuracy.append(train_acc)
        # val_accuracy.append(val_acc)

    plot_acc(train_accuracy, val_accuracy)
    plot_loss(train_losses, val_losses)


def save_model(model: nn.Module, directory_path: str) -> None:
    """
    функция для сохранения состояния модели
    :param model: модель
    :param directory_path: имя директории
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    model_path = directory_path + 'model.torch'
    torch.save(model, model_path)
    logging.info(f'model saved: {model_path}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args_parser.add_argument('--epoch', default=1, type=int, dest='epoch')
    args = args_parser.parse_args()

    assert args.epoch > 0

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_processed_dir = fileDir + config['data']['processed']

    with open(data_processed_dir + f'train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(data_processed_dir + f'test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    logging.info(f'datasets loaded')

    train_size = len(train_dataset)
    validation_size = int(0.3 * train_size)

    train_data, valid_data = random_split(train_dataset, [train_size - validation_size, validation_size],
                                          generator=torch.Generator().manual_seed(42)
                                          )
    train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    comma_net = CommaModel(num_class=3)
    for param in comma_net.pretrained_transformer.parameters():
        param.requires_grad = False

    logging.info(f'model created')

    fit(comma_net, train_loader, valid_loader, args.epoch)
    save_model(comma_net, fileDir + config['models'])
