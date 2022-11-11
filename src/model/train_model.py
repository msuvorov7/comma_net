import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.model.model import CommaModel
from src.visualization.visualize import plot_loss, plot_acc, plot_conf_matrix


fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def collate_fn(batch) -> dict:
    """
    Обработчик батча перед входом в модель.
    Забивает предложения pad-токенами до длинны самого длинного
    предложения в батче
    :param batch: батч данных
    :return:
    """
    max_len = max(len(row["feature"]) for row in batch)

    input_ids = torch.empty((len(batch), max_len), dtype=torch.long)
    input_target = torch.empty((len(batch), max_len), dtype=torch.long)
    word_mask = torch.empty((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.empty((len(batch), max_len), dtype=torch.long)

    for idx, row in enumerate(batch):
        to_pad = max_len - len(row["feature"])
        input_ids[idx] = torch.cat((row["feature"], torch.zeros(to_pad)))
        input_target[idx] = torch.cat((row["target"], torch.zeros(to_pad)))
        word_mask[idx] = torch.cat((row["word_mask"], torch.zeros(to_pad)))
        attention_mask[idx] = torch.cat((row["attention_mask"], torch.zeros(to_pad)))

    return {
        'feature': input_ids,
        'target': input_target,
        'word_mask': word_mask,
        'attention_mask': attention_mask
    }


def train(model: nn.Module,
          training_data_loader: DataLoader,
          validating_data_loader: DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: str):
    """
    функция для обучения на одной эпохе
    :param model: модель для обучения
    :param training_data_loader: тренировочный DataLoader
    :param criterion: функция потерь
    :param optimizer: оптимизатор
    :param device: cuda или cpu
    :return:
    """
    train_loss = 0.0
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    model.train()
    for batch in tqdm(training_data_loader):
        x, y, w_mask, att_mask = batch['feature'], batch['target'], batch['word_mask'], batch['attention_mask']
        x = x.to(device)
        y = y.view(-1).to(device)
        w_mask = w_mask.view(-1).to(device)
        att_mask = att_mask.to(device)

        y_predict = model(x, att_mask)

        y_predict = y_predict.view(-1, y_predict.shape[2])
        loss = criterion(y_predict, y)

        y_predict = torch.argmax(y_predict, dim=1).view(-1)
        correct += torch.sum(w_mask * (y_predict == y)).item()

        optimizer.zero_grad()
        train_loss += loss.item()
        loss.backward()

        optimizer.step()
        total += torch.sum(w_mask.view(-1)).item()

    train_loss /= len(training_data_loader)
    train_accuracy = correct / total

    correct = 0.0
    total = 0.0
    model.eval()
    for batch in tqdm(validating_data_loader):
        x, y, w_mask, att_mask = batch['feature'], batch['target'], batch['word_mask'], batch['attention_mask']
        x = x.to(device)
        y = y.view(-1).to(device)
        w_mask = w_mask.view(-1).to(device)
        att_mask = att_mask.to(device)

        y_predict = model(x, att_mask)

        y_predict = y_predict.view(-1, y_predict.shape[2])
        loss = criterion(y_predict, y)

        y_predict = torch.argmax(y_predict, dim=1).view(-1)
        correct += torch.sum(w_mask * (y_predict == y)).item()

        val_loss += loss.item()
        total += torch.sum(w_mask.view(-1)).item()

    val_loss /= len(validating_data_loader)
    val_accuracy = correct / total

    return train_loss, train_accuracy, val_loss, val_accuracy


def test(model: nn.Module,
         test_data_loader: DataLoader,
         num_classes: int,
         device: str,
         ):
    correct = 0.0
    total = 0.0

    tp = np.zeros(1 + num_classes, dtype=np.int64)
    fp = np.zeros(1 + num_classes, dtype=np.int64)
    fn = np.zeros(1 + num_classes, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    model.eval()
    for batch in tqdm(test_data_loader):
        x, y, w_mask, att_mask = batch['feature'], batch['target'], batch['word_mask'], batch['attention_mask']
        x = x.to(device)
        y = y.view(-1).to(device)
        w_mask = w_mask.view(-1).to(device)
        att_mask = att_mask.to(device)

        y_predict = model(x, att_mask)
        y_predict = y_predict.view(-1, y_predict.shape[2])
        y_predict = torch.argmax(y_predict, dim=1).view(-1)

        correct += torch.sum(w_mask * (y_predict == y)).item()
        total += torch.sum(w_mask.view(-1)).item()

        for i in range(y.shape[0]):
            if w_mask[i] == 0:
                # we can ignore this because we know there won't be
                # any punctuation in this position since we created
                # this position due to padding or sub-word tokenization
                continue

            cor = y[i]
            prd = y_predict[i]
            if cor == prd:
                tp[cor] += 1
            else:
                fn[cor] += 1
                fp[prd] += 1
            cm[cor][prd] += 1

    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = correct / total

    return precision, recall, f1, accuracy, cm


def fit(model: nn.Module,
        training_data_loader: DataLoader,
        validating_data_loader: DataLoader,
        testing_data_loader: DataLoader,
        epochs: int
        ) -> (list, list):
    """
    Основной цикл обучения по эпохам
    :param model: модель
    :param training_data_loader: набор для обучения
    :param validating_data_loader: набор для валидации
    :param testing_data_loader: набор для теста
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
        train_loss, train_acc, val_loss, val_acc = train(model,
                                                         training_data_loader,
                                                         validating_data_loader,
                                                         criterion,
                                                         optimizer,
                                                         device
                                                         )
        # checkpoint(epoch, model, 'models')
        print('Epoch: {}, Training Loss: {}, Validation Loss: {}, VAL_ACC: {}'.format(epoch,
                                                                                      train_loss,
                                                                                      val_loss,
                                                                                      val_acc)
              )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)

    precision, recall, f1, accuracy, cm = test(model, testing_data_loader, 3, device)
    print('precision: {}, recall: {}, f1: {}, accuracy: {}'.format(precision, recall, f1, accuracy))
    logging.info(f'confusion matrix:\n{cm}')

    plot_conf_matrix(cm, ['space', 'comma', 'dot'])
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

    fit(comma_net, train_loader, valid_loader, test_loader, args.epoch)
    save_model(comma_net, fileDir + config['models'])
