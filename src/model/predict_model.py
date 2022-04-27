import torch

from feature.build_dataloader import build_dataloader
from feature.build_features import build_features
from feature.dataset import CommaDataset


def predict():
    text = ['Показатели давления могут изменяться в зависимости от ряда факторов Даже у одного и того же '
            'пациента в течение суток наблюдаются колебания АД Например утром после пробуждения кровяное '
            'давление может быть низким после обеда оно может начать подниматься']
    input_ids, input_targets, target_mask, attention_mask = build_features(
        text=text,
        return_features=True)
    dataset = CommaDataset(input_ids, input_targets, target_mask, attention_mask)
    train_dataloader = build_dataloader(dataset, 1)
    model = torch.load('models/model.pth')

    with torch.no_grad():
        for batch in train_dataloader:
            x, y, y_mask, att_mask = batch['feature'], batch['target'], batch['target_mask'], batch['attention_mask']
            y_mask = y_mask.view(-1)
            y_predict = model(x, att_mask)

    y_predict = y_predict.view(-1, y_predict.shape[2])
    y_predict = torch.argmax(y_predict, dim=1).view(-1)

    print(y_predict)

    result = ""
    decode_idx = 0
    decode_map = {0: '', 1: ',', 2: '.'}
    words_original_case = ['SOS'] + text[0].split() + ['EOS']

    for i in range(y_mask.shape[0]):
        if y_mask[i] == 1:
            result += words_original_case[decode_idx]
            result += decode_map[y_predict[i].item()]
            result += ' '
            decode_idx += 1

    result = result.strip()
    print(result)
