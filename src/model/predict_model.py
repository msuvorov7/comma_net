import os
import re
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.feature.build_feature import build_features, ContentWrapper
from src.feature.dataset import CommaDataset
from src.model.train_model import collate_fn


fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')


decode_map = {0: '', 1: ',', 2: '.'}


def decode(words_original_case: list, y_predict: torch.Tensor, y_mask: torch.Tensor) -> str:
    result = ""
    decode_idx = 0

    for index in range(y_mask.shape[0]):
        if y_mask[index] == 1:
            if words_original_case[decode_idx] not in ('SOS', 'EOS'):
                result += words_original_case[decode_idx]
                result += decode_map[y_predict[index].item()]
                result += ' '
            decode_idx += 1

    result = result.strip()
    return result


def predict(path_to_text: str = 'predict_test.txt', batch_size: int = 2):
    with open(path_to_text) as file:
        text = file.read()

    content = ContentWrapper(max_size=150).fit(text)
    reshaped_text = content.get_split()

    input_ids, input_targets, target_mask, attention_mask = build_features(text)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = CommaDataset(input_ids, input_targets, target_mask, attention_mask)
    test_dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    # model = torch.load('models/model.pth')
    model = torch.load('models/model.torch')
    model.to(device)

    result = ""
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            x, y, y_mask, att_mask = batch['feature'], batch['target'], batch['target_mask'], batch['attention_mask']
            y_mask = y_mask.view(-1)
            x = x.to(device)
            att_mask = att_mask.to(device)
            y_predict = model(x, att_mask)

            y_predict = y_predict.view(-1, y_predict.shape[2])
            y_predict = torch.argmax(y_predict, dim=1).view(-1)

            print(y_predict)

            string = ' '.join(list(
                map(
                    lambda sent: 'SOS ' + sent + ' EOS',
                    reshaped_text[i * batch_size:i * batch_size + batch_size]
                )
            ))
            words_original_case = list(filter(lambda word: word != ' ', re.split('(\W)', string)))
            print(words_original_case)
            result += decode(words_original_case, y_predict, y_mask) + ' '
    print(result)
