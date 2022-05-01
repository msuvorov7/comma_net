import re

import torch

from feature.build_dataloader import build_dataloader
from feature.build_features import build_features, cut_text
from feature.dataset import CommaDataset


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
        text = file.readlines()

    # print(text)
    text = cut_text(100, text=text, is_train=False, return_text=True)

    input_ids, input_targets, target_mask, attention_mask = build_features(
        text=text,
        return_features=True,
        is_train=False,
    )
    # print(len(input_ids))
    # print(input_ids, attention_mask)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = CommaDataset(input_ids, input_targets, target_mask, attention_mask)
    test_dataloader = build_dataloader(dataset, batch_size=batch_size)

    model = torch.load('models/model.pth')
    model.to(device)

    # print(text)
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

            string = ' '.join(list(
                map(
                    lambda sent: 'SOS ' + sent + ' EOS',
                    text[i * batch_size:i * batch_size + batch_size]
                )
            ))
            words_original_case = list(filter(lambda word: word != ' ', re.split('(\W)', string)))
            result += decode(words_original_case, y_predict, y_mask) + ' '
    print(result)
