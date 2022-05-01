import pickle
from feature.build_dataloader import build_dataloader
from feature.dataset import CommaDataset
from model.predict_model import predict
from model.train_model import fit
from src.data.data_load import download_lenta
from src.feature.build_features import *


if __name__ == '__main__':
    download_lenta()
    extract_sample(sample_size=1_000)
    cut_text(100)
    build_features(return_features=False, is_train=True)

    with open('data/processed/input_ids.pkl', 'rb') as f:
        input_ids = pickle.load(f)
    with open('data/processed/input_targets.pkl', 'rb') as f:
        input_targets = pickle.load(f)
    with open('data/processed/target_mask.pkl', 'rb') as f:
        target_mask = pickle.load(f)
    with open('data/processed/attention_mask.pkl', 'rb') as f:
        attention_mask = pickle.load(f)

    print('data loaded')
    dataset = CommaDataset(input_ids, input_targets, target_mask, attention_mask)
    # print(dataset[0])
    # train_dataloader = build_dataloader(dataset, 32)

    print(len(dataset))

    # for batch in train_dataloader:
    #     x, y, y_mask, att_mask = batch['feature'], batch['target'], batch['target_mask'], batch['attention_mask']
    #     print(x)
    #     print(y)
    #     print(y_mask)
    #     print(att_mask)
    #     break

    fit(dataset, 4)
    predict()
