import os
import pickle
import sys
import unittest

from torch.utils.data import DataLoader

from feature.dataset import CommaDataset

# sys.path.insert(0, os.path.dirname(
#     os.path.dirname(os.path.realpath(__file__))
# ))

from model.train_model import collate_fn


class TestDataset(unittest.TestCase):
    def test_dataset_shape(self):
        # build_features()

        with open('../data/processed/input_ids.pkl', 'rb') as f:
            input_ids = pickle.load(f)
        with open('../data/processed/input_targets.pkl', 'rb') as f:
            input_targets = pickle.load(f)
        with open('../data/processed/target_mask.pkl', 'rb') as f:
            target_mask = pickle.load(f)
        with open('../data/processed/attention_mask.pkl', 'rb') as f:
            attention_mask = pickle.load(f)

        print('data_load loaded')
        dataset = CommaDataset(input_ids, input_targets, target_mask, attention_mask)

        for i, batch in enumerate(dataset):
            x, y, y_mask, att_mask = batch['feature'], batch['target'], batch['target_mask'], batch['attention_mask']
            # print(x.shape)
            # print(y.shape)
            # print(y_mask.shape)
            # print(att_mask.shape)
            # print(i)
            self.assertTrue(x.shape == y.shape == y_mask.shape == att_mask.shape)

    def test_batch_shape(self):

        with open('../data/processed/input_ids.pkl', 'rb') as f:
            input_ids = pickle.load(f)
        with open('../data/processed/input_targets.pkl', 'rb') as f:
            input_targets = pickle.load(f)
        with open('../data/processed/target_mask.pkl', 'rb') as f:
            target_mask = pickle.load(f)
        with open('../data/processed/attention_mask.pkl', 'rb') as f:
            attention_mask = pickle.load(f)

        print('data_load loaded')
        dataset = CommaDataset(input_ids, input_targets, target_mask, attention_mask)
        train_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
        for i, batch in enumerate(train_dataloader):
            x, y, y_mask, att_mask = batch['feature'], batch['target'], batch['target_mask'], batch['attention_mask']
            # print(x.shape)
            # print(y.shape)
            # print(y_mask.shape)
            # print(att_mask.shape)
            # print(i)
            self.assertTrue(x.shape == y.shape == y_mask.shape == att_mask.shape)


if __name__ == '__main__':
    unittest.main()
