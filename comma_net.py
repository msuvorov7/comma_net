#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


import torch
import numpy as np


from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences


from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
pretrained_transformer = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')


def preprocessing(text):
  tokenized_texts = [tokenizer.tokenize(sent) for sent in text]
  tokenized_texts = [['[SOS]'] + sentense + ['[EOS]'] for sentense in tokenized_texts]
  
  TOKENS = []
  Y = []
  for i in tokenized_texts:
      token = []
      y = []
      y_mask = []
      # для простоты оставили только два класса
      for word in i:
          if word == ',':
              y = y[:-1]
              y.append(1)
          elif word == '.':
              y = y[:-1]
              y.append(2)
          else:
              token.append(word)
              y.append(0)
      TOKENS.append(token)
      Y.append(y)
  print(Y[0])
  print(TOKENS[0])

  Y_MASK = []
  for i in text:
      y_mask = [1]
      for word in i.replace('—', '').replace(',', '').replace('.', '').split():
        # print(tokenizer.tokenize(word))
        word_pieces = tokenizer.tokenize(word)
        if len(word_pieces) == 1:
            y_mask.append(1)
        else:
            y_mask += [0 for _ in range(len(word_pieces)-1)]
            y_mask.append(1)
      y_mask.append(1)
      Y_MASK.append(y_mask)
  print(Y_MASK[0])

  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in TOKENS]
  input_ids = pad_sequences(
      input_ids,
      maxlen=256,
      dtype="long",
      truncating="post",
      padding="post"
  )    
  print('input_ids done')
  Y_IDS = pad_sequences(
      Y,
      maxlen=256,
      dtype="long",
      truncating="post",
      padding="post"
  )
  print('y_ids done')
  Y_MASK_IDS = pad_sequences(
      Y_MASK,
      maxlen=256,
      dtype="long",
      truncating="post",
      padding="post"
  )
  print('y_mask_ids done')

  attention_masks = [[float(i>0) for i in seq] for seq in input_ids]

  return input_ids, Y_IDS, Y_MASK_IDS, attention_masks



DEVICE = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
DEVICE


class CommaModel(nn.Module):
    def __init__(self) -> None:

        super(CommaModel, self).__init__()
        bert_dim = 768
        hidden_size = bert_dim

        self.hidden_size = hidden_size
        self.pretrained_transformer = pretrained_transformer
        self.lstm = nn.LSTM(input_size=bert_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=True)

        self.linear = nn.Linear(in_features=hidden_size * 2,
                                out_features=3) ######

    def forward(self, x: torch.tensor, attn_masks: torch.tensor) -> torch.tensor:
        # add dummy batch for single sample
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])
        # (B, N, E) -> (B, N, E)
        x = self.pretrained_transformer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x


a = torch.load('model_base', map_location=torch.device('cpu'))


txt = st.text_area('Text to analyze', 'Показатели давления могут изменяться в зависимости от ряда факторов Даже\
у одного и того же пациента в течение суток наблюдаются колебания АД Например утром после пробуждения кровяное\
давление может быть низким после обеда оно может начать подниматься',
    height=300)


def run_sentiment_analysis(text):
    input_ids, Y_IDS, Y_MASK_IDS, attention_masks = preprocessing(text)

    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(Y_IDS)
    train_y_mask = torch.tensor(Y_MASK_IDS)
    train_masks = torch.tensor(attention_masks)

    x = train_inputs.to(DEVICE)

    attn_mask = train_masks.to(DEVICE)

    y_mask = train_y_mask.view(-1)

    with torch.no_grad():
        y_predict = a(x, attn_mask)

    y_predict = y_predict.view(-1, y_predict.shape[2])
    y_predict = torch.argmax(y_predict, dim=1).view(-1)


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
    return result[4:-4]

st.write('Result:', run_sentiment_analysis([txt]))
