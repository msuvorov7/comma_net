import os
import re
import sys
import yaml
import torch
import logging
import argparse

import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split

from src.model.model import CommaModel, FocalLoss, ModelLightning
from src.feature.dataset import CommaV2, collate_fn, TARGETS, TOKENIZER
from src.visualization.visualize import plot_conf_matrix


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def save_tokenizer(tokenizer, directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    tokenizer_path = os.path.join(directory_path, 'distilrubert_tokenizer.json')    
    tokenizer.backend_tokenizer.save(tokenizer_path)
    logging.info(f'tokenizer saved: {tokenizer_path}')


def save_state_dict(model: nn.Module, directory_path: str) -> None:
    """
    функция для сохранения состояния модели
    :param model: модель
    :param directory_path: имя директории
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    model_path = os.path.join(directory_path, 'comma_model.sd')
    torch.save(model.state_dict(), model_path)
    logging.info(f'state_dict saved: {model_path}')


def export_to_onnx(model: nn.Module, directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    model_path = os.path.join(directory_path, 'comma_model.onnx')
    
    dummy_input = (torch.randint(0, TOKENIZER.vocab_size, size=(1, 128)), torch.ones(1, 128, dtype=torch.int64))
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=['x', 'attn_mask'],
        output_names=['output'],
        dynamic_axes={'x': {0: 'batch_size', 1: 'seq_len'}, 'attn_mask': {0: 'batch_size', 1: 'seq_len'}},
    )

    logging.info(f'exported to onnx: {model_path}')


def normalize(text: pd.Series) -> pd.Series:
    normalize_pattern = re.compile(r'[^а-яa-z0-9\s\,\.\-\?\!]')
    return (
        text
        .dropna()
        .apply(lambda item: item.lower().replace('—', '-'))
        .apply(lambda item: re.sub(normalize_pattern, '', item))
    )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args_parser.add_argument('--epochs', default=1, type=int, dest='epochs')
    args_parser.add_argument('--batch_size', default=64, dest='batch_size', type=int)
    args_parser.add_argument('--accumulate_grad_batches', default=1, dest='accumulate_grad_batches', type=int)
    args = args_parser.parse_args()

    assert args.epochs > 0

    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    # data load
    train_df = pd.read_parquet('data/raw/test.parquet', columns=['text'])
    test_df = pd.read_parquet('data/raw/test.parquet', columns=['text'])
    
    train_text = normalize(train_df['text']).values.tolist()[:]
    test_text = normalize(train_df['text']).values.tolist()[:]

    train_dataset = CommaV2(
        data=train_text,
        tokenizer=TOKENIZER,
        max_length=128,
        mode='train',
    )
    test_dataset = CommaV2(
        data=test_text,
        tokenizer=TOKENIZER,
        max_length=128,
        mode='test',
    )
    train_size = len(train_dataset)
    validation_size = int(0.3 * train_size)

    train_dataset, valid_dataset = random_split(
        train_dataset, [train_size - validation_size, validation_size],
        generator=torch.Generator().manual_seed(42)
    )
    valid_dataset.mode = 'valid'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    logging.info(f'datasets loaded')

    # init model
    comma_net = CommaModel(num_class=len(TARGETS) + 1)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.35, 0.35, 0.1, 0.1]))
    criterion = FocalLoss(alpha=4, gamma=2)
    model = ModelLightning(
        model=comma_net,
        criterion=criterion,
        t_max=int(len(train_dataset) / (args.batch_size * args.accumulate_grad_batches) + 1) * args.epochs,
        lr=1e-3,
    )
    logging.info(f'model created')

    # train
    lr_callback =pl.callbacks.LearningRateMonitor(
        logging_interval='step',
    )
    trainer = pl.Trainer(
        accelerator='mps',
        # logger=logger,
        callbacks=[lr_callback, ],
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision="bf16-mixed",
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        gradient_clip_val=5,
        use_distributed_sampler=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    trainer.test(model, dataloaders=test_loader)

    cm = model.conf_matrix.compute()
    plot_conf_matrix(cm, ['none',] + list(TARGETS))

    save_tokenizer(TOKENIZER, config['models'])
    save_state_dict(model.model, config['models'])
    export_to_onnx(model.model, config['models'])
