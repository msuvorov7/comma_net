import argparse
import logging
import os
import bz2
import sys

import numpy as np
import pandas as pd
import yaml
from six.moves import urllib
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def download_dataset(directory_path: str) -> None:
    """
    Загрузка датасета новостей Ленты
    :param directory_path: путь до raw директории
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    url = 'https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2'
    logging.info(f'downloading url: {url}')

    data = urllib.request.urlopen(url)
    file_path = os.path.join(directory_path, os.path.basename(url))
    with open(file_path, 'wb') as f:
        f.write(data.read())

    logging.info('Extracting data_load')
    with open(file_path, 'rb') as source, open(directory_path + 'lenta.csv', 'wb') as dest:
        dest.write(bz2.decompress(source.read()))

    logging.info(f'file saved in {directory_path}')

    os.remove(file_path)


def train_test_split_dataset(directory_path: str, test_size: float):
    """
    Разделение датасета на train и test части по фиксированному random_state
    :param directory_path: путь до raw директории
    :param test_size: размер тестовой части
    :return:
    """
    df = pd.read_csv(directory_path + 'lenta.csv')

    train, test, _, _ = train_test_split(df,
                                         np.zeros(len(df)),
                                         test_size=test_size,
                                         random_state=42
                                         )
    train.to_csv(directory_path + 'train.csv', index=False)
    test.to_csv(directory_path + 'test.csv', index=False)

    logging.info(f'train/test splits created')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args_parser.add_argument('--test_size', default=0.35, dest='test_size', type=float)
    args = args_parser.parse_args()

    assert args.test_size > 0

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_raw_dir = fileDir + config['data']['raw']
    download_dataset(data_raw_dir)
    train_test_split_dataset(data_raw_dir, args.test_size)
