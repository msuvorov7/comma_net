import argparse
import logging
import os
import sys

import yaml
import onnxruntime
import numpy as np

from tokenizers import Tokenizer

from src.feature.dataset import IND_TO_TARGET_TOKEN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def loader(text: str, max_length: int, batch_size: int):
    words = text.split()
    data = []
    for i in range(0, len(words), max_length):
        data.append(' '.join(words[i: i + max_length]))
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args = args_parser.parse_args()

    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)
    
    onnx_model = onnxruntime.InferenceSession(os.path.join(config['models'], 'comma_model.onnx'))
    tokenizer = Tokenizer.from_file(os.path.join(config['models'], 'distilrubert_tokenizer.json'))
    tokenizer.enable_padding()

    text = ' '.join([
        'привет извини что голосовой просто мне так проще будет объяснить первое домашнее задание этому было реализовать крестики нолики',
        'с помощью обучения с подкреплением и библиотеки джим',
        'а втовое домашнее задание было доказать форму или расписать я здесь не светник потому что я не сделала и испугалась математики',
        'но возможно я смогу потом найти из клинф презентации',
        'а третье это было насем домашняя работа ты была вроде кактрольная которая можно было доделать дома там тоже какая то графовая модель',
        'и есть вероятности и ну кажется почитать вероятностиь какого то события я тоже здесь и советник потому что все пугае',
        'математике ну могу поискать может быть меня в диалогах',
        'с кем то это осталось задание вот и на последний паре что такого еще одну задачу она есть его презентации и она указыана как задача',
        'а вот две предыдущие я не уверен что указано поэтому и последнее можно найти на гитхаб в презентации',
    ])
    batch_size = 3
    max_words = 80

    result = []
    for batch in loader(text, max_words, batch_size):
        inputs = tokenizer.encode_batch(
            batch,
            add_special_tokens=True,
        )
        input_ids = np.asarray([_.ids for _ in inputs]).astype(np.int64).reshape(len(batch), -1)
        attention_mask = np.asarray([_.attention_mask for _ in inputs]).astype(np.int64).reshape(len(batch), -1)

        model_input = {
            onnx_model.get_inputs()[0].name: input_ids,
            onnx_model.get_inputs()[1].name: attention_mask,
        }
        model_output = onnx_model.run(None, model_input)[0].argmax(axis=-1).reshape(len(batch), -1)

        decoded_batch = []
        for i in range(len(batch)):
            sentence_tokens = []
            for tok, trg in zip(input_ids[i], model_output[i]):
                sentence_tokens.append(tok)
                if trg != 0:
                    sentence_tokens.append(IND_TO_TARGET_TOKEN[trg])
            decoded_batch.append(sentence_tokens)

        result += tokenizer.decode_batch(decoded_batch, skip_special_tokens=True)

    for s in ' '.join(result).split('.'):
        print(s.strip() + '.')
