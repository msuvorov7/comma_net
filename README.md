# Comma Net

## Нейросеть для расстановки знаков препинания

Модель на основе `DeepPavlov/distilrubert-tiny-cased-conversational-v1`.

[Статья](https://habr.com/ru/company/barsgroup/blog/563854/), с которой все началось.
[Репозиторий](https://github.com/sviperm/neuro-comma) ребят.
Постарался упростить их реализацию.

Обучение происходило на датасете новостей Ленты.

Загрузка датасета:
```python
python -m src.data_load.data_load --test_size=0.35
```

Для запуска пайплайна обучения:
```python
python -m src.model.train_model --epoch=2
```

Для запуска с обученной моделью: 
```python
python -m src.model.predict
```
