# Второе ДЗ

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelA/made-adv-ml-hw2/blob/master/solution.ipynb)

[Описание задачи](/task.pdf)

[Описание API](http://api.rating.chgk.net/tournaments)

[Рейтинг игроков](https://rating.chgk.info/players.php)


## Требования для запуска

1. Python 3.8 или выше
2. Настроенная Anaconda
3. Установленный [Jupyter Lab или Notebook](https://jupyter.org/)
4. Git LFS (исходные данных хранятся в LFS)
5. Опционально CUDA 11.0 или выше (для обучения EM алгоритма на GPU).

## Как запустить

Можно запустить пример сразу в Google Colab.

### Локальная настройка

Создать новое окружение для Anaconda:
```
conda env create -n env_name --file ./environment.yml
conda activate env_name
pip install -r ./requirements.txt
```
Если исходных данных нет в директории `data`, то выполнить команду:
```
git lfs pull
```

Выполнить предподготовку данных:
```
dvc repro train_test_split
```

Запустить `jupyter lab` или `notebook`. Открыть `solution.ipynb`.
