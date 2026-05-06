Setup instructions

1) Create and activate a virtual environment managed by uv:

```powershell
uv venv --python 3.11
.venv\Scripts\activate
```

2) Install everything that can be handled automatically from `pyproject.toml`:

```powershell
uv sync --python 3.13
```

3) Prepare the dataset by running the following command:

```powershell
uv run src/preprocessing.py
```

4) Run train.py to train the model:

```powershell
uv run src/train.py
```
You can resume training from a checkpoint by providing the path to the checkpoint file:

```powershell
uv run src/train.py --resume path/to/checkpoint.pth
```

5) Run predict.py to make predictions with the trained model:

```powershell
uv run src/predict.py
```

