# Raw dataset

Положите сюда файлы из облачной папки организаторов:

- `train.json` — 13 000 размеченных текстов
- `val.json` — 1 500 размеченных текстов

После загрузки зафиксируйте данные в DVC:

```bash
uv run dvc add data/raw/train.json data/raw/val.json
git add data/raw/*.dvc .gitignore
git commit -m "track raw dataset with DVC"
```

Для remote-хранилища (S3/GDrive/local path):

```bash
uv run dvc remote add -d storage /path/to/dvc/storage   # или s3://bucket/path
uv run dvc push
```
