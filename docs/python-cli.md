# Python CLI で固定テストを実行する

このプロジェクトの主な使い方は Colab ノートブックです。CLI は Tesla T4 環境から固定テストを個別に実行するときに使います。Python 3.10 以降と、T4 に対応する Torch と CUDA が必要です。CLI は CPU や別の GPU に切り替えません。

## T4 用の環境を確認する

T4 対応の Torch と CUDA が入った Python 環境を有効にしてから、次を実行します。2行目に Tesla T4 が表示されることを確認してください。Torch と CUDA 関連パッケージは `requirements-colab.txt` に含まれません。

```console
uv run --active --no-sync -- python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
uv pip install --requirement requirements-colab.txt
uv pip install --no-deps --editable .
```

Colab ノートブックを使う場合、CLI 用のパッケージ設定は不要です。

## 固定入力を作る

このコマンドは画像、音声、動画の入力を `./runs/inputs` に作ります。音声には Google FLEURS の日本語サンプルを使います。データの出典とライセンスは[NOTICE](https://github.com/Sunwood-ai-labs/lfm3-colab-lab/blob/main/NOTICE.md)を確認してください。

```console
uv run --active --no-sync -- python -m lfm3_lab prepare --output-dir ./runs --cache-dir ./cache
```

## モデルを実行する

`run` は一度に1つのモデルを実行します。入力を用意した後、次のキーから選んでください。

| モデル | キー |
| --- | --- |
| LFM3-1.7B | `text` |
| LFM3-VL-2B | `vision` |
| LFM3-Audio-2B | `audio` |
| LFM3-Omni-2B | `omni` |
| d1-2B | `d1` |
| d1-Omni-2B | `d1_omni` |

次の例は LFM3-1.7B を既定の3回で実行します。

```console
uv run --active --no-sync -- python -m lfm3_lab run --model text --output-dir ./runs
```

回数を変えるには `--repeats` を指定します。回答、測定結果、ログは `./runs` に保存されます。

Omni と d1-Omni の固定テストは Audio が作った音声も使います。どちらかを単独実行する場合は、先に Audio テストを実行してください。

## 結果を表示して保存する

全モデルの集計を表示するには `summarize` を実行します。`export` は結果と実行ログを ZIP にまとめます。

```console
uv run --active --no-sync -- python -m lfm3_lab summarize --output-dir ./runs
uv run --active --no-sync -- python -m lfm3_lab export --output-dir ./runs
```

ZIP は実行環境で作る成果物です。リポジトリには含めません。

## Python から呼び出す

次の例は、パッケージをインストールした環境の Jupyter または Colab で実行します。

```python
from lfm3_lab import Lab

lab = Lab(output_dir="runs", cache_dir="cache")
lab.prepare()
result = lab.run("text", repeats=3)
lab.show(result)
```

自由入力には `run_text` を使います。

```python
answer = lab.run_text("18 と 24 を足してください。", repeats=1)
print(answer.answer)
```

各モデルの成功数は小規模な固定ケースに対する値です。モデル全般の性能を表しません。
