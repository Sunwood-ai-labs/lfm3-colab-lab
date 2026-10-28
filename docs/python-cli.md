# Python CLI で実行する

Colab ノートブックがこのプロジェクトの主な使い方です。CLI は T4 環境から固定テストを個別に実行したい場合に使います。Python 3.10 以降と NVIDIA Tesla T4 が必要です。CLI は CPU や別の GPU へ自動で切り替えません。

## 環境を用意する

Torch と CUDA が T4 に対応する Python 環境を用意し、リポジトリのルートで次を実行します。

```console
python -m pip install --requirement requirements-colab.txt
python -m pip install --no-deps --editable .
```

`requirements-colab.txt` は Transformers などの実験用パッケージを入れます。Torch、TorchVision、TorchAudio、Triton、NVIDIA CUDA パッケージは含みません。T4 に合う Torch と CUDA の組み合わせを先に用意してください。

Colab ノートブックを実行する場合、これらのインストール手順は不要です。ノートブックが依存関係を確認してからパッケージを設定します。

## 固定入力を用意する

次のコマンドで実験用の画像、音声、動画を `./runs/inputs` に作成します。入力には Google FLEURS の日本語音声サンプルを使います。データの出典とライセンスは[NOTICE](../NOTICE.md)を確認してください。

```console
python -m lfm3_lab prepare --output-dir ./runs --cache-dir ./cache
```

## モデルを実行する

`run` は1回に1つのモデルを実行します。入力準備後に、試したいモデルのキーを指定してください。

| モデル | キー |
| --- | --- |
| LFM3-1.7B | `text` |
| LFM3-VL-2B | `vision` |
| LFM3-Audio-2B | `audio` |
| LFM3-Omni-2B | `omni` |
| d1-2B | `d1` |
| d1-Omni-2B | `d1_omni` |

たとえば、LFM3-1.7B を3回測るには次を実行します。既定の繰り返し数も3回です。

```console
python -m lfm3_lab run --model text --output-dir ./runs
```

別の回数を使うには `--repeats` を指定します。実験結果、ケースの回答、実行ログは `./runs` に保存されます。

## 結果を表示して保存する

全モデルの集計を表示するには `summarize` を実行します。`export` は保存した結果と実行ログを ZIP にまとめます。

```console
python -m lfm3_lab summarize --output-dir ./runs
python -m lfm3_lab export --output-dir ./runs
```

ZIP は実行環境で生成するローカル成果物です。リポジトリには含めません。

## Python から呼び出す

次の表示例は、IPythonが入っているJupyterまたはColab用です。通常のターミナルでは、上のCLIコマンドを使ってください。

```python
from lfm3_lab import Lab

lab = Lab(output_dir="runs", cache_dir="cache")
lab.prepare()
result = lab.run("text", repeats=3)
lab.show(result)
```

自由入力の文章を試すには `run_text` を使います。

```python
answer = lab.run_text("18 と 24 を足してください。", repeats=1)
print(answer.answer)
```

Omni と d1-Omni の固定テストは Audio が作る音声も使います。個別に実行する場合は、先に Audio のテストを実行してください。成功数は小規模な固定ケースに対する値で、モデル全般の性能を表すものではありません。


