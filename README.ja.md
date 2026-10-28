<div align="center">
  <img src="assets/banner.svg" alt="LFM3 Colab Lab の図形バナー" width="100%">
  <h1>LFM3 Colab Lab</h1>
  <p>Liquid AI の6モデルを、Google Colab の Tesla T4 で試すノートブックと実験記録です。</p>
  <p><a href="README.md">English</a> · <strong>日本語</strong></p>
</div>

<p align="center">
  <a href="https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/LFM3_Colab_T4.ipynb"><img alt="Open in Google Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  <a href="https://github.com/Sunwood-ai-labs/lfm3-colab-lab/actions/workflows/quality.yml"><img alt="Notebook and source quality" src="https://github.com/Sunwood-ai-labs/lfm3-colab-lab/actions/workflows/quality.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

## Colab で試す

1. [公開ノートブックを Colab で開く](https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/LFM3_Colab_T4.ipynb)。
2. ランタイムの種類で Tesla T4 GPU を選びます。
3. 共通準備セルのあと、試したいモデルのセルを実行します。全モデルを試す場合は「すべてのセルを実行」します。
4. 結果は各モデルのセルの下に表示されます。

このリンクは出力を含まない公開ノートブックを開きます。実行を始めるには、Colab でセルを実行してください。T4 を使えないセッションではノートブックは推論を開始しません。初回はモデルを Hugging Face からダウンロードします。

同じ入力を繰り返して測る場合は、ノートブック先頭の `REPEATS` を `3` にします。公開版の記録は各ケース1回の実行です。Omni と d1-Omni は Audio セルの出力も使うため、Audio を先に実行します。

## 6モデルで試せること

| モデル | 試せる内容 | 実験コード |
| --- | --- | --- |
| LFM3-1.7B | 日本語の言い換え、テキストからのツール呼び出し | [`run_text_prompt.py`](experiments/run_text_prompt.py)、[`models.py`](src/lfm3_lab/models.py) |
| LFM3-VL-2B | 画像や動画についての質問、画像からのツール呼び出し | [`run_video_tools.py`](experiments/run_video_tools.py) |
| LFM3-Audio-2B | 音声認識、音声生成、音声対話、音声ツール | [`run_audio.py`](experiments/run_audio.py) |
| LFM3-Omni-2B | 音声と画像、または音声と動画についての質問 | [`run_audio.py`](experiments/run_audio.py) |
| d1-2B | テキストや画像の分類、項目抽出、位置と個数の判定 | [`run_decisions.py`](experiments/run_decisions.py) |
| d1-Omni-2B | テキスト、画像、音声を使った判定 | [`run_decisions.py`](experiments/run_decisions.py) |

モデル名、Hugging Face の取得先、固定リビジョンは [`models.py`](src/lfm3_lab/models.py) にあります。重みはこのリポジトリに含めず、実行時に取得します。

## 結果を読む

2026年10月28日に、GitHub で公開したノートブックを Google Colab Python CLI から Tesla T4・FP16 で実行した記録があります。実行基盤のエラーはなく、6モデルすべての結果を取得しました。下表は各ケース1回の実行で確認したタスク別の結果です。

| モデル | 通過した検査 / 記録数 | 残った結果 |
| --- | ---: | --- |
| LFM3-1.7B | 2 / 2 | 機能上の失敗なし |
| LFM3-VL-2B | 3 / 3 | 機能上の失敗なし |
| LFM3-Audio-2B | 9 / 15 | 4件は未採点、2件は機能上の失敗 |
| LFM3-Omni-2B | 1 / 3 | 2件は機能上の失敗 |
| d1-2B | 2 / 2 タスク群 | 機能上の失敗なし |
| d1-Omni-2B | 3 / 4 タスク群 | 1件は機能上の失敗 |

この表は小規模な異なるタスクの記録で、モデル間の順位ではありません。Audio の通過には波形の生成だけを確かめた項目も含みます。未採点の項目と失敗例は[公開ノートブックの実行確認](docs/verification.md)を参照してください。

- [保存済みの1回実行ノートブック](https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/results/T4_2026-10-28.ipynb)
- [公開ノートブックの機械可読な実行記録](results/public-notebook-run.json)
- [以前の反復測定と失敗例](docs/results.md)
- [参考実験の機械可読な要約](results/summary.json)

別の参考実験では、LFM3-Audio-2B の日本語音声対話と音声ツールはそれぞれ0/3でした。LFM3-Omni-2B は青い図形への質問に3回とも応答せず、d1-Omni-2B は音声で「青い図形の7」を尋ねると、3回とも「3」と答えました。これらは公開ノートブックの1回実行とは別の結果です。[参考実験の結果と条件](docs/results.md)を参照してください。

`results/summary.json` は参考実験の機械可読な要約です。主なケースは3回測定し、一部の音声ケースは1回測定しました。`evidence_status` は参考実験の時点で公開ノートブックをまだ実行していなかったことを示します。後日実行した公開ノートブックは `results/public-notebook-run.json` に別の記録があります。

## 開発する

ドキュメントのプレビューとビルドはリポジトリのルートで実行します。

```console
npm ci
npm run docs:dev
npm run docs:build
node tools/check_docs.mjs
npm run docs:preview
```

配布用スモークチェックは `uv run --no-project python tools/check_distribution.py` で実行できます。Python CLI、ノートブックの確認方法、テストコマンドは[開発への参加方法](CONTRIBUTING.md)を参照してください。ノートブックを Colab で使う場合、ローカルの Python CLI は必要ありません。

## 詳細

- [日本語ドキュメント](https://sunwood-ai-labs.github.io/lfm3-colab-lab/)
- [英語ドキュメント](https://sunwood-ai-labs.github.io/lfm3-colab-lab/en/)
- [Python CLI の使い方](docs/python-cli.md)
- [モデルとデータのライセンス・出典](NOTICE.md)
- [変更履歴](CHANGELOG.md)
- [開発への参加方法](CONTRIBUTING.md)
