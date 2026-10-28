---
layout: home
title: LFM3 Colab Lab
titleTemplate: Liquid AI models on a Colab T4
hero:
  name: LFM3 Colab Lab
  text: Liquid AI の6モデルを Colab T4 で試す
  tagline: 実行用ノート、測定結果、公開ノートブックの確認記録をまとめています。
  image:
    src: /logo.svg
    alt: 6つの図形をつないだロゴ
  actions:
    - theme: brand
      text: Colab でノートブックを開く
      link: https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/LFM3_Colab_T4.ipynb
    - theme: alt
      text: 実験結果を見る
      link: /results
    - theme: alt
      text: 実行確認を読む
      link: /verification
features:
  - title: 6つのモデル
    details: テキスト、画像と動画、音声、複合入力、分類や項目抽出の固定例を試せます。
  - title: 実行条件を明記
    details: Tesla T4 と FP16 を使います。モデルの重みは実行時に Hugging Face から取得します。
  - title: 2種類の記録
    details: 3回測定の参考実験と、公開ノートブックの1回実行を別々に記録しています。
---

## まず Colab で試す

1. 「Colab でノートブックを開く」を選びます。
2. Colab のランタイム設定で Tesla T4 GPU を選びます。
3. すべてのセルを実行します。各モデルの出力はノートブックに表示されます。

ノートブックは T4 以外の GPU では推論を開始しません。1つのモデルだけ実行する場合は、共通準備セルを先に実行してください。Omni と d1-Omni には Audio セルの出力も必要です。

## 結果は測定の種類を分けて読む

2026年10月28日の参考実験は、ケースに応じて3回測定しました。公開ノートブックの実行確認は各ケース1回です。試した内容と採点方法も異なるため、成功数はモデルの順位を示しません。

| 記録 | 用途 | 詳細 |
| --- | --- | --- |
| 参考実験 | 短い固定入力の応答、時間、PyTorch メモリを測定 | [実験結果と限界](/results) |
| 公開ノートブックの実行確認 | GitHub の公開版を Colab T4 で取得して実行 | [実行確認と失敗例](/verification) |

元の記録は [`results/summary.json`](https://github.com/Sunwood-ai-labs/lfm3-colab-lab/blob/main/results/summary.json) と [`results/public-notebook-run.json`](https://github.com/Sunwood-ai-labs/lfm3-colab-lab/blob/main/results/public-notebook-run.json) にあります。

英語のページは右上の言語メニューから切り替えられます。
