# 開発への参加方法

バグ報告、実験案、文書の修正を歓迎します。実験コードやノートブックを変える場合は、変更したモデル、入力、実行条件、確認方法を Issue または Pull Request に書いてください。

## Issue を作成する

- ノートブックや CLI の不具合には[不具合報告フォーム](.github/ISSUE_TEMPLATE/bug_report.yml)を使ってください。
- 新しい評価例の提案には[実験案フォーム](.github/ISSUE_TEMPLATE/experiment_request.yml)を使ってください。
- Colab かローカル CLI か、使用したモデル、ノートブックのコミット、入力例、実際の出力を記載してください。
- ログや入力に個人情報、アクセストークン、非公開データを含めないでください。

## ドキュメントを確認する

Node.js と npm を用意し、リポジトリのルートで実行します。

```console
npm ci
npm run docs:dev
npm run docs:build
node tools/check_docs.mjs
npm run docs:preview
```

`docs:build` は日本語・英語ページを生成します。続けて `node tools/check_docs.mjs` を実行し、ビルド後のローカルリンク、見出しへのリンク、画像、言語別ページを確認します。

### ドキュメント依存関係を更新する

`package.json` では VitePress 1.6.4 を使い、推移依存のセキュリティ勧告に対応するため Vite を 6.4.4 に固定しています。依存関係を更新したら、次をすべて実行してください。

```console
npm audit --audit-level=high
npm run docs:build
node tools/check_docs.mjs
```

## Python とノートブックを確認する

Python コマンドは `uv` で実行します。テストとノートブック検査にはモデルの重みは不要です。配布用スモークチェックは wheel（Python 配布ファイル）をビルドし、一時環境へインストールします。合成キャッシュを使って `prepare` を実行し、画像・音声・動画の入力を生成します。モデル登録表と実験スクリプトのハッシュも確認対象です。実データセットやモデルの重みは取得しません。

```console
uv run --with pytest --with nbformat --with ipython --with pillow --with-editable . python -m pytest -q
uv run --with nbformat --with ipython --with pillow --with-editable . python tools/check_notebook.py
uv run --with-editable . python tools/check_public_payload.py
uv run --with-editable . python -m lfm3_lab --help
uv run --no-project python tools/check_distribution.py
```

テストやノートブック検査の成功は、モデルの回答が正しいことを示しません。モデルの実際の出力を報告する場合は、実行したノートブック、GPU、データ、モデルリビジョン、実行回数を明記してください。

## 実験結果を変更する

- `notebooks/LFM3_Colab_T4.ipynb` は Colab で試すクリーンな入力ノートブックです。
- `notebooks/results/T4_2026-10-28.ipynb` は、2026年10月28日の公開版実行結果を含む記録です。
- `results/public-notebook-run.json` はその1回実行の機械可読な記録です。
- `results/summary.json` と [`docs/results.md`](docs/results.md) は別に行った参考実験を示します。両方の結果を1つの試行数やスコアとして扱わないでください。

新しい実行記録を追加する場合は、既存の記録を上書きせず、実行条件と結果の単位を明記してください。失敗、未採点、実行基盤のエラーを区別して残してください。モデルの重みや秘密情報はコミットしないでください。

## Pull Request を送る

変更範囲、確認コマンド、既知の制限を Pull Request に記載してください。画面やモデル推論を確認していない場合は、その範囲を確認済みと書かないでください。テンプレートは[Pull Request の説明](.github/PULL_REQUEST_TEMPLATE.md)を参照してください。
