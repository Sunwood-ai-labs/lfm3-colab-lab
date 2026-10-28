<div align="center">
  <img src="assets/banner.svg" alt="Geometric banner for LFM3 Colab Lab" width="100%">
  <h1>LFM3 Colab Lab</h1>
  <p>Try six Liquid AI models in a Google Colab Tesla T4 notebook and inspect the recorded experiments.</p>
  <p><strong>English</strong> · <a href="README.ja.md">日本語</a></p>
</div>

<p align="center">
  <a href="https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/LFM3_Colab_T4.ipynb"><img alt="Open in Google Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  <a href="https://github.com/Sunwood-ai-labs/lfm3-colab-lab/actions/workflows/quality.yml"><img alt="Notebook and source quality" src="https://github.com/Sunwood-ai-labs/lfm3-colab-lab/actions/workflows/quality.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

## Run it in Colab

1. [Open the public notebook in Colab](https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/LFM3_Colab_T4.ipynb).
2. Select a Tesla T4 GPU in the runtime settings.
3. Run the shared setup cell, then run the model cell you want. Choose **Runtime > Run all** to try every model.
4. Each result appears below its model cell.

The link opens the public notebook without saved outputs. Run its cells in Colab to start inference. The notebook stops before inference when the session has no T4. On the first run, it downloads model weights from Hugging Face.

Set `REPEATS` to `3` in the first notebook cell to repeat each case. The published execution record uses one run per case. The Omni and d1-Omni suites also use audio from the Audio cell, so run Audio first.

## What the six models do

| Model | Try it for | Experiment code |
| --- | --- | --- |
| LFM3-1.7B | Japanese text rewriting and text tool calls | [`run_text_prompt.py`](experiments/run_text_prompt.py), [`models.py`](src/lfm3_lab/models.py) |
| LFM3-VL-2B | Image and video questions, image tool calls | [`run_video_tools.py`](experiments/run_video_tools.py) |
| LFM3-Audio-2B | Speech recognition, speech generation, voice chat, audio tools | [`run_audio.py`](experiments/run_audio.py) |
| LFM3-Omni-2B | Questions about audio with images or video | [`run_audio.py`](experiments/run_audio.py) |
| d1-2B | Text and image classification, extraction, locations, and counts | [`run_decisions.py`](experiments/run_decisions.py) |
| d1-Omni-2B | Decisions from text, images, and audio | [`run_decisions.py`](experiments/run_decisions.py) |

The model names, Hugging Face IDs, and pinned revisions live in [`models.py`](src/lfm3_lab/models.py). This repository does not include model weights. The notebook downloads them when you run it.

## Read the results

On 2026-10-28, the public GitHub notebook was fetched and run through the Google Colab Python CLI on a Tesla T4 with FP16. The record covers one run per case across all six models.

| Model | Checks passed / records | Remaining results |
| --- | ---: | --- |
| LFM3-1.7B | 2 / 2 | No functional failures |
| LFM3-VL-2B | 3 / 3 | No functional failures |
| LFM3-Audio-2B | 9 / 15 | 4 ungraded records, 2 functional failures |
| LFM3-Omni-2B | 1 / 3 | 2 functional failures |
| d1-2B | 2 / 2 task groups | No functional failures |
| d1-Omni-2B | 3 / 4 task groups | 1 functional failure |

These are small, different task sets. Their counts do not rank the models. Some Audio checks only confirm that a waveform was generated. See the [public notebook execution record](docs/verification.md) for ungraded cases and failures.

- [Saved notebook from the one-run execution](https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/results/T4_2026-10-28.ipynb)
- [Machine-readable public notebook record](results/public-notebook-run.json)
- [Separate historical repeated measurements and failures](docs/results.md)
- [Machine-readable historical summary](results/summary.json)

The separate reference experiment repeated most cases three times. Some audio cases ran once. The Audio Japanese voice chat and voice-tool results were 0/3 each. Omni returned no answer to the blue-shape question in all three runs. d1-Omni answered 3 when the spoken question asked for the blue 7. See [historical results and conditions](docs/results.md).

`results/summary.json` is the machine-readable reference record. Its `evidence_status` says that the public GitHub notebook had not yet been run at that earlier point. The later public notebook execution has a separate record in `results/public-notebook-run.json`.

## Develop

Run the documentation preview and build from the repository root.

```console
npm ci
npm run docs:dev
npm run docs:build
node tools/check_docs.mjs
npm run docs:preview
```

See [Contributing](CONTRIBUTING.md) for the Python CLI, notebook checks, and test commands. You do not need the local Python CLI to run the notebook in Colab.

## More information

- [English documentation](https://sunwood-ai-labs.github.io/lfm3-colab-lab/en/)
- [日本語ドキュメント](https://sunwood-ai-labs.github.io/lfm3-colab-lab/)
- [Python CLI guide](docs/python-cli.md)
- [Model and data licenses and sources](NOTICE.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
