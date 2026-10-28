---
layout: home
title: LFM3 Colab Lab
titleTemplate: Liquid AI models on a Colab T4
hero:
  name: LFM3 Colab Lab
  text: Try six Liquid AI models on a Colab T4
  tagline: Colab notebook, measured results, and a record of the public notebook run.
  image:
    src: /logo.svg
    alt: Six geometric tiles connected in a grid
  actions:
    - theme: brand
      text: Open the Colab notebook
      link: https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/LFM3_Colab_T4.ipynb
    - theme: alt
      text: Read the results
      link: /en/results
    - theme: alt
      text: Read the run validation
      link: /en/verification
features:
  - title: Six models
    details: Try fixed examples for text, image and video, audio, mixed input, classification, and extraction.
  - title: Clear run conditions
    details: The notebook uses a Tesla T4 and FP16. It downloads model weights from Hugging Face at runtime.
  - title: Two separate records
    details: The site separates the repeated reference experiment from the one-repetition public notebook run.
---

## Start in Colab

1. Select **Open the Colab notebook**.
2. Choose a Tesla T4 GPU in the Colab runtime settings.
3. Run all cells. The notebook displays each model's output below its cell.

The notebook stops before inference on GPUs other than a T4. To run one model, run the shared setup cell first. Omni and d1-Omni also need the output from the Audio cell.

## Read each result with its run type

The reference experiment on 2026-10-28 measured each applicable case three times. The public notebook validation used one repetition per case. The tested tasks and scoring rules also differ, so pass counts do not rank the models.

| Record | Purpose | Details |
| --- | --- | --- |
| Reference experiment | Measure fixed responses, timing, and PyTorch memory | [Results and limitations](/en/results) |
| Public notebook validation | Fetch and run the public GitHub notebook on a Colab T4 | [Run validation and failures](/en/verification) |

The original records are [`results/summary.json`](https://github.com/Sunwood-ai-labs/lfm3-colab-lab/blob/main/results/summary.json) and [`results/public-notebook-run.json`](https://github.com/Sunwood-ai-labs/lfm3-colab-lab/blob/main/results/public-notebook-run.json).

## Pages

- [Results and limitations](/en/results): model outputs, measurement conditions, failures, and untested scope.
- [Public notebook validation](/en/verification): the executed version, inputs, and per-model checks.
- [Run the Python CLI](/en/python-cli): run the fixed suites outside the Colab notebook.

Use the language menu in the header to switch to Japanese.
