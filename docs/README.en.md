# LFM3 Colab T4 experiments

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/LFM3_Colab_T4.ipynb)

Try six Liquid AI models on a Google Colab NVIDIA Tesla T4. Run examples for text rewriting and tool calls, image and video questions, speech recognition and voice chat, and structured decisions. The notebook has one named cell per model. It fetches experiment code from a fixed commit in this repository. Model weights download from Hugging Face and are not included here.

## Run the notebook

1. Open the notebook with the Colab badge above.
2. In Colab, change the runtime type and select a T4 GPU.
3. Choose **Runtime > Run all**. Each model reports its result below its cell.

To run one model by itself, run the shared setup cell first. The Omni and d1-Omni cells also need audio from the Audio cell, so run Audio first. The notebook stops before inference if the GPU is not a Tesla T4. Colab may not offer a T4 in every session. Model downloads take time on the first run.

## What the six models do

| Model | Try it for | Historical result from 2026-10-28 |
| --- | --- | --- |
| LFM3-1.7B | Text rewriting and text tool calls | 6/6 responses passed |
| LFM3-VL-2B | Image and video questions, image tool calls | 9/9 responses passed |
| LFM3-Audio-2B | Speech recognition, speech generation, voice chat | English voice chat and voice tool: 3/3 each. Both Japanese tests: 0/3 each |
| LFM3-Omni-2B | Questions about audio with images or video | 3/9 passed. All three blue-shape answers were empty |
| d1-2B | Classification, extraction, image locations and counts | 2/2 task groups passed |
| d1-Omni-2B | Decisions from text, images, and audio | 3/4 task groups passed |

These small suites test different tasks, so their pass counts do not rank the models. See the [results and limitations](results.md) for test conditions, failures, and measurements.

## Result date

The repeated reference measurements come from a separate Colab T4 experiment on 2026-10-28. The public notebook was also fetched from GitHub and completed on a Tesla T4 with FP16, including all six model suites and the editable text example. This first-run notebook uses one repetition per case. See [the executed notebook](https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/results/T4_2026-10-28.ipynb) and [the validation record](verification.md).

## License

The original experiment code and documentation in this repository use the MIT License. The model weights are not included. Each model is published under the LFM Open License v1.0, which is separate from the MIT License. See [NOTICE](../NOTICE.md) for model and data sources.

## More information

- [Japanese quick start](../README.md)
- [Results and limitations](results.md)
- [Optional Python CLI guide (Japanese)](python-cli.md)
- [Third-party model and data notices](../NOTICE.md)

The notebook starts with one repetition per case. Set `REPEATS = 3` in the first configuration cell for repeated measurements. A one-sample timing is a demonstration, not a stable comparison.
