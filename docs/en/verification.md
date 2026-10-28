# Public notebook validation

On 2026-10-28, the public notebook was fetched from GitHub and run with the Google Colab Python CLI. The run used a Tesla T4 and FP16. All six model suites and the editable text example completed without infrastructure errors.

- [Try the notebook in Colab](https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/LFM3_Colab_T4.ipynb)
- [Open the notebook with saved outputs](https://colab.research.google.com/github/Sunwood-ai-labs/lfm3-colab-lab/blob/main/notebooks/results/T4_2026-10-28.ipynb)
- [Read the machine-readable validation record](https://github.com/Sunwood-ai-labs/lfm3-colab-lab/blob/main/results/public-notebook-run.json)

## What was checked

- The input notebook matched the public GitHub version, including each cell's source.
- The actual Git commit, pinned model revisions, and source manifest matched the recorded values.
- All six models produced outputs on a Tesla T4 with FP16.
- No code cell reported an error, and the preinstalled Torch and CUDA-related packages remained unchanged.
- The output ZIP passed its CRC check and contained no credentials or model weights.

The editable prompt asked for the sum of 18 and 24 in one sentence. The model returned 42 in a two-sentence answer. This single prompt is an example, not a general instruction-following guarantee.

## Results from this one-repetition run

The notebook's initial setting uses one repetition per case. These counts belong to the public notebook run. Keep them separate from the [three-repetition reference experiment](/en/results).

| Model | Checks passed / records | Ungraded | Functional failures |
| --- | ---: | ---: | ---: |
| LFM3-1.7B | 2 / 2 | 0 | 0 |
| LFM3-VL-2B | 3 / 3 | 0 | 0 |
| LFM3-Audio-2B | 9 / 15 | 4 | 2 |
| LFM3-Omni-2B | 1 / 3 | 0 | 2 |
| d1-2B | 2 / 2 task groups | 0 | 0 |
| d1-Omni-2B | 3 / 4 task groups | 0 | 1 |

Speech content and pronunciation of the TTS outputs were not graded in this run.

These counts do not rank the models. The tested functions and pass criteria differ. Audio's passed checks include waveform-only generation checks. The run also inspected the content of speech recognition, translation, and sound-description outputs. One d1 task group contains multiple questions.

Wrong answers remain in the results for Japanese voice chat and mixed inputs. A successful runtime and a correct model answer are separate outcomes.

## Versions used for reproduction

| Item | Value |
| --- | --- |
| Commit used to fetch the notebook | `7b2f79224a6ad0c83cd31f607b1ff42a941da9ca` |
| Experiment-code commit fetched by the notebook | `1fda1d099245243e9af2ab235f34280a7230779e` |
| SHA-256 of the executed notebook | `7d8ff55a2214cdebdc3b568d2098aa221333fe116017ab1f3c8ee116805745c1` |

The executed notebook contains media outputs. If GitHub's preview does not show them, open the Colab link above. You can read the saved outputs without connecting to a runtime.

The T4 runtime was stopped after the results were recovered.
