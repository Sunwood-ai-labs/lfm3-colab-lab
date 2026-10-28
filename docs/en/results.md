# Results from the 2026-10-28 Colab T4 experiment

This page reports the reference experiment run on a Google Colab Tesla T4 on 2026-10-28. It is a separate record from the public notebook validation. See [the public notebook run](/en/verification) for that result.

All six models loaded and completed inference. Answer quality varied by model and task. These small representative tests do not support ranking the models by pass count.

## Reference results by model

| Model | Tested tasks | Result |
| --- | --- | --- |
| LFM3-1.7B | Japanese rewriting, addition tool | 6/6 responses passed |
| LFM3-VL-2B | Video changes, addition tool from an image | 9/9 responses passed |
| LFM3-Audio-2B | Speech recognition and generation, voice chat, audio tool | 17 of 31 records passed, 6 failed, 8 ungraded |
| LFM3-Omni-2B | Questions about audio with images and video | 3 of 9 responses passed |
| d1-2B | Classification, extraction, image locations and counts | 2/2 task groups passed |
| d1-Omni-2B | Decisions from text, images, and audio | 3 of 4 task groups passed |

Pass counts refer to responses or task groups from each model's small fixed suite. The tasks and scoring units differ, so the counts are not directly comparable.

## VRAM and timing

| Model | Peak PyTorch VRAM (allocated / reserved) | Representative time |
| --- | --- | --- |
| LFM3-1.7B | 3.24 / 3.26 GiB | Rewriting: 0.57 s |
| LFM3-VL-2B | 6.05 / 6.17 GiB | 3-second video: 1.23 s |
| LFM3-Audio-2B | 3.86 / 3.90 GiB | Japanese speech recognition: 0.69 s |
| LFM3-Omni-2B | 4.68 / 4.80 GiB | Audio with video: 0.99 s. The answer was wrong. |
| d1-2B | 3.71 / 5.15 GiB | Text decision: 0.124 s |
| d1-Omni-2B | 4.19 / 5.57 GiB | Text decision: 0.122 s |

## Failures and ungraded outputs

LFM3-Audio-2B passed the English voice-chat and audio-tool checks 3/3 each. It passed neither Japanese check: both scored 0/3. Across three attempts to transcribe one 6.96-second Japanese recording, it confused *gun-tou* (群島, archipelago) with *gun-kan* (軍艦, warship). The character error rate for that recording was 14.3%, counting punctuation. This single recording is not a general Japanese recognition score.

Audio generation produced a waveform. Checks for waveform validity and nonzero audio count as passed. This run did not confirm Japanese pronunciation. Whisper also misrecognized both the generated speech and the human recording, so it could not independently verify pronunciation.

The LFM3-Audio-2B translation changed 群島 to `land`. Its sound description identified electronic tones and silence but got the duration wrong. Neither output was automatically scored.

LFM3-Omni-2B answered the image-and-audio questions about a red shape correctly in all three attempts. Its answers about a blue shape were empty in all three attempts. On the audio-and-video question, it returned two numbers instead of choosing the requested one.

d1-Omni-2B passed its text, image, and audio classification task groups. When audio asked for the blue 7, the model answered 3 in all three measured attempts.

## Earlier JSON extraction suite

In a separate existing test, LFM3-1.7B passed 15/15 text-to-JSON responses. LFM3-VL-2B passed 0/10 invoice-image-to-JSON responses. That image test used a large input image. The table's 6.05 / 6.17 GiB peak came from this test. The added video test alone peaked at 3.88 / 3.96 GiB.

VRAM is the maximum allocated and reserved memory reported by PyTorch during inference. It is not total GPU usage and excludes driver and CUDA-context memory.

## Measurement conditions

The experiment used a Tesla T4, FP16, and Eager attention. The main shared versions were Python 3.13.16, PyTorch 2.11.0+cu130, and Transformers 5.21.0. The run did not change Torch or CUDA-related packages installed before it.

Times are medians for short fixed inputs. They exclude model downloads and loading. Text, VL, and d1 measurements followed warm-up. Audio and Omni had no independent warm-up. Those suites measured each case three times. Audio translation and tone description ran once each. The timings cover different inference paths and should not be compared directly.

## Pinned models and code

The model revisions on Hugging Face and SHA-256 hashes of the experiment scripts appear in the [reference result record](https://github.com/Sunwood-ai-labs/lfm3-colab-lab/blob/main/results/summary.json). A script SHA-256 identifies file contents; it is not a Git commit ID.

## Not covered

The experiment did not cover fine-tuning, all 18 audio languages, maximum context length, CPUs or NPUs, comparisons with other models, or empirical probability calibration.

Next, read [the public notebook validation](/en/verification) to compare the one-repetition run with this reference experiment.
