# Run the fixed tests with the Python CLI

The Colab notebook is the main way to use this project. Use the CLI to run a fixed test from a Tesla T4 environment. It requires Python 3.10 or later and Torch with CUDA support for a T4. The CLI does not fall back to a CPU or another GPU.

## Check the T4 environment

Activate a Python environment that already has Torch and CUDA support for a T4. Run this check. The second line must show Tesla T4. The `requirements-colab.txt` file does not include Torch or CUDA-related packages.

```console
uv run --active --no-sync -- python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
uv pip install --requirement requirements-colab.txt
uv pip install --no-deps --editable .
```

You do not need to install CLI packages to run the Colab notebook.

## Prepare the fixed inputs

This command creates image, audio, and video inputs in `./runs/inputs`. The audio uses a Japanese sample from Google FLEURS. See [NOTICE](https://github.com/Sunwood-ai-labs/lfm3-colab-lab/blob/main/NOTICE.md) for data sources and licenses.

```console
uv run --active --no-sync -- python -m lfm3_lab prepare --output-dir ./runs --cache-dir ./cache
```

## Run a model

The `run` command runs one model at a time. After preparing the inputs, choose one of these keys.

| Model | Key |
| --- | --- |
| LFM3-1.7B | `text` |
| LFM3-VL-2B | `vision` |
| LFM3-Audio-2B | `audio` |
| LFM3-Omni-2B | `omni` |
| d1-2B | `d1` |
| d1-Omni-2B | `d1_omni` |

This example runs LFM3-1.7B three times, the default.

```console
uv run --active --no-sync -- python -m lfm3_lab run --model text --output-dir ./runs
```

Set `--repeats` to use a different count. The command saves answers, measurements, and logs under `./runs`.

The fixed tests for Omni and d1-Omni also use audio created by the Audio suite. Run the Audio suite first if you run either model by itself.

## Show and export results

Run `summarize` to display the aggregate results for all models. `export` puts the results and run logs into a ZIP file.

```console
uv run --active --no-sync -- python -m lfm3_lab summarize --output-dir ./runs
uv run --active --no-sync -- python -m lfm3_lab export --output-dir ./runs
```

The ZIP is a local artifact. The repository does not include it.

## Call the Python API

Run this example in Jupyter or Colab with the package installed in the notebook's environment.

```python
from lfm3_lab import Lab

lab = Lab(output_dir="runs", cache_dir="cache")
lab.prepare()
result = lab.run("text", repeats=3)
lab.show(result)
```

Use `run_text` for a free-form prompt.

```python
answer = lab.run_text("Add 18 and 24.", repeats=1)
print(answer.answer)
```

The pass counts are for small fixed cases. They do not measure general model performance.
