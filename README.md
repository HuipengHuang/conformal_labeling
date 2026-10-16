# Provable Selective Auto-Labeling with Reliability Guarantees

This is the official implementation for *Provable Selective Auto-Labeling with Reliability Guarantees* at NeurIPS 2026.

## Setup

Use Python 3.9 or later. Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Repository structure

* `algorithm/selection.py` — Conformal Labeling algorithm.
* `algorithm/preprocess.py` — Data loading and preprocessing.
* `datasets/` — CSV and JSONL files used in the experiments.
* `main.py` — Run Conformal Labeling on the provided data.
* `run_imagenet.py` — Generate ImageNet predictions and confidence scores.

## Usage

### Run Conformal Labeling

The repository includes predictions and confidence scores for:

* MMLU with Qwen3-32B.
* ImageNet with ResNet-34.
* MATH-500 and MATH-L5 with Qwen3-4B-Instruct-2507 and Qwen3-4B-Thinking-2507.

Run the following commands from the repository root:

```bash
# Run on MMLU with Qwen3-32B
python main.py --dataset mmlu --model Qwen3-32B --alpha 0.1

# Run on ImageNet with ResNet-34
python main.py --dataset imagenet --model resnet34 --alpha 0.1

# Run on MATH-500 with Qwen3-4B-Instruct-2507
python main.py --dataset math500 --model Qwen3-4B-Instruct-2507 --alpha 0.05 --n_calib 300
```

Key options:

* `--alpha` — Target FLR level (default: `0.1`).
* `--n_calib` — Number of calibration samples (default: `1000`).
* `--num_trials` — Number of trials (default: `100`).
* `--random` / `--no-random` — Enable or disable randomized p-values (enabled by default).

The script reports the average realized FLR, power, and selection ratio across trials, along with the model's overall error rate.

### Generate ImageNet predictions

To regenerate `datasets/resnet34_imagenet.csv`, point `--path` to the ImageNet validation directory, with one subdirectory per class:

```bash
python run_imagenet.py --path /path/to/imagenet --model resnet34
```
