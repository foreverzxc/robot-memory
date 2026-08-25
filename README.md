# Robot Memory

Research code for memory-augmented vision-action policies in robotic
manipulation.

The repository currently contains two experiment families:

- A button-sequence benchmark with sequence modeling and test-time memory
  updates.
- RoboMME PickXtimes and SwingXtimes experiments using a memory-augmented
  action decoder.

The repository name is intentionally method-agnostic. Future memory
mechanisms, including linear-attention memory, can be added without changing
the project identity.

## Repository layout

~~~text
robot-memory/
|-- button_task/             Button benchmark data and environment helpers
|-- references/              TTT and experiment reference implementations
|-- scripts/                 Training, evaluation, caching, and diagnostics
|-- train_button.py          Button baseline training entry point
|-- train_button_ttt.py      Button sequence-memory training entry point
|-- vendor/                  Vendored model utilities used by button runs
|-- ara/                     Research notes and experiment evidence
|-- requirements_button.txt  Button benchmark dependencies
|-- requirements_extra.txt   Optional research dependencies
~~~

Generated outputs are intentionally excluded from Git. Keep checkpoints,
HDF5 data, feature caches, videos, and logs in runs/ or data/ locally.

## Button benchmark

The button experiments use two camera views and a 7-D robot action. The model
receives the complete password sequence; it must use the observation history
to determine progress. The rollout scripts report success, failed episodes,
press count, and step count, and can generate GIF visualizations.

The dataset is not included. Set the HDF5 path explicitly:

~~~bash
python train_button.py \
  --h5 /path/to/demos.h5 \
  --split button_task/password_split.json \
  --out runs/button_baseline
~~~

For the sequence-memory experiment:

~~~bash
python train_button_ttt.py \
  --h5 /path/to/demos.h5 \
  --split button_task/password_split.json \
  --out runs/button_memory \
  --ttt online
~~~

## RoboMME experiments

The RoboMME scripts are:

~~~text
scripts/cache_robomme_ttt.py       Build compact feature caches
scripts/split_robomme_cache.py     Make reproducible episode splits
scripts/evaluate_robomme_ttt.py    Offline held-out action evaluation
scripts/rollout_robomme_ttt.py     Simulator rollout and MP4 visualization
~~~

The simulator rollout uses the vendored base-policy runtime in
`vendor/base_policy/`. It does not require a separate model-code checkout.
Missing base-policy assets are downloaded into `weights/` from Hugging Face at
startup. The DINOv3 repository may require `hf auth login` because it is gated.

The current offline metrics are action-space diagnostics. They are not
simulator success rates. A simulator rollout must report episode outcomes
such as success, failure, timeout, and a saved video.

Run one simulator episode after installing the official RoboMME environment:

~~~bash
python scripts/rollout_robomme_ttt.py \
  --task PickXtimes \
  --episode 0
~~~

Use `--task SwingXtimes` for the second checkpoint, or `--episode -1` to run
all episodes in the selected split. The script reports success rate and writes
an MP4 plus a JSON summary under `runs/robomme_rollouts/`. The two small
task-specific normalization files are tracked in `weights/`; model weights
themselves stay local.

The first run downloads the frozen base checkpoint, DINOv3, and BERT into
`weights/`. To authenticate for the gated DINOv3 repository:

~~~bash
pip install -U huggingface_hub
hf auth login
~~~

You can also download these assets before starting the simulator:

~~~bash
python scripts/download_robomme_base.py
~~~

The trained TTT decoder checkpoints are experiment artifacts and are not
published by this repository. Place them at the paths listed in
`weights/README.md` or pass `--checkpoint` explicitly.

## Environment

Create a fresh Python environment on the target machine. Do not copy a
Windows virtual environment to Linux. The button experiments require PyTorch,
NumPy, h5py, robosuite, MuJoCo, einops, and the packages listed in the
requirements files.

For RoboMME simulator rollouts, install the official RoboMME benchmark
separately. It uses ManiSkill/SAPIEN and Vulkan:

~~~bash
git clone https://github.com/RoboMME/robomme_benchmark.git
cd robomme_benchmark
uv sync
uv pip install -e .
~~~

The current repository does not vendor the RoboMME simulator, training data,
large model checkpoints, or generated rollout artifacts.

## License and provenance

This repository contains research code and vendored components with their own
licenses. Check the license file in each vendored component before
redistributing it.
