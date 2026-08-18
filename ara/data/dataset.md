# Dataset

- **Source**: `E:\WM\turbovla\data\button_demos\random_pw6_lang_1000\demos.h5` (HDF5; WM button collection).
- **Size**: 1000 demonstrations; 764 belong to the 48 train passwords and 236 to the 16 holdout passwords. Episode length range 272–405 frames; mean ≈ 339.5 frames.
- **Fields per episode**: `action` (T,7), `image` / `wrist_image` (T,H,W,3 uint8), `state` (T,8), attributes `password`, `instruction`, `success`, `failed`, `press_count`, `seed`.
- **Task**: two-button password; length-6 passwords over {1,2}; success requires 6 correct presses with no wrong press.
- **Label files**:
  - `runs/full_train_labels.npz`: 317 train episodes with per-frame `/count` arrays; 48/48 train passwords covered. 290 strict replay labels + 27 best-effort labels.
  - `runs/full48_labels.npz`: merged label set (same content as full_train_labels plus best-effort episodes; used by current runs).
- **Filtering**: training uses only episodes present in the label file when `--remaining-cond` or `--curriculum-epochs` is active.
- **Ethics/consent**: synthetic robot manipulation data; no human subjects.
