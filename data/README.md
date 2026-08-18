# Data

按钮任务数据不复制到本目录，直接读取 WM 原始 HDF5：

```
E:\WM\turbovla\data\button_demos\random_pw6_lang_small\demos.h5
E:\WM\turbovla\data\button_demos\random_pw6_lang_100\demos.h5
E:\WM\turbovla\data\button_demos\random_pw6_lang_1000\demos.h5
```

## 已确认的数据结构

每个 HDF5 group：
- `action`: `(T, 7) float32`
- `image`: `(T, 256, 256, 3) uint8`
- `wrist_image`: `(T, 256, 256, 3) uint8`
- `state`: `(T, 8) float32`
- attrs: `password`, `instruction`, `success`, `failed`, `press_count`

## 检查命令

```powershell
python scripts\check_button_data.py --h5 "E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5"
```

密码 train/holdout 划分在：

```
button_task\password_split.json
```

