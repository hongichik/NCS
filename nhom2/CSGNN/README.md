# CSGNN
Codes of paper "Category-aware Self-supervised Graph Neural Network for Session-based Recommendation"


## Requirements
- Python 3.12.13
- PyTorch 2.x

Create and activate a new environment (recommended):
```shell
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### datasets
1. directory  `/datasets/id/` is the dataset of CSGNN model 
2. directory `/datasets/pre_training/` is the embeddings of pre-training
3. directories `/datasets/filter*` are the datasets of ablation experiments
4. `datasets/retailrocket/` contains raw Retailrocket CSV files.

### Prepare Retailrocket
Generate CSGNN training files from raw Retailrocket CSV:

```shell
python datasets/retailrocket_interest.py
```

This command creates:
- `datasets/retailrocket/id/train.txt`
- `datasets/retailrocket/id/test.txt`
- `datasets/retailrocket/id/category_train.txt`
- `datasets/retailrocket/id/category_test.txt`


  
### Train and evaluate
```shell
python main.py --dataset nowplaying --beta 0.005 --embSize 100 > result.output 
python main.py --dataset retailrocket --beta 0.005 --embSize 100 > retailrocket.output
```

Optional GPU selection:
```shell
python main.py --dataset nowplaying --gpu_id 0
python main_sparse.py --dataset diginetica --gpu_id 0
```

Save logs to a specific folder:
```shell
python main.py --dataset retailrocket --gpu_id 0 --log_dir logs
python main.py --dataset retailrocket --gpu_id 0 --log_dir logs --log_file run_retailrocket.log
```
