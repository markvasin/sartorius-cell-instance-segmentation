# sartorius-cell-instance-segmentation

## How to run

1. Install dependencies

```bash
# install dependencies using conda environment  
cd sartorius-cell-instance-segmentation
conda create --name sartorius python=3.7
conda activate sartorius
pip install -r requirements.txt
 ```   

2. Download dataset

```bash
cd data
kaggle competitions download -c sartorius-cell-instance-segmentation
 ```   

3. Download detectron2 for local installation (for inference)

```bash
kaggle datasets download -d slawekbiel/detectron-05
```