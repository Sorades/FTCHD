# FTCHD

## An interpretable deep learning model for first-trimester fetal cardiac screening
![Study Overview](assets/study_overview.png)

Effective first-trimester screening for congenital heart disease (CHD) remains an unmet clinical need, hampered by technical limitations and the absence of validated diagnostic tools. To address this, we collected a vast cohort of 108,521 first-trimester cardiac screenings conducted across multiple regions in China, from which 8,062 Doppler flow four-chamber view images were selected. Using this curated dataset, we developed and validated an interpretable deep learning (DL) model that mimics clinical reasoning with diastolic flow patterns, providing accurate and explainable CHD diagnosis in first-trimester. Interpretability analyses confirmed its diagnostic logic strongly aligns with clinical expertise. In rigorous evaluations, the model demonstrated high accuracy across multiple external validation datasets, matched or surpassed experienced clinicians, and showed potential to augment their diagnostic capabilities. To our knowledge, this is the first validated interpretable DL system for first-trimester CHD screening, potentially enabling earlier intervention through an advanced diagnostic window.

## Env
Recommended environment:
- python 3.12.4
- pytorch 2.4.0
- torchvision 0.19.0
- lightning 2.4.0
- transformers 4.52.3

See `pyproject.toml` for more details. If you are using [pixi](https://github.com/prefix-dev/pixi/), simply run:
```bash
pixi i && pixi shell
```
We also provide a conda environment file `environment.yaml` for easy setup:
```bash
conda env create -f environment.yaml
conda activate ftchd
```
## Training & Evaluation
```bash
# For binary classification
python src/fit.py exp=binary test=true
# For subtype classification
python src/fit.py exp=subtype test=true
```

## Data Availability
The researchers who apply to access the data should provide a methodologically sound proposal to qingqingwu@ccmu.edu.cn to gain access, and they must sign a data access agreement during the application.

## Citation
If you find this work useful for your research, please kindly cite our paper:
```
@article{lei2025interpretable,
  title={An interpretable deep learning model for first-trimester fetal cardiac screening},
  author={Lei, Wenjia and Wen, Chi and Li, He and Yang, Shuihua and Shen, Kuifang and Yuan, Hongxia and Li, Hezhou and Xu, Hong and Gao, Xinru and Zhang, Simin and others},
  journal={npj Digital Medicine},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
