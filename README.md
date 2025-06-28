# A Lightweight Multi-Scale Refinement Network for Gastrointestinal Disease Classification
This is the implementation of the paper "A Lightweight Multi-Scale Refinement Network for Gastrointestinal Disease Classification"
For more information, check out our paper on ()


![architecture-6](https://github.com/user-attachments/assets/523434e3-a66b-44d7-a6f1-b8f5c02cb248)


# Requirements
-Python 3.11.11

-PyTorch version: 2.6.0

-CUDA version: 12.4

-CUDA device: Tesla T4

-TensorBoard version: 2.18.0

-NumPy version: 1.26.4

-scikit-learn version: 1.2.2

-timm version: 1.0.15

# Datasets
Two gastrointestinal datasets were used in this paper as follows:

https://datasets.simula.no/kvasir/

https://osf.io/84e7f/


                            
    ../                                  
    ├── data/               
    |   ├── data_loader_kvasir.py
    │   |   ├── train/
    │   |   │   ├── NORMAL
    │   |   |   └── Esophagitis
    │   |   |   └── Polyps
    │   |   |   └── Ulcerative Colitis
    │   |   |   └── Dyed and Lifted Polyps
    │   |   |   └── Dyed Resection Margins
    │   |   └── test/
    │   |   │   ├── NORMAL
    │   |   |   └── Esophagitis
    │   |   |   └── Polyps
    │   |   |   └── Ulcerative Colitis
    │   |   |   └── Dyed and Lifted Polyps
    │   |   |   └── Dyed Resection Margins
    |   ├── data_loader_Gastrovision.py
    │   │   ├── train/
    │   │   │   ├── Accessory tools
    │   │   │   └── Angiectasia
    │   │   |   └── Barrett's esophagus
    │   │   │   └── Blood in lumen
    │   │   │   └── Cecum
    │   │   │   └── Colon diverticula
    │   │   │   └── Colon polyps
    │   │   │   └── Colorectal cancer
    │   │   |   └── Duodenal bulb
    │   │   │   └── Dyed-lifted-polyps
    │   │   │   └── Dyed-resection-margins
    │   │   │   └── Erythema 
    │   │   │   └── Esophageal varices
    │   │   │   └── Esophagitis
    │   │   |   └── Gastric polyps
    │   │   │   └── Gastroesophageal junction normal z-line
    │   │   │   └── Ileocecal valve
    │   │   │   └── Mucosal inflammation large bowel 
    │   │   │   └── Normal esophagus
    │   │   │   └── Normal mucosa and vascular pattern in the large bowel
    │   │   |   └── Normal stomach
    │   │   │   └── Pylorus
    │   │   │   └── Resected polyps, Resection margins
    │   │   │   └── Retroflex rectum  
    │   │   │   └── Small bowel terminal ileum 
    │   │   │   └── Ulcer    
    │   │   └── test/
    │   │   │   ├── Accessory tools
    │   │   │   └── Angiectasia
    │   │   |   └── Barrett's esophagus
    │   │   │   └── Blood in lumen
    │   │   │   └── Cecum
    │   │   │   └── Colon diverticula
    │   │   │   └── Colon polyps
    │   │   │   └── Colorectal cancer
    │   │   |   └── Duodenal bulb
    │   │   │   └── Dyed-lifted-polyps
    │   │   │   └── Dyed-resection-margins
    │   │   │   └── Erythema 
    │   │   │   └── Esophageal varices
    │   │   │   └── Esophagitis
    │   │   |   └── Gastric polyps
    │   │   │   └── Gastroesophageal junction normal z-line
    │   │   │   └── Ileocecal valve
    │   │   │   └── Mucosal inflammation large bowel 
    │   │   │   └── Normal esophagus
    │   │   │   └── Normal mucosa and vascular pattern in the large bowel
    │   │   |   └── Normal stomach
    │   │   │   └── Pylorus
    │   │   │   └── Resected polyps, Resection margins
    │   │   │   └── Retroflex rectum  
    │   │   │   └── Small bowel terminal ileum 
    │   │   │   └── Ulcer
    ├── model/
    |   ├── backbone_utils.py
    |   ├── CBAM.py
    |   └── CROSS_ATTENTION.py
    |   └── transformer.py
    |   └── MSR.py
    |   └── FocalLoss.py
    ├── README.md           
    ├── Train_Kvasir.py
    ├── Train_Gastrovision.py
    ├── Grad_Kvasir.py
    ├── Grad_Gastrovision.py
    ├── Conf_Kvasir.py
    └── Conf_Gastrovision.py             

# Training 
to train the model use the following command:

python Train_Kvasir.py --bsz 32
>                     --lr 1e-4
>                     --niter 60
>                     --data_dir data/data_loader_kvasir.py
>                     --logpath "your_experiment_name"
> ```

python Train_Gastrovision.py --bsz 32
>                     --lr 1e-4
>                     --niter 60
>                     --data_dir data/data_loader_Gastrovision.py
>                     --logpath "your_experiment_name"
> ```

## Arguments

- `--bsz`: Batch size for training.
- `--lr`: Learning rate for the optimizer.
- `--niter`: Number of training iterations (epochs).  
- `--data_dir`: Directory where the dataset is located. 
- `--logpath`: Directory to save the best model checkpoint.

  ## Citation
If you use this repository in your work, please cite the following paper:
