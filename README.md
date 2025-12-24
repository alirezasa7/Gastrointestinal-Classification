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
Six gastrointestinal datasets were used in this paper as follows:

Kvasir-v1 :  https://datasets.simula.no/kvasir/

GastroVision : https://datasets.simula.no/gastrovision/

Kvasir-Capsule : https://datasets.simula.no/kvasir-capsule/

Kvasir-v2 : https://datasets.simula.no/kvasir/

 Hyper-Kvasir : https://datasets.simula.no/hyper-kvasir/

WCEBleedGen https://zenodo.org/records/10156571




                            
    ../                                  
    ├── data/               
    |   ├── data_loader_kvasir_v1.py
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
    |   ├── data_loader_Kvasir_Capsule.py
    │   |   ├── train/
    │   |   │   ├── Ampulla of vater
    │   |   |   └── Angiectasia
    │   |   |   └── Blood - fresh
    │   |   |   └── Blood - hematin
    │   |   |   └── Erosion
    │   |   |   └── Erythema
    │   |   |   └── Foreign body
    │   |   |   └── Ileocecal valve
    │   |   |   └── Lymphangiectasia
    │   |   |   └── Normal clean mucosa
    │   |   |   └── Polyp
    │   |   |   └── Pylorus
    │   |   |   └── Reduced mucosal view
    │   |   |   └── Ulcer
    │   |   └── test/
    │   |   │   ├── Ampulla of vater
    │   |   |   └── Angiectasia
    │   |   |   └── Blood - fresh
    │   |   |   └── Blood - hematin
    │   |   |   └── Erosion
    │   |   |   └── Erythema
    │   |   |   └── Foreign body
    │   |   |   └── Ileocecal valve
    │   |   |   └── Lymphangiectasia
    │   |   |   └── Normal clean mucosa
    │   |   |   └── Polyp
    │   |   |   └── Pylorus
    │   |   |   └── Reduced mucosal view
    │   |   |   └── Ulcer
    |   ├── data_loader_Hyper_Kvasir.py
    │   |   ├── train/
    │   |   │   ├── barretts
    │   |   |   └── barretts-short-segment
    │   |   |   └── bbps-0-1
    │   |   |   └── bbps-2-3
    │   |   |   └── cecum
    │   |   |   └── dyed-lifted-polyps
    │   |   |   └── dyed-resection-margins
    │   |   |   └── esophagitis-a
    │   |   |   └── esophagitis-b-d
    │   |   |   └── hemorrhoids
    │   |   |   └── ileum
    │   |   |   └── impacted-stool
    │   |   |   └── polyps
    │   |   |   └── pylorus
    │   |   |   └── retroflex-rectum
    │   |   |   └── retroflex-stomach
    │   |   |   └── ulcerative-colitis-grade-0-1
    │   |   |   └── ulcerative-colitis-grade-1
    │   |   |   └── ulcerative-colitis-grade-1-2
    │   |   |   └── ulcerative-colitis-grade-2
    │   |   |   └── ulcerative-colitis-grade-2-3
    │   |   |   └── ulcerative-colitis-grade-3
    │   |   |   └── z-line
    │   |   └── test/
    │   |   │   ├── barretts
    │   |   |   └── barretts-short-segment
    │   |   |   └── bbps-0-1
    │   |   |   └── bbps-2-3
    │   |   |   └── cecum
    │   |   |   └── dyed-lifted-polyps
    │   |   |   └── dyed-resection-margins
    │   |   |   └── esophagitis-a
    │   |   |   └── esophagitis-b-d
    │   |   |   └── hemorrhoids
    │   |   |   └── ileum
    │   |   |   └── impacted-stool
    │   |   |   └── polyps
    │   |   |   └── pylorus
    │   |   |   └── retroflex-rectum
    │   |   |   └── retroflex-stomach
    │   |   |   └── ulcerative-colitis-grade-0-1
    │   |   |   └── ulcerative-colitis-grade-1
    │   |   |   └── ulcerative-colitis-grade-1-2
    │   |   |   └── ulcerative-colitis-grade-2
    │   |   |   └── ulcerative-colitis-grade-2-3
    │   |   |   └── ulcerative-colitis-grade-3
    │   |   |   └── z-line
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
    |   ├── data_loader_WCEBleedGen.py
    │   |   ├── train/
    │   |   │   ├── bleeding
    │   |   |   └── non-bleeding
    │   |   └── test/
    │   |   │   ├── bleeding
    │   |   |   └── non-bleeding
    ├── model/
    |   ├── backbone_utils.py
    |   ├── CBAM.py
    |   └── CROSS_ATTENTION.py
    |   └── transformer.py
    |   └── MSR.py
    |   └── FocalLoss.py
    ├── README.md  
    ├── dataloader_selection.py
    ├── Train.py
    ├── Grad_Kvasir.py
    ├── Grad_Gastrovision.py
    ├── Conf_Kvasir.py
    └── Conf_Gastrovision.py             

# Training 
to train the model use the following command:

python Train.py --bsz 32
>               --lr 1e-4
>               --niter 60
>               --dataset "name_of_the_selected_dataset"
>               --num_classes "class_numbers_of_the_selected_dataset"
>               --data_dir "your_dataset_directory"
>               --logpath "your_experiment_name"



# Testing 
to test the model use the following command:
python Test.py --bsz 32
>               --dataset "name_of_the_selected_dataset"
>               --num_classes "class_numbers_of_the_selected_dataset"
>               --data_dir "your_dataset_directory"
>               --weights "your_weights_directory"




## Arguments

- `--bsz`: Batch size for training.
- `--lr`: Learning rate for the optimizer.
- `--niter`: Number of training iterations (epochs).  
- `--data_dir`: Directory where the dataset is located. 
- `--logpath`: Directory to save the best model checkpoint.
- `--weights`: Directory to use the best model checkpoint.

  ## Citation
If you use this repository in your work, please cite the following paper:
