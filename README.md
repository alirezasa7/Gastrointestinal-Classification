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
    ├── common/             
    ├── data/               
    |   ├── data_loader_kvasir.py
    |   ├── Kermany/
    │   |   ├── train/
    │   |   │   ├── NORMAL
    │   |   |   └── PNEUMONIA
    │   |   └── test/
    │   |       ├── NORMAL
    │   |       └── PNEUMONIA
    |   ├── Cohen/
    │   │   ├── train/
    │   │   │   ├── COVID19
    │   │   │   ├── NORMAL
    │   │   |   └── PNEUMONIA
    │   │   └── test/
    │   │       ├── COVID19
    │   │       ├── NORMAL
    │   │       └── PNEUMONIA
    │   └── Chest_Xray_Masks_and_Labels/
    │       ├── train/
    │       │   ├── images
    │       |   └── masks
    │       └── test/
    │           ├── images
    │           └── masks
    ├── model/
    |   ├── MSR.py
    |   ├── backbone_utils.py
    |   └── transformer.py
    ├── README.md           
    ├── train_Cohen.py
    ├── train_Kermany.py
    ├── train_seg.py
    ├── test_Cohen.py
    ├── test_Kermany.py
    └── test_seg.py             

.../                      
├── common/             
├── data/               
|   ├── data_loader_kvasir.py
│   |   ├── train/
│   |   │   ├── NORMAL
│   |   |   └── Esophagitis
│   |   |   └── Polyps
│   |   |   └── Ulcerative Colitis
│   |   |   └── Dyed and Lifted Polyps
│   |   |   └── Dyed Resection Margins

│   |   ├── test/
│   |   │   ├── NORMAL
│   |   |   └── Esophagitis
│   |   |   └── Polyps
│   |   |   └── Ulcerative Colitis
│   |   |   └── Dyed and Lifted Polyps
│   |   |   └── Dyed Resection Margins
|   ├── Cohen/
│   │   ├── train/
│   │   │   ├── COVID19
│   │   │   ├── NORMAL
│   │   |   └── PNEUMONIA
│   │   └── test/
│   │       ├── COVID19
│   │       ├── NORMAL
│   │       └── PNEUMONIA
│   └── Chest_Xray_Masks_and_Labels/
│       ├── train/
│       │   ├── images
│       |   └── masks
│       └── test/
│           ├── images
│           └── masks
├── model/
|   ├── MSR.py
|   ├── backbone_utils.py
|   └── transformer.py
├── README.md           
├── train_Cohen.py
├── train_Kermany.py
├── train_seg.py
├── test_Cohen.py
├── test_Kermany.py
└── test_seg.py 
'''
