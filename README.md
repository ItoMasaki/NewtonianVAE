# NewtonianVAE
NewtonianVAE : [here](https://arxiv.org/abs/2006.01959)  
Tactile-Sensitive NewtonianVAE : [here](https://arxiv.org/abs/2203.05955)

## Directory architecture
.  
├── README.md  
├── collect_data.py  
├── config  
│   └── sample  
│       ├── collect_dataset  
│       │   ├── point_mass.yml  
│       │   └── reacher2d.yml  
│       └── train  
│           └── train.yml  
├── environments  
│   ├── __init__.py  
│   ├── reacher_nvae.py  
│   └── reacher_nvae.xml  
├── models  
│   ├── __init__.py  
│   ├── controller.py  
│   ├── distributions.py  
│   └── model.py  
├── requirements.txt  
├── reset.bash  
├── train.py  
└── utils  
    ├── memory.py  
    └── visualize.py  

## How to train [ sample ]

### Collect data
```bash
./collect_data
```

### Train model
```bash
./train
```

## Dependencies
pixyz>=0.3.3  
torch>=1.12.0  
numpy>=1.23.1  
PyYaml>=5.3.1  
matplotlib>=3.5.2  
tqdm>=4.64.0  
dm_control>=1.0.8  
tensorboardX>=2.5.1  
