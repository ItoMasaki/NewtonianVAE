# NewtonianVAE
NewtonianVAE : [here](https://arxiv.org/abs/2006.01959)  
Tactile-Sensitive NewtonianVAE : [here](https://arxiv.org/abs/2203.05955)


## How to train [ sample ]

### Collect data
```bash
./collect_data --config config/sample/collect_dataset/point_mass.yml
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
