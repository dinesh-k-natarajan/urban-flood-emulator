# urban-flood-emulator
This repository contains code for our KI2025 paper "Deep learning emulators for large-scale, high-resolution urban pluvial flood prediction"

![abstract](images/fig_dataset_creation_training.png)

## Dataset
 
The dataset can be downloaded from Zenodo: https://doi.org/10.5281/zenodo.17098296

## Installation
Please create a new virtual environment using Python 3.8 and install the required packages using the following command: 

```bash
pip install -r requirements.txt
```

## Usage
### 1. Terrain feature extraction
Extract the additional terrain features from the digital elevation maps using the following script. 

_Note_: Please make sure to define the appropriate data directories in the script.
```bash
python extract_terrain_features.py
```

### 2. Training
The various architectures are defined in the configuration files in `configs/`. To train the UNet architecture, specify the corresponding YAML file as a command line argument. 

_Note:_ The training scripts were tested using a GPU. Please modify the configurations according to your hardware.
```bash
python train.py -c configs/UNet.yaml
```

# Citation
Natarajan, D.K., Stricker, M., Mukherjee, R., Charfuelan, M., Nuske, M., Dengel, A. (2026). Deep Learning Emulators for Large-Scale, High-Resolution Urban Pluvial Flood Prediction. In: Braun, T., Paaßen, B., Stolzenburg, F. (eds) KI 2025: Advances in Artificial Intelligence. KI 2025. Lecture Notes in Computer Science(), vol 15956. Springer, Cham. https://doi.org/10.1007/978-3-032-02813-6_17

```bibtex
@InProceedings{natarajan2025flood,
author="Natarajan, Dinesh Krishna
and Stricker, Marco
and Mukherjee, Rushan
and Charfuelan, Marcela
and Nuske, Marlon
and Dengel, Andreas",
title="Deep Learning Emulators for Large-Scale, High-Resolution Urban Pluvial Flood Prediction",
booktitle="KI 2025: Advances in Artificial Intelligence",
year="2026",
publisher="Springer Nature Switzerland",
pages="228--235",
isbn="978-3-032-02813-6",
doi={10.1007/978-3-032-02813-6_17}
}
```
