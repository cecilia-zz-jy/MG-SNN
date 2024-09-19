# Motion Feature Extraction Using Magnocellular-inspired Spiking Neural Networks for Drone Detection

This repository contains the code for the paper "Motion Feature Extraction Using Magnocellular-inspired Spiking Neural Networks for Drone Detection." This project aims to use a magnocellular-inspired spiking neural network (MG-SNN) for effective motion feature extraction to enhance drone detection.

## Installation
To set up the environment for this project, follow the steps below:
1. Clone the repository:
   ```bash
   git clone https://github.com/cecilia-zz-jy/MG-SNN.git
   cd MG-SNN
    ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
    ```
# Usage
## Data Loader
The ``dataloader.py`` script is used for loading the dataset. Make sure to place your dataset in the appropriate directory or modify the script to point to the correct path.

## Training
To train the MG-SNN model, use the ``train.py`` script. You can modify the training parameters such as learning rate, batch size, and number of epochs directly in the script or via command-line arguments.
   ```bash
   python train.py
   ```
# Utilities
The ``utils.py`` script contains various utility functions used throughout the project, such as data preprocessing and evaluation metrics.

## Project Structure
* ``dataloader.py``: Script for loading and preprocessing the dataset.
*  ``dataset``: The VMD dataset.
* ``train.py``: Script for training the MG-SNN model.
* ``test.py``: Script for testing and evaluating the trained model.
* ``utils.py``: Utility functions used across the project.
* ``requirements.txt``: A list of required Python packages.

## Citation

  ```bash
   @article{zheng2024motion,
     title={Motion Feature Extraction Using Magnocellular-inspired Spiking Neural Networks for Drone Detection},
     author={Jiayi Zheng, Yaping Wan, Xin Yang, Hua Zhong, Minghua Du, and Gang Wang},
     journal={Journal Name},
     year={2024}
   }
  ```

## Acknowledgments
This project was inspired by and references code from the [S4NN repository](https://github.com/SRKH/S4NN.git). We would like to thank the authors of this repository for their valuable contributions to the field.


