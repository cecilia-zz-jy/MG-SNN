# Visual Magnocellular Dynamics Dataset (VMD)

This repository contains partial examples of the **Visual Magnocellular Dynamics Dataset (VMD)**. These datasets are provided to help researchers test and experiment with our proposed method. Please note that these files are only a subset of the full dataset, which we are currently optimizing and supplementing. More details on the dataset structure and release plans are described below.

## Files Included

1. **test.7z**: 
   - Contains testing examples for the VMD dataset.
   
2. **train_magno.7z**:
   - Contains magnocellular training examples.
   
3. **train_raw.7z**:
   - Contains raw training examples, prior to any augmentation or preprocessing.

## Dataset Overview

In this research, we utilized the output from the magnocellular pathway as the label information for our neural network model and developed the **Visual Magnocellular Dynamics Dataset (VMD)** as illustrated in Figure 4 of our paper. 

This dataset is constructed based on the **Anti-UAV-2021 Challenge dataset** and the **Anti-UAV-2023 Challenge dataset** [ https://anti-uav.github.io/dataset/]. The videos showcase natural and man-made elements in the backgrounds, such as clouds, buildings, trees, and mountains, realistically simulating scenarios encountered in drone surveillance tasks. The dataset includes target objects of varying sizes, from large to extremely small, intensifying the difficulty of object detection. 

To enhance the VMD dataset, the **Anti-UAV-2023 Challenge dataset** was used, specifically for small target recognition tasks. This dataset includes more challenging video sequences featuring dynamic backgrounds, complex rapid movements, and small targets, thereby encompassing a wider range of small target drone scenarios.

### Key Characteristics:
- **Total Samples**: 650 video samples
  - **Training**: 500 samples
  - **Testing**: 150 samples
- **Scenes**: Diverse natural and man-made scenes, including open skies, urban environments, forests, and mountains
- **Targets**: Small objects with varied motion, challenging the model's detection capabilities
- **Motion Complexity**: Static and dynamic backgrounds, targets moving at different speeds and directions
- **Resolution**: 120×100 pixels to ensure computational efficiency
- **Frame Rate**: 20 frames per second
- **Video Duration**: 5 to 10 seconds
- **Preprocessing**: 
  - Pixel value normalization
  - Retention of content within salient bounding boxes for precise labeling

The **VMD dataset** is created based on the magnocellular pathway computational model and developed using the bioinspired library in OpenCV. The preprocessing steps applied ensure high quality and consistency across the dataset.

## Important Notice

This dataset is only a partial release and is meant as an illustrative example. The full VMD dataset includes additional modifications and enhancements, which are not yet public due to ongoing improvements. 

We plan to release a more comprehensive version in the future, along with a supplementary paper that will detail the dataset generation process, including:
- The methodology
- Preprocessing techniques
- Augmentation strategies
- Data selection criteria

This will provide the research community with sufficient information to recreate similar datasets and understand our approach, while maintaining the confidentiality of certain technical aspects of our dataset.

## Usage

To use these files, download the `.7z` archives and extract them using any standard archive tool (such as [7-Zip](https://www.7-zip.org/)). The data can then be used for training and testing machine learning models.

## Citation

If you use this dataset in your research, please cite our work. Citation details will be provided once the supplementary paper is published.

---

We hope this partial dataset will be a helpful resource for your research and development efforts. For any questions or feedback, please feel free to open an issue or contact us directly.

