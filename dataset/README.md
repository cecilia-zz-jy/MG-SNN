# Visual Magnocellular Dynamics Dataset (VMD)

## Overview

The **Visual Magnocellular Dynamics Dataset (VMD)** is developed as part of our research on leveraging the magnocellular pathway for neural network training. This dataset is based on the **Anti-UAV-2021 Challenge dataset** and the **Anti-UAV-2023 Challenge dataset** [ https://anti-uav.github.io/dataset/], and has been specifically curated to simulate realistic drone surveillance scenarios.

In our research, the output from the magnocellular pathway is used as label information for the neural network model, and the **VMD dataset** has been created to represent challenging surveillance environments, including diverse natural and man-made scenes.

### Dataset Composition

The VMD dataset consists of **650 video samples**:
- **500 training samples**: Found in `train_magno.7z` and `train_raw.7z`.
- **150 test samples**: Found in `test.7z`.

### Dataset Details

The videos in the VMD dataset feature scenes with diverse natural and man-made elements, such as clouds, buildings, trees, and mountains. These scenes have been selected to closely simulate environments encountered during drone surveillance tasks. Key features of the dataset include:
- **Varied target sizes**: Objects of varying sizes, from large to extremely small, making object detection more challenging.
- **Dynamic backgrounds**: Sequences include dynamic backgrounds and complex movements, intensifying the difficulty of small target recognition.
- **Enriched small target tasks**: With the integration of challenging sequences from the Anti-UAV-2023 Challenge dataset, the VMD dataset covers a wider range of small target drone scenarios.

### Video and Preprocessing Information

- **Resolution**: All video frames are resized to **120×100 pixels** to ensure computational efficiency.
- **Frame Rate**: Each video is set at **20 frames per second**, with durations ranging from 5 to 10 seconds.
- **Salient content only**: The dataset retains only the content within salient bounding boxes for accurate labeling.
- **Normalization**: Pixel values are normalized to ensure consistency across the dataset.
- The **VMD dataset** is created based on the magnocellular pathway computational model and developed using the bioinspired library in OpenCV. The preprocessing steps applied ensure high quality and consistency across the dataset.

## File Structure

The VMD dataset is divided into the following files for easy access:

- `train_magno.7z`: Contains training data with magnocellular pathway preprocessing applied.
- `train_raw.7z`: Contains raw training data without magnocellular pathway preprocessing.
- `test.7z`: Contains test data for model evaluation.

### Usage Instructions

1. Download and extract the `.7z` files.
2. Load the dataset into your preferred machine learning environment.
3. Preprocessed (magnocellular pathway) data is provided in the `train_magno.7z` file, and raw training data is available in `train_raw.7z`.

## Future Plans

We are currently optimizing and supplementing the VMD dataset to improve both its quality and volume. Once this process is complete, we plan to release more detailed documentation, including:
- The methodology behind the dataset creation.
- Preprocessing and augmentation techniques.
- Data selection criteria.

## Citation

If you use this dataset in your research, please cite our work.

---

We hope this partial dataset will be a helpful resource for your research and development efforts. For any questions or feedback, please feel free to open an issue or contact us directly.
