# Detect Temperatures: Semantic Segmentation and Temperature Prediction in Urban Landscapes  

## Overview  
This project explores the innovative intersection of **semantic segmentation** and **temperature prediction** within urban environments using the **Cityscapes Dataset**. By leveraging the hierarchical capabilities of **Swin Transformers**, we uncover relationships between urban elements visible in street-level imagery and their corresponding ambient temperatures. The research provides insights into **urban climate dynamics**, aiding smarter city planning and environmental studies.  

## Motivation  
Urban areas exhibit diverse **microclimates**, influenced by features like buildings, roads, vegetation, and water bodies. Traditional temperature prediction methods often rely on **point-based meteorological data**, which fails to capture urban spatial variability. This project aims to bridge that gap by utilizing **visual cues from urban imagery** to predict temperature variations, considering factors such as:  
- Urban heat island effects  
- Sun exposure and reflection  
- Vegetation density  
- Temporal dynamics (e.g., daily and seasonal variations)  

## Dataset  
We utilize the **Cityscapes Dataset**, a benchmark for urban street scene understanding, which includes:  
- **5,000 images** with fine annotations (30 classes grouped into 8 categories: humans, vehicles, constructions, nature, etc.)  
- Ambient **temperature metadata** for each image  
- Images captured in **50 European cities** over different months in favorable weather conditions  
- High-resolution imagery (720p), with a training set of **2,975 images**, a validation set of **500 images**, and a test set of **1,525 images**  

This dataset's diversity makes it ideal for exploring the relationship between **urban features** and temperature prediction.  

## Methodology  
The model architecture integrates **semantic segmentation** and **temperature prediction** into a multimodal network with two main stages:  
1. **Segmentation Stage**: Extracts semantic features from urban scenes.  
2. **Temperature Prediction Stage**: Utilizes segmentation features to predict ambient air temperatures.  

We explore two approaches:  
- **Direct Approach**: A regression model predicts temperatures from segmentation outputs.  
- **Integrated Approach**: Fine-tunes a **Swin Transformer** to directly predict temperatures from segmentation features.  

### Computational Strategy  
Given our computational resources (Colab Pro GPUs), we adopt a flexible training strategy:  
- Direct training on the Cityscapes Dataset when feasible.  
- **Transfer learning** using a pre-trained Swin Transformer (trained on ImageNet and ADE20k) for fine-tuning when necessary.  

### Loss Function  
We use a **combined loss function** that incorporates:  
- **Cross-Entropy Loss**: For accurate segmentation.  
- **Mean Squared Error (MSE)**: For precise temperature prediction.  

## Evaluation  
The project is evaluated based on:  
- **Root Mean Squared Error (RMSE)** for temperature prediction. Success is defined as achieving RMSE values ≤ 6 degrees, on par with or surpassing baseline models in existing literature.  
- The effectiveness of the combined loss function in balancing segmentation and temperature prediction tasks.  

## Results  
By successfully integrating segmentation features with temperature prediction, the project highlights the complex relationship between urban features and climate dynamics.  

## Installation  

### Prerequisites  
- Python 3.8+  
- Colab Pro account for training  
- Libraries: PyTorch, torchvision, transformers, NumPy, Matplotlib  

### Steps  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/detect-temperatures.git  
   cd detect-temperatures  
