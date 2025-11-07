# Distributed Machine Learning for Consistency-Violation Analysis
### Single GPU and Multi-Worker Tensorflow Implementation

This repository contains the codebase for analyzing **Consistency-Violating Faults (CVFs)** using feed-forward and distributed machine-learning models.  
It supports both **single-GPU training** and **multi-node distributed training** via `tf.distribute.MultiWorkerMirroredStrategy`.

---

## Repository Structure
v2/ 
|-src/ #core training module
|   |-trainer_distributed.py
|   |- trainer_single_gpu.py
|   |- ...
|-datasets/ #Input CSV datasets
|-logs/
|-generatedataset.py # Generate dataset using dijkstra's token ring algorithm
|-main.py
|-launch_training.sh # Launcher for both  single and distributed

## How to run?

### **A. Local single-GPU Training**

- Generate dataset
    `python3 generatedataset.py`
    This will make two datasets:
    1. Training dataset including data from `generatedataset(starting_node, max_node, max_pred_node)`
    2. Actual data set for prediction from `generatedataset(max_prediction_node,prediction_node)`

