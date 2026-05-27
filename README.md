# Optimized-DAPT-Distil - Installation Guide

This folder contains all the necessary files to replicate and execute the Optimized Domain-Adaptive Pre-Training and Knowledge Distillation framework.

## Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (Optional, but recommended for faster inference)

## Setup Instructions
1. Open a terminal/command prompt and navigate to this installer directory.
2. Create a virtual environment:
   python -m venv venv
3. Activate the environment:
   - Windows: venv\Scripts\activate
   - Mac/Linux: source venv/bin/activate
4. Install the required dependencies:
   pip install -r environment/requirements.txt

## How to Run the Evaluation Demo
To test the distilled model using the provided sample dataset, run:
   python source_code/main.py --model_path weights_and_models/distilled_model/ --data sample_dataset/sample_data.csv
