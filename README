# VGGFace2 Database and Evaluation

## Overview
This project focuses on building a facial recognition database using the VGGFace2 dataset and evaluating its performance. The steps below will guide you through setting up the project, building the vector database, and evaluating the model.

## Prerequisites
- Python 3.7+
- Required Python packages (listed in `requirements.txt`)

## Setup
### Installation
1. Clone the repository.
2. Navigate to the project directory.
3. Install the required Python packages using the following command:
   ```bash
   pip install -r requirements.txt
   ```

### Directory Structure
Ensure your directory structure follows this format:
```
project/
│
├── vector_db.py
├── recognize.py
├── requirements.txt
├── paths1.pkl
├── faiss_index1.bin
└── VGG-Face2/
    ├── data/
        ├── test/
        └── train/
```

## Usage
### Step 1: Build the Vector Database
Run the `vector_db.py` script to extract features and build the Faiss index:
```bash
python vector_db.py
```

### Step 2: Evaluate the Model
Run the `recognize.py` script to evaluate the model using the created database:
```bash
python recognize.py
```

## Notes
- Ensure that the VGGFace2 dataset is correctly placed in the `VGG-Face2/data/` directory with `test` and `train` subdirectories.
- Adjust the paths in the scripts if your directory structure is different.

## GITHUB URL https://github.com/Morgan66666/project_cv