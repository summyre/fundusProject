# Retinal Disease Classification
## Included Files
`train.py` - train the model using the dataset (main file to be run)

`models.py` - defines custom CNN and ResNet-18 getter function

`dataset.py` - holds dataset class and transforms

`functions.py` - any extra functions (e.g. model evaluations - graphs)

`handle_dups` - gets rid of duplicates across class folders

`requirements.txt` - libraries to be installed to run the code

## Environment
Python 3.10

## Setup
# If in restricted Windows
`conda create -n venv python=3.10`

`conda activate venv`

`pip install -r requirements.txt`

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

# If in Linux
`pip install -r requirements.txt`

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
