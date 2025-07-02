# PMD-Process-Analysis
This repository contains the analysis and optimization of the Plasma Metal Deposition (PMD) process. It focuses on modeling the relationship between process parameters (wire feed speed, current, voltage) and deposition outcomes (layer height, width). The goal is to improve process stability and efficiency using data-driven techniques.

## Installation
To install the required packages, run the following command:

```bash
conda env create -f conda_env.yaml

```

## TODO list what needs to be done


### Dataset 1
- [x] Load the dataset
- [x] Sync the welddata with layerdata
- [x] Compare lag between change in wirefeed and change in layer height
- [ ] Create a model to predict the layer height and width based on the process parameters
- [ ] Evaluate the model performance
- [ ] Visualize the results

### Dataset 2