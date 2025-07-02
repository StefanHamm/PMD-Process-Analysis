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
- [x] Sync the welddata with layerdata, this is done by a knn where we fit n points of the mesurement data to the weld data via x,y corrdinate
- [x] Compare lag between change in wirefeed and change in layer height
- [ ] Create a model to predict the layer height and width based on the process parameters
- [ ] Evaluate the model performance
- [ ] Visualize the results

### Dataset 2
- [ ] subtract layer 0 following layers (not sure if needed since we calculate the the deposition hight)
- [ ] Make syncing proces faster by using multiple processing
- [ ] Add a pass indicator to the passes.
- [ ] combine the collected data (individual csv) of Dataset 2 into one dataset