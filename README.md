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
- [x] Compare lag between change in wirefeed and change in layer height using pearson correlation
- [ ] Create a model to predict the layer height and width based on the process parameters
- [ ] Evaluate the model performance
- [ ] Visualize the results

### Dataset 2
- [ ] subtract layer 0 following layers (not sure if needed since we calculate the the deposition hight)
- [x] fix layer 0 outliers if deviates more than 1mm from mean set to layer mean
- [x] Preprocess Mode 4 and 5 by cutoff ends and remove datapoints if they are already recorded by previous layers
- [x] Make syncing proces faster by using multiple processing
- [x] Add a pass indicator to the passes.
- [ ] combine the collected data (individual csv) of Dataset 2 into one dataframe
- [ ] Fix the pass indicator and layer height calculation for one dataset where there is only one layer pass
- [ ] Create a model to predict layer height via input parameters (layer,voltage,Current)
- [ ] Evaluate model performance