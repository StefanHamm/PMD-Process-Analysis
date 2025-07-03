
import pandas as pd
from typing import List
import plotly.express as px

def get_column_names(path: str) -> List[str]:
    """
    Get the column names from the CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        List[str]: A list of column names.
    """
    df = pd.read_csv(path, nrows=0)
    return df.columns.tolist()

def load_dataset(path: str, ColumsToRead: List[str], OutputColumnNames: List[str]) -> pd.DataFrame:
    """
    Load the dataset from the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    Args:
        path (str): The path to the CSV file.
        ColumsToRead (List[str]): The columns to read from the CSV file.
        OutputColumnNames (List[str]): The names of the output columns.
    """
    df = pd.read_csv(path)
    # convert the timestampt to pd.to_datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    df = df[ColumsToRead]
    # Rename the columns to the output column names
    df.columns = OutputColumnNames    
    return df

def standar_scale_X_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function scales the the X values of the dataframe to a mean of 0 and a standard deviation of 1.

    Args:
        df (pd.DataFrame): Input dataframe with the X values to be scaled.

    Returns:
        pd.DataFrame: DataFrame with the scaled X values.
    """
    # Bit more complicated first we the unique numbers of the "Layer" column
    unique_layers = df['Layer'].unique()
    # Now we scale the data for each layer on mode 2 and mode 4
    
    df_scaled = df.copy()
    for layer in unique_layers:
        df_scaled.loc[(df_scaled["Layer"] == layer)&(df_scaled["Mode"]==4), "X"] = (df_scaled.loc[(df_scaled["Layer"] == layer)&(df_scaled["Mode"]==4), "X"] - df_scaled.loc[(df_scaled["Layer"] == layer)&(df_scaled["Mode"]==4), "X"].mean()) / df_scaled.loc[(df_scaled["Layer"] == layer)&(df_scaled["Mode"]==4), "X"].std()
        df_scaled.loc[(df_scaled["Layer"] == layer)&(df_scaled["Mode"]==2), "X"] = (df_scaled.loc[(df_scaled["Layer"] == layer)&(df_scaled["Mode"]==2), "X"] - df_scaled.loc[(df_scaled["Layer"] == layer)&(df_scaled["Mode"]==2), "X"].mean()) / df_scaled.loc[(df_scaled["Layer"] == layer)&(df_scaled["Mode"]==2), "X"].std()
        
    return df_scaled

def plot_XY_coords_of_datapoints(df: pd.DataFrame, title: str = "X-Y Coordinates of Data Points") -> None:
    """
    For each layer plot the X and Y coordinates of the data points. Each plot should have the points of Mode 2 and 4, there shoudl be as many plots as layers.
    Args:
        df (pd.DataFrame): The dataframe containing the data points.
        title (str): The title of the plot.
    
    """
    
    #create a scatter plot for each layer 
    #All plots should be aligned in a grid
    fig = px.scatter(df, x='X', y='Y', color='Mode', facet_col='Layer', title=title)
    fig.update_layout(title=title, width=1000, height=800)
    fig.show()

def calculate_measurement_error(df: pd.DataFrame, mode: int, layer: int) -> float:
    """
    Calculates the absolute measurement error in mm for a given mode and layer.
    The error is estimated as 1% of the mean absolute difference between Z and Z-Height.
    
    For DEVICE: OCP662X0135 Laserdistanzsensor Triangulation
    Not accounting for temperature drift!
    
    Args:
        df (pd.DataFrame): Welddata
        mode (int): Mode number (e.g., 4 or 5)
        layer (int): Layer number to calculate error for
    
    Returns:
        float: Estimated absolute measurement error in mm
    """
    subset = df[(df['Mode'] == mode) & (df['Layer'] == layer)]
    errors = (subset['Z'] - subset['Z-Height']).abs()
    mean_error = errors.mean()
    return mean_error * 0.01,max(errors)*0.01,min(errors)*0.01

def calculate_errors_for_all_layers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes measurement errors for all layers in Mode 4 and 5.
    
    Args:
        df (pd.DataFrame): Welddata

    Returns:
        pd.DataFrame: DataFrame with columns ['Mode', 'Layer', 'Error_mm']
    """
    results = []

    for mode in [4, 5]:
        layers = df[df['Mode'] == mode]['Layer'].unique()
        for layer in layers:
            error,maxerror,minerror = calculate_measurement_error(df, mode, layer)
            results.append({'Mode': mode, 'Layer': layer, 'ErrorMean_mm': error,"ErrorMax_mm":maxerror,"ErrorMin_mm":minerror})

    return pd.DataFrame(results)

    
def fix_measureing_outliers_of_y(df:pd.DataFrame) -> pd.DataFrame:
    """The measurement of "Layer"==0 sometimes has measurement errors that deviate more than 2mm from the other data pooints. This function calculates the mean of Mode=4 for "Z-Height" and sets the datapoints that deviate more than 2mm to pd.nan.
    Afterwards the values of get interpolated
    
    Args:
        df (pd.DataFrame): Welddata

    Returns:
        pd.DataFrame: Fixed df for Mode 4
    """
    dfcopy = df.copy()
    # Filter Mode==4
    mode4_df = dfcopy[dfcopy['Mode'] == 4]
    
    # Calculate mean Z-Height for Layer==0 and Mode==4
    layer0_mask = (mode4_df['Layer'] == 0)
    mean_z = mode4_df.loc[layer0_mask, 'Z-Height'].mean()
    
    # Find outliers in Layer==0 with deviation > 2mm
    outliers_mask = layer0_mask & (dfcopy['Mode'] == 4) & (dfcopy['Z-Height'].sub(mean_z).abs() > 2)
    
    # Set outliers to NaN
    dfcopy.loc[outliers_mask, 'Z-Height'] = np.nan
    
    # Interpolate missing values (linear interpolation)
    dfcopy.loc[df['Mode'] == 4, 'Z-Height'] = dfcopy.loc[dfcopy['Mode'] == 4, 'Z-Height'].interpolate()
    
    return dfcopy

def cutoff_start_end_of_axis(df:pd.DataFrame,column="Y",mode=4, cutoff=0.1) -> pd.DataFrame:
    """This fuction cleans the dataset by removing the first and last cutoff percent of the Y values of all layers.

    Args:
        df (pd.DataFrame): welding data dataframe
        cutoff (float, optional): How much gets cut off in the beginning and end. Defaults to 0.1.

    Returns:
        pd.DataFrame: Cleaned dataframe with the Y values cut off.
    """
    
    df = df.copy()
    
    # min and max values of y coordinates for Mode 5 per layer
    def get_interrange(datasetframe):
        df_mode_4 = datasetframe[datasetframe["Mode"] == mode]
        min_y = df_mode_4.groupby("Layer")[column].min().min()
        max_y = df_mode_4.groupby("Layer")[column].max().max()
        # print(min_y)
        # print(max_y)

        interrange = max_y - min_y
        return interrange,min_y,max_y
    
    interrange, min_y, max_y = get_interrange(df)
    df = df[(df[column]> min_y + interrange * cutoff) & (df[column] < max_y - interrange * cutoff)]
    return df

from sklearn.neighbors import BallTree
import numpy as np




def filter_mode_5_crossection_points(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """This function takes a df of welding points and removes data points from layers 
    that are too close to points in any of the previous layers. This is specifically for 
    points where Mode is 5, ensuring we only keep new material added in a given welding pass as datapoints.

    Args:
        df (pd.DataFrame): Weld data DataFrame, must contain 'X', 'Y', 'Z-Height', 'Mode', and 'Layer' columns.
        threshold (float): The distance threshold. A point is removed if its nearest neighbor in a previous layer is within this distance.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the redundant Mode 5 points removed.
    """
    
    features = ["X", "Y", "Z-Height"]
    
    # Create a copy to avoid modifying the original DataFrame
    outputdf = df.copy()
    
    # Isolate the data we need to process
    df_mode5 = df[df["Mode"] == 5].copy()
    if df_mode5.empty:
        # No Mode 5 points to filter, return the original df copy
        return outputdf
        
    layers = df_mode5["Layer"].unique()
    
    # Sort layers to process them in chronological order
    layers = np.sort(layers)
    
    # If there's only one layer or no layers, no filtering is needed
    if len(layers) < 2:
        return outputdf
        
    # --- Initialization ---
    # Start with the points from the very first layer as our initial "valid" point cloud.
    # These points cannot be redundant since there are no previous layers.
    first_layer_df = df_mode5[df_mode5["Layer"] == layers[0]]
    cumulative_points = first_layer_df[features].values
    
    # --- Iteration and Filtering ---
    # Iterate through the subsequent layers
    for layer_num in layers[1:]:
        # If there are no points from previous layers to compare against, skip building the tree
        if cumulative_points.shape[0] == 0:
            # In this case, all points in the current layer are considered new.
            # We'll add them to the cumulative set later.
            pass
        else:
            # Build the BallTree with all valid points from previous layers
            tree = BallTree(cumulative_points, leaf_size=40)
        
        # Get the points for the current layer
        current_layer_df = df_mode5[df_mode5["Layer"] == layer_num]
        if current_layer_df.empty:
            continue
            
        current_points = current_layer_df[features].values
        
        # If there were no previous points, all current points are kept
        if cumulative_points.shape[0] == 0:
            points_to_keep = current_points
            
        else:
            # For each point in the current layer, find the distance to the nearest neighbor in the tree
            # This is a vectorized operation, much faster than a for-loop
            distances, _ = tree.query(current_points, k=1)
            # The result is a 2D array, so we flatten it to a 1D array of distances
            distances = distances.flatten()
            
            # Identify which points to REMOVE (distance <= threshold)
            mask_to_drop = distances <= threshold
            indices_to_drop = current_layer_df.index[mask_to_drop]
            
            # Remove these rows from our main output DataFrame
            if not indices_to_drop.empty:
                outputdf.drop(indices_to_drop, inplace=True)
                
            # Identify which points to KEEP (distance > threshold) to add to our reference point cloud
            mask_to_keep = ~mask_to_drop
            points_to_keep = current_points[mask_to_keep]

        # Update the cumulative point cloud with the new, valid points from this layer
        if points_to_keep.shape[0] > 0:
            if cumulative_points.shape[0] == 0:
                cumulative_points = points_to_keep
            else:
                cumulative_points = np.vstack([cumulative_points, points_to_keep])

    return outputdf

def cut_off_layer_0_mode5(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cuts off the left and right part of layer 0 on the measurement of mode 5.
    If a data point of layer 0 has no data point with a smaller X value in any higher layer (>0), 
    it is removed (left cutoff).
    Similarly, if no data point in higher layers has a greater X value, that layer 0 point is removed (right cutoff).
    
    Args:
        df (pd.DataFrame): Weld dataframe
    
    Returns:
        pd.DataFrame: DataFrame with layer 0 mode 5 data points cut off on left and right edges
    """
    df = df.copy()  # to be safe
    
    # Filter mode 5 data
    mode5_df = df[df['Mode'] == 5]
    
    # Extract X values for higher layers (>0)
    higher_layers_x = mode5_df[mode5_df['Layer'] > 0]['X'].values
    
    # Function to check if point should be kept
    def keep_point(x):
        has_smaller = any(higher_layers_x < x)
        has_greater = any(higher_layers_x > x)
        # Keep point only if it has smaller X points on higher layers (left) and greater X points (right)
        return has_smaller and has_greater
    
    # Apply for layer 0 points of mode 5
    layer0_mask = (df['Mode'] == 5) & (df['Layer'] == 0)
    to_keep = df.loc[layer0_mask, 'X'].apply(keep_point)
    
    # Remove points where keep_point is False
    df = df.drop(df.loc[layer0_mask].index[~to_keep].to_list())
    
    return df

def preprocess_welding_data(df: pd.DataFrame, cutoffy: float = 0.1,cutoffx: float = 0.1, threshold: float = 0.8) -> pd.DataFrame:
    """
    Applies a sequence of preprocessing steps to clean and filter the welding data.
    
    Steps:
        1. Fix outliers in 'Z-Height' for Mode 4, Layer 0.
        2. Cut off start/end of Y-axis for Mode 4.
        3. Cut off start/end of X-axis for Mode 5.
        4. Filter redundant Mode 5 points across layers based on spatial threshold.
        5. Cut off edge points in Mode 5, Layer 0 based on higher layer coverage.
    
    Args:
        df (pd.DataFrame): Raw welding data.
        cutoffy (float): Cutoff percentage for Y-axis trimming. Default is 0.1 (10%).
        cutoffx (float): Cutoff percentage for X-axis trimming. Default is 0.1 (10%).
        threshold (float): Distance threshold for Mode 5 filtering. Default is 0.8 mm.
    
    Returns:
        pd.DataFrame: Preprocessed welding data.
    """
    df = df.copy()
    df = fix_measureing_outliers_of_y(df)
    df = cutoff_start_end_of_axis(df, column="Y", mode=4, cutoff=cutoffy)
    df = cutoff_start_end_of_axis(df,column="X", mode=5, cutoff=cutoffx)
    df = filter_mode_5_crossection_points(df, threshold=threshold)
    df = cut_off_layer_0_mode5(df)
    return df



def plot3D_plotly(df,mode):
    """Plot a 3D scatter plot of the X, Y and Z-Height values of the dataframe for a given mode.

    Args:
        df (pd.df): welding data dataframe
        mode (int): Either 4 or 5
    """
    
    printframe = df.copy()
    printframe["Layer"] = printframe['Layer'].astype(str)
    
    fig = px.scatter_3d(printframe[printframe['Mode']==mode], x='X', y='Y', z='Z-Height',color='Layer')
    fig.show()


def plot2D_crossection(df, mode,colxaxis = "X",colyaxis = "Z-Height",coloring = "Layer"):
    """Creates 2 plot for a specified mode and variable for x and y axis for a given mode.

    Args:
        df (pd.df): Df with weld data
        mode (int): Mode which to display e.g. 2,4,5 
        colxaxis (str, optional): Column to put on the x axis of the plot. Defaults to "X".
        colyaxis (str, optional): Column to put on the y axis of the plot. Defaults to "Z-Height".
        color (str,optional): Column used to color the values.
    """
        
    
    unique_layers = df[df["Mode"] == mode]["Layer"].unique()
    # Filter dataframe for the mode and layers 0 to 8
    filtered_df = df[(df["Mode"] == mode) & (df["Layer"].between(0, max(unique_layers)))].copy()
    filtered_df["Layer"] = filtered_df['Layer'].astype(str)

    fig = px.scatter(
        filtered_df,
        x=colxaxis,
        y=colyaxis,
        color=coloring,
        title=f"Scatter Plot of Layers 0–{max(unique_layers)} (Mode {mode})",
        labels={"X": "X", "Z-Height": "Z-Height"},
        opacity=0.6,
        color_discrete_sequence =px.colors.qualitative.Dark24,
    )
    fig.update_layout(
        legend_title_text='Layer',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        width=800,
        height=500
    )
    fig.show()
    



def process_layer(layer, df_copy, axisToScale, knn, passesPerMeasurement):
    """Process a single layer in a separate process"""
    
    # Get current welding layer (Mode 2)
    layer_mode_2 = df_copy[(df_copy["Mode"] == 2) & (df_copy["Layer"] == layer)]
    
    # Get measurement layers (Mode 4)
    prev_layer_mode_4 = df_copy[(df_copy["Mode"] == 4) & (df_copy["Layer"] == layer - 1)]
    next_layer_mode_4 = df_copy[(df_copy["Mode"] == 4) & (df_copy["Layer"] == layer)]

    # Create KNN models
    prev_knn, next_knn = NearestNeighbors(n_neighbors=knn), NearestNeighbors(n_neighbors=knn)
    
    if prev_layer_mode_4.empty or next_layer_mode_4.empty:
        print(f"No previous or next layer data for Layer {layer}. Skipping height calculation.")
        return pd.DataFrame()  # Return empty dataframe if no data
    
    if not prev_layer_mode_4.empty:
        prev_knn.fit(prev_layer_mode_4[["X", "Y"]].values)
    if not next_layer_mode_4.empty:
        next_knn.fit(next_layer_mode_4[["X", "Y"]].values)

    # Initialize result lists
    results = []
    
    for index in tqdm(layer_mode_2.index, desc=f"Processing Layer {layer}", unit="point"):
        x, y = df_copy.loc[index, ["X", "Y"]].values
        
        # Get previous height
        if not prev_layer_mode_4.empty:
            _, prev_indices = prev_knn.kneighbors([[x, y]])
            prev_neighbors = prev_layer_mode_4.iloc[prev_indices[0]]
            prev_height = prev_neighbors["Z-Height"].mean()
            prev_X_mean = prev_neighbors["X"].mean()
            prev_Y_mean = prev_neighbors["Y"].mean()
        else:
            prev_height = np.nan
            prev_X_mean = np.nan
            prev_Y_mean = np.nan
            
        # Get next height
        if not next_layer_mode_4.empty:
            _, next_indices = next_knn.kneighbors([[x, y]])
            next_neighbors = next_layer_mode_4.iloc[next_indices[0]]
            next_height = next_neighbors["Z-Height"].mean()
            if pd.isna(next_height):
                print(f"NaN next_height for layer {layer}, index {index}")
            next_X_mean = next_neighbors["X"].mean()
            next_Y_mean = next_neighbors["Y"].mean()
        else:
            next_height = np.nan
            next_X_mean = np.nan
            next_Y_mean = np.nan

        # Calculate deposited height
        if pd.notnull(prev_height) and pd.notnull(next_height):
            dep_height = (next_height - prev_height) / passesPerMeasurement
        else:
            dep_height = np.nan
            
        # Store results
        results.append({
            'index': index,
            'prev_height': prev_height,
            'next_height': next_height,
            'dep_height': dep_height,
            'prev_X_mean': prev_X_mean,
            'prev_Y_mean': prev_Y_mean,
            'next_X_mean': next_X_mean,
            'next_Y_mean': next_Y_mean
        })
    
    return results

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def sync_measurement_with_weld_mp(df: pd.DataFrame, meanscaleAxis=True, axisToScale="X", 
                                  knn=3, passesPerMeasurement=1, n_processes=None):
    """
    Multiprocessed version of sync_measurement_with_weld function.
    
    Args:
        df (pd.DataFrame): The dataframe with weld data
        meanscaleAxis (bool, optional): The tool head moves along one axis. This tells it to mean scale it for Mode 2 and Mode 4 to be overlay it better. Defaults to True.
        axisToScale (str, optional): Tells which axis to mean scale. Defaults to "X".
        knn (int, optional): Number of neighbours used to map Z-Height value. Defaults to 3
        passesPerMeasurement (int, optional): How many weld passes were performed till measurement? This divides the deposited layerheight by this value.
        n_processes (int, optional): Number of processes to use. If None, uses CPU count.
    """
    
    weldAxis = "X"
    if axisToScale == "X":
        weldAxis = "Y"
        
    df_copy = df.copy()
    
    unique_layers = df[df["Mode"] == 4]["Layer"].unique()
    
    if meanscaleAxis:
        for layer in unique_layers:
            # normalize X and Y coordinates by standard scaling it with a mean of 0 and a standard deviation of 1
            layer_mode_4_mask = (df_copy["Layer"] == layer) & (df_copy["Mode"] == 4)
            layer_mode_2_mask = (df_copy["Layer"] == layer) & (df_copy["Mode"] == 2)
            
            if layer_mode_4_mask.any():
                mode_4_data = df_copy.loc[layer_mode_4_mask, axisToScale]
                df_copy.loc[layer_mode_4_mask, axisToScale] = (mode_4_data - mode_4_data.mean()) / mode_4_data.std()
            
            if layer_mode_2_mask.any():
                mode_2_data = df_copy.loc[layer_mode_2_mask, axisToScale]
                df_copy.loc[layer_mode_2_mask, axisToScale] = (mode_2_data - mode_2_data.mean()) / mode_2_data.std()
    
    # Initialize new columns
    df_copy['prev_height'] = np.nan
    df_copy['next_height'] = np.nan
    df_copy['dep_height'] = np.nan
    df_copy['prev_X_mean'] = np.nan  
    df_copy['prev_Y_mean'] = np.nan  
    df_copy['next_X_mean'] = np.nan  
    df_copy['next_Y_mean'] = np.nan
    
    unique_layers.sort()
    
    # Set up multiprocessing
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"Using {n_processes} processes for layer processing")
    
    # Process layers starting from layer 1 (skip layer 0 as it needs previous layer)
    layers_to_process = unique_layers[1:]
    
    if len(layers_to_process) == 0:
        print("No layers to process (need at least 2 layers)")
        return df_copy[df_copy["Mode"] == 2].copy()
    
    # Create partial function with fixed parameters
    process_layer_partial = partial(
        process_layer,
        df_copy=df_copy,
        axisToScale=axisToScale,
        knn=knn,
        passesPerMeasurement=passesPerMeasurement
    )
    
    # Process layers in parallel
    with mp.Pool(processes=n_processes) as pool:
        all_results = pool.map(process_layer_partial, layers_to_process)
    
    # Combine results back into dataframe
    for layer_results in all_results:
        if layer_results:  # Check if results is not empty
            for result in layer_results:
                index = result['index']
                df_copy.loc[index, [
                    'prev_height', 'next_height', 'dep_height',
                    'prev_X_mean', 'prev_Y_mean',
                    'next_X_mean', 'next_Y_mean'
                ]] = [
                    result['prev_height'], result['next_height'], result['dep_height'],
                    result['prev_X_mean'], result['prev_Y_mean'],
                    result['next_X_mean'], result['next_Y_mean']
                ]
    
    return df_copy[df_copy["Mode"] == 2].copy()


from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def sync_measurement_with_weld(df:pd.DataFrame,meanscaleAxis = True, axisToScale = "X", knn = 3,passesPerMeasurement = 1):
    """This fucntion synchronizes the measurement data of Mode = 4 with the weld data of Mode 2. Other Modes get lost!

    Args:
        df (pd.DataFrame): The dataframe with weld data
        meanscaleAxis (bool, optional): The tool head moves along one axis. This tells it to mean scale it for Mode 2 and Mode 4 to be overlay it better. Defaults to True.
        axisToScale (str, optional): Tells which axis to mean scale. Defaults to "X".
        knn (int,optional): Number of neigbours used to map Z-Height value. Defaults to 3
        passesPerMeasurement (int,optional): How many weld passes were performed till measurement? This divides the deposited layerheight by this value.
    """
    weldAxis = "X"
    if axisToScale == "X":
        weldAxis = "Y"
        
    df_copy = df.copy()
    
    unique_layers = unique_layers = df[df["Mode"] == 4]["Layer"].unique()
    
    if meanscaleAxis:
        for layer in unique_layers:
            # normalize X and Y coordinates by standard scaling it with a mean of 0 and a standard deviation of 1
            df_copy.loc[(df_copy["Layer"] == layer)&(df_copy["Mode"]==4), axisToScale] = (df_copy.loc[(df_copy["Layer"] == layer)&(df_copy["Mode"]==4), axisToScale] - df_copy.loc[(df_copy["Layer"] == layer)&(df_copy["Mode"]==4), axisToScale].mean()) / df_copy.loc[(df_copy["Layer"] == layer)&(df_copy["Mode"]==4), axisToScale].std()
            #df_cleaned_copy.loc[(df_cleaned_copy["Layer"] == layer)&(df_cleaned_copy["Mode"]==4), "Y"] = (df_cleaned_copy.loc[(df_cleaned_copy["Layer"] == layer)&(df_cleaned_copy["Mode"]==4), "Y"] - df_cleaned_copy.loc[(df_cleaned_copy["Layer"] == 0)&(df_cleaned_copy["Mode"]==4), "Y"].mean()) / df_cleaned_copy.loc[(df_cleaned_copy["Layer"] == 0)&(df_cleaned_copy["Mode"]==4), "Y"].std()
            
            df_copy.loc[(df_copy["Layer"] == layer)&(df_copy["Mode"]==2), axisToScale] = (df_copy.loc[(df_copy["Layer"] == layer)&(df_copy["Mode"]==2), axisToScale] - df_copy.loc[(df_copy["Layer"] == layer)&(df_copy["Mode"]==2), axisToScale].mean()) / df_copy.loc[(df_copy["Layer"] == layer)&(df_copy["Mode"]==2), axisToScale].std()
    
    df_copy['prev_height'] = np.nan
    df_copy['next_height'] = np.nan
    df_copy['dep_height'] = np.nan
    df_copy['prev_X_mean'] = np.nan  
    df_copy['prev_Y_mean'] = np.nan  
    df_copy['next_X_mean'] = np.nan  
    df_copy['next_Y_mean'] = np.nan
    
    unique_layers.sort()
    
    
    for layer in unique_layers[1:]:
        # Get current welding layer (Mode 2)
        layer_mode_2 = df_copy[(df_copy["Mode"] == 2) & (df_copy["Layer"] == layer)]
        
        # Get measurement layers (Mode 4)
        prev_layer_mode_4 = df_copy[(df_copy["Mode"] == 4) & (df_copy["Layer"] == layer - 1)]
        next_layer_mode_4 = df_copy[(df_copy["Mode"] == 4) & (df_copy["Layer"] == layer)]

        # Create KNN models
        prev_knn, next_knn = NearestNeighbors(n_neighbors=knn), NearestNeighbors(n_neighbors=knn)
        if prev_layer_mode_4.empty or next_layer_mode_4.empty:
            print(f"No previous or next layer data for Layer {layer}. Skipping height calculation.")
        
        
        if not prev_layer_mode_4.empty:
            prev_knn.fit(prev_layer_mode_4[["X", "Y"]].values)
        if not next_layer_mode_4.empty:
            next_knn.fit(next_layer_mode_4[["X", "Y"]].values)

        for index in tqdm(layer_mode_2.index, desc=f"Processing Layer {layer}", unit="point"):
            x, y = df_copy.loc[index, ["X", "Y"]].values
            
            # Get previous height
            if not prev_layer_mode_4.empty:
                _, prev_indices = prev_knn.kneighbors([[x, y]])
                prev_neighbors = prev_layer_mode_4.iloc[prev_indices[0]]
                prev_height = prev_neighbors["Z-Height"].mean()
                prev_X_mean = prev_neighbors["X"].mean()  # Mean X of neighbors
                prev_Y_mean = prev_neighbors["Y"].mean()  # Mean Y of neighbors
            else:
                print(f"No previous layer data for Layer {layer - 1}. Skipping height calculation for index {index}.")
                prev_height = np.nan
                
            # Get next height
            if not next_layer_mode_4.empty:
                _, next_indices = next_knn.kneighbors([[x, y]])
                
                next_neighbors = next_layer_mode_4.iloc[next_indices[0]]
                next_height = next_neighbors["Z-Height"].mean()
                if next_height is pd.NA:
                    print(next_neighbors["Z-Height"])
                next_X_mean = next_neighbors["X"].mean()  # Mean X of neighbors
                next_Y_mean = next_neighbors["Y"].mean()  # Mean Y of neighbors
            else:
                next_height = np.nan

            # Update dataframe
            df_copy.loc[index, [
                'prev_height', 'next_height',
                'prev_X_mean', 'prev_Y_mean',
                'next_X_mean', 'next_Y_mean'
            ]] = [
                prev_height, next_height,
                prev_X_mean, prev_Y_mean,
                next_X_mean, next_Y_mean
            ]
            
            # Calculate deposited height
            if pd.notnull(prev_height) and pd.notnull(next_height):
                df_copy.loc[index, 'dep_height'] = (next_height - prev_height)/passesPerMeasurement
            else:
                print(f"Missing height data for index {index}. Setting deposited height to NaN.")
                df_copy.loc[index, 'dep_height'] = np.nan
    
    return df_copy[df_copy["Mode"]==2].copy()

from sklearn.cluster import KMeans
def add_pass_indicator(df: pd.DataFrame, columnName="Pass", numberOfPasses=2):
    """
    Adds a column indicating the pass number (1 or 2) within each layer,
    based on clustering of Z heights (assuming 2 passes per layer).
    Args:
        df (pd.DataFrame): DataFrame with columns 'Layer' and Z-height (e.g. 'MD/LSZ')
        columnName (str, optional): Name of the new column. Defaults to "Pass".
        numberOfPasses (int,optional): How many passes per layer are done
    """
    dfcopy = df.copy()
    dfcopy[columnName] = 1  # Default pass
   
    for layer in dfcopy['Layer'].unique():
        mask = dfcopy['Layer'] == layer
        z_values = dfcopy.loc[mask, 'Z'].values.reshape(-1, 1)
       
        if len(z_values) <= 1:
            continue  # Not enough data to cluster
       
        # Only apply clustering for Mode == 2 (assuming Mode column exists)

        mode_mask = dfcopy['Mode'] == 2
        layer_mode_mask = mask & mode_mask
        
        if not layer_mode_mask.any():
            continue  # No Mode == 2 data in this layer
        
        z_values_mode2 = dfcopy.loc[layer_mode_mask, 'Z'].values.reshape(-1, 1)
        
        if len(z_values_mode2) <= 1:
            continue  # Not enough Mode == 2 data to cluster
        
        # Cluster into two passes based on Z height for Mode == 2 data
        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
        labels = kmeans.fit_predict(z_values_mode2)
        centers = kmeans.cluster_centers_.flatten()
        
        # Create proper mapping: lower Z center gets pass 1, higher Z center gets pass 2
        sorted_center_indices = np.argsort(centers)
        label_to_pass = {}
        for i, center_idx in enumerate(sorted_center_indices):
            # Find which cluster label corresponds to this center
            cluster_label = center_idx
            label_to_pass[cluster_label] = i + 1
        
        # Apply pass numbers to Mode == 2 data
        pass_values = [label_to_pass[label] for label in labels]
        dfcopy.loc[layer_mode_mask, columnName] = pass_values

   
    return dfcopy