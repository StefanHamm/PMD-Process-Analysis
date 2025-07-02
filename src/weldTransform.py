
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

def cutoff_start_end_y(df:pd.DataFrame, cutoff=0.1) -> pd.DataFrame:
    """This fuction cleans the dataset by removing the first and last cutoff percent of the Y values of all layers.

    Args:
        df (pd.DataFrame): welding data dataframe
        cutoff (float, optional): How much gets cut off in the beginning and end. Defaults to 0.1.

    Returns:
        pd.DataFrame: Cleaned dataframe with the Y values cut off.
    """
    
    # min and max values of y coordinates for Mode 5 per layer
    def get_interrange(datasetframe):
        df_mode_4 = datasetframe[datasetframe["Mode"] == 4]
        min_y = df_mode_4.groupby("Layer")["Y"].min().min()
        max_y = df_mode_4.groupby("Layer")["Y"].max().max()
        print(min_y)
        print(max_y)

        interrange = max_y - min_y
        return interrange,min_y,max_y
    
    interrange, min_y, max_y = get_interrange(df)
    df = df[(df["Y"]> min_y + interrange * cutoff) & (df["Y"] < max_y - interrange * cutoff)]
    return df

from sklearn.neighbors import BallTree
import numpy as np


def filter_mode_5_crossection_points(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """This function takes a df of welding points and removes data points from layers 
    that are too close to points in any of the previous layers. This is specifically for 
    points where Mode is 5, ensuring we only keep new material added in a given welding pass.

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


def plot2D_crossection(df, mode):
    """Plot a 2D scatter plot (X vs Z-Height) for layers 0 to 8 for a given mode using Plotly."""
    
    unique_layers = df[df["Mode"] == mode]["Layer"].unique()
    # Filter dataframe for the mode and layers 0 to 8
    filtered_df = df[(df["Mode"] == mode) & (df["Layer"].between(0, max(unique_layers)))].copy()
    filtered_df["Layer"] = filtered_df['Layer'].astype(str)

    fig = px.scatter(
        filtered_df,
        x='X',
        y='Z-Height',
        color='Layer',
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
    
    