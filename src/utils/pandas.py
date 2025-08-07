import os
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd

from .detector import digitize

pd.options.mode.chained_assignment = None

format_lookup = {".csv": pd.read_csv,
                 ".pkl": pd.read_pickle}

class GateVersion(Enum):
    v91 = 0
    v92 = 1
    
VERSION: GateVersion = GateVersion.v92

def set_gate_version(version: GateVersion):
    VERSION = version


def open_dataframe_by_extension(path: str) ->  pd.DataFrame:
    """Load a tabular dataset into a pandas DataFrame based on file extension.
    Supported formats: .csv, .pkl, .npy, .npz

    Args:
        path (str): Path to the particle readout file file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.

    Raises:
        KeyError: If the file extension is not supported.
    """
    _, ext = os.path.splitext(path)
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".pkl":
        return pd.read_pickle(path)
    elif ext == ".npy":
        return pd.DataFrame(np.load(path))
    elif ext == ".npz":
        return pd.DataFrame(np.load(path)["arr_0"])
    else:
        raise KeyError(f"Invalid file extension {ext}")


def split_layers_z(df: pd.DataFrame, gate_version: GateVersion = VERSION):
    """
    Compute the layer index (`z`) for each hit based on geometry and Gate version.

    Args:
        df (pd.DataFrame): Input DataFrame containing hit information.
        gate_version (GateVersion, optional): GATE version used for simulation.

    Returns:
        pd.DataFrame: DataFrame with new 'z' column added.
    """
    if gate_version == GateVersion.v91:
        rear_scanner_layer_count = {i: 
                len(df[df["baseID"] < i].groupby(["level1ID", "baseID"])) \
                for i in range(0,df.baseID.max() + 1)}
        
        apply_layer_z = lambda x: rear_scanner_layer_count[x.baseID] + x.level1ID
        df["z"] = df.apply(apply_layer_z, axis=1)
    else:
        if "level2ID" in df.columns and "level1ID" in df.columns:
            df["z"] = (df["level2ID"] != -1) * (2 + df["level2ID"]) + \
                      (df["level2ID"] == -1) *  df["level1ID"]
        else:
            df["z"] = df["volumeID[2]"] * 2 + df["volumeID[3]"]
            
    return df


def merge_duplicate_pixels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate pixel hits originating from same particle (eventID, trackID, z) ) by
    summing energy deposition and averaging position.

    Args:
        df (pd.DataFrame): Input DataFrame of pixel hits.

    Returns:
        pd.DataFrame: Deduplicated DataFrame with merged hits.
    """
    df = digitize(df, sub_pixel_resolution=1)
    df = split_layers_z(df)
    aggregation_dict = dict.fromkeys(df, 'first')
    aggregation_dict.update({'edep': 'sum',
                             'posX': 'mean',
                             'posY': 'mean',
                             'posZ': 'mean'})
    
    return df.groupby(["eventID", "trackID", "z"], as_index=False) \
            .agg(aggregation_dict) \
            .reset_index(drop=True)


def remove_disconnected(df: pd.DataFrame, allowed_skips=0):
    """Removes secondary particles from last layers, where the consecutive
    layers are not directly connected or more than a limited amount of layers
    are skipped.

    Args:
        df (pd.DataFrame): Input DataFrame of hits.
        allowed_skips (int, optional): Number of layers that are allowed to be skipped. Defaults to 0.

    Returns:
        pd.DataFrame: Filtered DataFrame with disconnected layers removed.
    """
    unique_z = np.sort(df.z.unique())
    skip = (unique_z[1:] - unique_z[:-1] - 1) > allowed_skips
    occ = np.where(skip)[0]
    if len(occ) == 0: return df
    mask = ~(unique_z < occ[0])
    df = df[~df.z.isin(unique_z[mask])]

    return df


def edep_to_cluster_size(edep: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return (4.2267 * (edep * 1000./25.) ** 0.65 + 0.5)


def append_cluster_size(df: pd.DataFrame) -> pd.DataFrame:
    """Determines cluster size in pixels for a given energy deposition per thickness of the epitaxial layer
    and appends it to a given dataframe

    cf. H. Pettersen (https://github.com/HelgeEgil/DigitalTrackingCalorimeterToolkit)

    Args:
        df (pd.DataFrame): Dataframe containing the detector data with energy deposition (edep) by epitaxial layer MeV

    Returns:
        pd.DataFrame: Dataframe with cluster size in number of pixels (column: cluster)
    """
    df["cluster"] = edep_to_cluster_size(df.edep).astype(int)
    return df


def generate_valid_tracks_by_graph_idx(df: pd.DataFrame) -> List[List[int]]:
    """Generate a list of trajectories containing the indices of valid tracks generated 
    from the ground truth of the monte carlo simulation.

    Args:
        df (pd.DataFrame): Input DataFrame of hits.

    Returns:
        List[List[int]]: List of particle tracks (each as a list of graph indices).
    """
    assert "z" in df.columns, "Data frame must contain z column"
    assert "eventID" in df.columns, "Data frame must contain eventID column"
    
    df_sorted = df.sort_values("z", ascending=True)
    df_sorted["idx"] = np.arange(len(df_sorted))
    df_tracks = df_sorted.groupby("eventID")
    return list(df_tracks.apply(lambda x: list(x.idx)))


def preprocess(df: pd.DataFrame):
    """
    Run full preprocessing pipeline to prepare data for HitGraph construction.

    Pipeline includes:
      - Digitization
      - Layer (z) calculation
      - Disconnected layer removal
      - Pixel merging
      - Cluster size estimation

    Args:
        df (pd.DataFrame): Input DataFrame of hits.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = digitize(df, sub_pixel_resolution=1)
    df = split_layers_z(df)
    df = remove_disconnected(df)
    df = merge_duplicate_pixels(df)
    df = append_cluster_size(df)
    df.attrs["preprocessed"] = True #FIXME: attr is experimental and may change in a future version
    return df


def is_preprocessed(df: pd.DataFrame):
    return df.attrs.get("preprocessed") == True

def is_spot_scanning(df: pd.DataFrame):
    return "spotX" in df.columns and "spotY" in df.columns