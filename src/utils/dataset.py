import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .graph import HitGraph
from .pandas import (is_spot_scanning, open_dataframe_by_extension, preprocess)


class HitGraphDataset:
    def __init__(self, directory: str, device: torch.device = "cpu") -> None:
        """Initialize a HitGraphDataset for loading .graph files from a given directory.

        Args:
            directory (str): Directory containing serialized HitGraph files.
            device (torch.device): Device to map the loaded HitGraphs to.
        """
        self.n = 0
        self.directory = directory
        self.device = device
        self.files = [f for f in os.listdir(directory) if f.endswith('.graph')]
        
    def sample(self) -> HitGraph:
        index = int(np.random.choice(len(self.files), 1))
        return self.__load_graph(index)
    
    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n < len(self.files):
            env = self.__load_graph(self.n)
            self.n += 1
            return env
        else:
            raise StopIteration
        
    def __load_graph(self, index: int) -> HitGraph:
        G: HitGraph = torch.load(os.path.join(self.directory, self.files[index]), map_location=self.device)
        G.device = self.device
        return G
    
    def load_graph(self, index):
        G: HitGraph = torch.load(os.path.join(self.directory, self.files[index]), map_location=self.device)
        G.device = self.device
        return G
    
    
    @classmethod
    def __try_create(cls, path: str):
        if not os.path.isdir(path):
            os.makedirs(path)
                
    @classmethod
    def __export_to_file(cls, df: pd.DataFrame, splits: List[np.ndarray], 
                         db_dir: str, skip_tracker: bool, cluster_threshold: float, 
                         device: str, offset: int = None):
        """Internal method to export a list of graphs from event splits.

        Args:
            df (pd.DataFrame): Full dataframe of particle hits.
            splits (List[np.ndarray]): List of eventID splits, each for one graph.
            db_dir (str): Directory where graphs will be saved.
            skip_tracker (bool): Whether to exclude tracker layers.
            cluster_threshold (float): Threshold for clustering hits.
            device (str): Device to assign the graphs.
            offset (int, optional): Offset for file naming.
        """
        for i, split in enumerate(splits):
            d = df.loc[df.eventID.isin(split)]
            G = HitGraph.from_df(d, skip_tracker=skip_tracker,
                                   to_wet=False, filter_secondaries=False,
                                   cluster_threshold=cluster_threshold, device=device)
            torch.save(G, os.path.join(db_dir, f"{i if offset is None else i + offset}.graph"))
            
            
            
    @classmethod
    def from_file(cls, file_path: str, db_dir_join: str, events_per_env: int = 100, 
                  primaries_per_spot = None,  skip_tracker: bool = False, cluster_threshold: float = 2, 
                  device: str = "cpu", offset: int = 0) -> int:
        """Creates a partial dataset containing multiple environments from a single pandas/numpy input file
        containing raw simulated data (detector readouts).

        Args:
            file_path (str): Path to input dataframe/numpy file.
            db_dir_join (str): Directory where graphs will be stored.
            events_per_env (int, optional): Number of events per environment. Defaults to 100. Defaults to 100.
            primaries_per_spot (_type_, optional): Number of primaries per beam spot position. Defaults to None.
            skip_tracker (bool, optional): Exclude tracking layers (for debugging and testing). Defaults to False.
            cluster_threshold (float, optional): Cluster threshold for secondary filtering. Defaults to 2. Defaults to 2.
            device (str, optional): Computational device where dataset should be crated. Defaults to "cpu".
            offset (int, optional): Integer offset for numbering the dataset files. Defaults to 0.

        Returns:
            int: New index offset based on the number of created files.
        """
        df = open_dataframe_by_extension(file_path)
        df = preprocess(df)
        
        cls.__try_create(db_dir_join)
        
        if not is_spot_scanning(df):
            df["spotX"] = 0
            df["spotY"] = 0
            
        for _, spot in df[["eventID", "spotX", "spotY"]].groupby(["spotX", "spotY"]):
            events = spot.groupby("eventID").size().keys().astype(int)
            event_groups = [events[i*events_per_env:(i+1)*events_per_env]  \
                            for i in range(len(events)//events_per_env)\
                            if (i+1)*events_per_env <= len(spot)]
            
            if primaries_per_spot is not None:
                assert events_per_env <= primaries_per_spot
                event_groups_selection = primaries_per_spot//events_per_env
                event_groups = event_groups[:event_groups_selection]
                
            cls.__export_to_file(df, event_groups, db_dir_join, skip_tracker, 
                                 cluster_threshold, device, offset=offset)
            offset += len(event_groups)
        
        return offset
    
    @classmethod
    def from_files(cls, file_paths: List[str], db_names: Union[str, List[str]], prefix: Optional[str], db_dir: str, events_per_env: int = 100, 
                        primaries_per_spot = None, skip_tracker: bool = False, cluster_threshold: float = 2, 
                        device: str = "cpu"):
        """Creates a dataset containing hit graphs from multiple pandas/numpy input files 
        containing raw simulated data (detector readouts). Data requires manual separation 
        of train/test data --> should be independent of each other.

        Args:
            file_paths (List[str]): List of file paths containing all data.
                        db_names (Union[str, List[str]]): Dataset name(s) for subdirectory naming.
            prefix (Optional[str]): Optional prefix for subdirectories.
            db_dir (str): Root directory to store all environments.
            events_per_env (int, optional): Number of events per environment. Defaults to 100.
            primaries_per_spot (int, optional): Number of primaries per spot. Defaults to None.
            skip_tracker (bool, optional): Exclude tracker layers. Defaults to False.
            cluster_threshold (float, optional): Cluster threshold for secondary filtering. Defaults to 2.
            device (str, optional): Device for computation. Defaults to "cpu".
        """
        offset = 0
        if isinstance(db_names, str):
            db_names = [db_names] * len(file_paths)

        for file_path, name in zip(file_paths, db_names):
            db_dir_join = f"{db_dir}/{prefix}_{name}/{events_per_env}"
            offset = cls.from_file(file_path, db_dir_join, events_per_env, primaries_per_spot, 
                                   skip_tracker, cluster_threshold, device, offset)
                    