import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import os
import fcntl
import jsons
import random
import pathlib
from tqdm import tqdm
from deepdiff import DeepDiff
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Generator

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.eval import get_metrics_for_dataset, subset
from src.supervised.gnn.models import HGInteractionNetwork, LGInteractionNetwork
from src.supervised.utils.losses import hamming_loss
from src.utils.transforms.graph import CustomGraphTransform
from src.supervised.combinatorial.lsa import CombinatorialSolver
from src.supervised.combinatorial.margins import cost_margin
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .dataset import HitGraphDataset

def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    torch.use_deterministic_algorithms(True, warn_only=False)


def delete_test_info(json: dict):
    json = jsons.dumps(json) #FIXME:
    json = jsons.loads(json)
    
    keys = ["directory", "test_events", "test_files"]
    for key in keys:
        del json["dataset"][key]

    return json


class LockDirectory():
    def __init__(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory

    def __enter__(self):
        self.dir_fd = os.open(self.directory, os.O_RDONLY)
        try:
            fcntl.flock(self.dir_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError as ex:             
            raise Exception(f"Another training instance is already running in dir {self.directory} - quitting.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # fcntl.flock(self.dir_fd,fcntl.LOCK_UN)
        os.close(self.dir_fd)


class ConfigBase(ABC):
    def __init__(self) -> None:
        pass
    
    def to_json(self, del_test_info: bool = False, no_jdkwargs: bool = False):
        d = self.__dict__
        if del_test_info: d = delete_test_info(d)
        jdkwargs =  None if no_jdkwargs else {"indent": 4}
        return jsons.dumps(d, jdkwargs=jdkwargs)
    
    def to_file(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_json(cls, data: str, to_object: bool = True):
        if to_object:
            return jsons.loads(data, cls)
        else:
            return jsons.loads(data)
    
    @classmethod
    def from_file(cls, path: str, to_object: bool = True):
        with open(path, "r") as f:
            return cls.from_json(f.read(), to_object)


class GraphTransfromConfig(ConfigBase):
    def __init__(self, 
                 stage1_theta_dd: float, 
                 stage1_theta_dt: float,
                 d_encoding: int, scaling: float,
                 to_line_graph: bool) -> None:
        
        super().__init__()
        self.stage1_theta_dd = stage1_theta_dd
        self.stage1_theta_dt = stage1_theta_dt
        self.d_encoding = d_encoding
        self.scaling = scaling
        self.to_line_graph = to_line_graph

    def generate_from_config(self):
        return CustomGraphTransform(**self.__dict__)

    
class DatasetConfig(ConfigBase):
    def __init__(self, directory: str, train_files: List[str], 
                 validation_files: List[str], test_files: Union[Dict[str, str], None],
                 train_events: int, validation_events: int, test_events: Union[List[int], None], 
                 filter_secondaries: bool = False, skip_tracker: bool = False, cluster_threshold: int = 2) -> None:
        
        super().__init__()
        self.directory = directory
        self.train_files = train_files
        self.validation_files = validation_files
        self.test_files = test_files
        self.train_events = train_events
        self.test_events = test_events
        self.validation_events = validation_events
        self.filter_secondaries = filter_secondaries
        self.skip_tracker = skip_tracker
        self.cluster_threshold = cluster_threshold
        
    def _get_config_dir(self):
        return os.path.join(self.directory, "dataset.json")

    def exists(self):
        f = self._get_config_dir()
        
        if not os.path.exists(self.directory) or \
           not os.path.exists(f) :
            return False, False
    
        try: comp = DatasetConfig.from_file(f)
        except: return False, False
        
        return True, comp.__dict__ == self.__dict__
    
         
    def generate_from_config(self, device: str = "cpu"):
        exists, matches = self.exists()
        
        if exists and matches:
            print("Skipping dataset creation..."); return
        elif exists and not matches:
            raise ValueError("Existing dataset does not match the dataset specified in config.")
        
        print("Generating datasets from config...")
        train_dir_join = f"{self.directory}/train_{self.train_events}"
        validation_dir_join = f"{self.directory}/val_{self.train_events}"


        offset = 0
        for train_file in self.train_files:
            offset += HitGraphDataset.from_file(train_file, train_dir_join, 
                                                self.train_events, skip_tracker=self.skip_tracker, 
                                                cluster_threshold=self.cluster_threshold, device=device, offset=offset)
            
        offset = 0
        for validation_file in self.validation_files:
            offset += HitGraphDataset.from_file(validation_file, validation_dir_join, 
                                                self.validation_events, skip_tracker=self.skip_tracker, 
                                                cluster_threshold=self.cluster_threshold, device=device, offset=offset)
        
        if self.test_files != None:
            for events in tqdm(self.test_events):
                HitGraphDataset.from_files(self.test_files.values(), self.test_files.keys(), "test", self.directory, 
                                                    events, skip_tracker=self.skip_tracker, cluster_threshold=self.cluster_threshold, 
                                                    device=device)
            
        with open(self._get_config_dir(), "w") as f:
            f.write(self.to_json())
            
    
    def load_train_from_config(self, device: str = "cpu"):
        train_dir_join = f"{self.directory}/train_{self.train_events}"
        return HitGraphDataset(train_dir_join, device=device)
    
    def load_validation_from_config(self, device: str = "cpu"):
        val_dir_join = f"{self.directory}/val_{self.validation_events}"
        return HitGraphDataset(val_dir_join, device=device)
    
    def iterate_events(self):
        for events in self.test_events:
            yield events
            
    def iterate_phantoms(self):
        for phantom in self.test_files.keys():
            yield phantom
    
    def load_test_from_config(self, phantom, events, device: str = "cpu"):
        dir_join = f"{self.directory}/test_{phantom}/{events}"
        dataset = HitGraphDataset(dir_join, device=device)
        return dataset

class CheckpointConfig:
    def __init__(self, checkpoint_dir: str, checkpoint_freq: int) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq        
                
class ModelConfig(ConfigBase):
    def __init__(self, model_dir: str, node_features: int, edge_features: int, 
                 hidden_size: int, line_graph: bool, checkpoints: Optional[CheckpointConfig]) -> None:
        super().__init__()
        self.line_graph = line_graph
        self.model_dir = model_dir
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_size = hidden_size
        self.checkpoints = checkpoints
        
    def generate_from_config(self, device: str) -> nn.Module:
        line_graph = self.line_graph

        return LGInteractionNetwork(**self.__dict__).to(device) if line_graph else \
               HGInteractionNetwork(**self.__dict__).to(device) 

    
    def load_run_logs(self):
        return pd.read_csv(os.path.join(self.model_dir, f"run_logs.csv"))
    
    def __get_best_checkpoint_index(self, run: int):
        event_acc = EventAccumulator(self.load_run_logs().runs.iloc[run])
        event_acc.Reload()
        val_perf = pd.DataFrame(event_acc.Scalars("Performance/pur_val"))["value"]
        return val_perf.argmax() * 100
    
    def __save_model_to_dir(self, model: Union[LGInteractionNetwork, HGInteractionNetwork], model_dir: str, step: int, run: int):
        model_dir = model_dir if step == None else os.path.join(model_dir, f"run_{run}")
        step_txt = step if step != None else ""
        save_dir = os.path.join(model_dir, f"interaction_net_{step_txt}_{run}.pt")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), save_dir)

    
    def __load_model_from_dir(self, model_dir, step: Optional[int], run: int, device: str) -> Union[LGInteractionNetwork, HGInteractionNetwork]:
        model_dir = model_dir if step == None else os.path.join(model_dir, f"run_{run}")
        step_txt = step if step != None else ""

        model = self.generate_from_config(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, f"interaction_net_{step_txt}_{run}.pt"), map_location=device))
        return model
    
    def __load_best_model_from_dir(self, model_dir: str, run: int, device: str) -> Union[LGInteractionNetwork, HGInteractionNetwork]:
        best_checkpoint_index = self.__get_best_checkpoint_index(run)
        return self.__load_model_from_dir(self.checkpoints.checkpoint_dir, best_checkpoint_index, run, device)

    def save_model(self, model: Union[LGInteractionNetwork, HGInteractionNetwork], run: int) -> None:
        self.__save_model_to_dir(model, self.model_dir, None, run)

    def save_model_checkpoint(self, model: Union[LGInteractionNetwork, HGInteractionNetwork], checkpoint_dir: str, step: int, run: int):
        if self.checkpoints == None:
            return Exception("No checkpoint config defined.")
        
        self.__save_model_to_dir(model, self.checkpoints.checkpoint_dir, step, run)
    
    def load_from_config(self, run: int, device: str, early_stopping: bool = True) -> Union[LGInteractionNetwork, HGInteractionNetwork]:
        return self.__load_model_from_dir(self.model_dir, None, run, device) if not early_stopping else \
               self.__load_best_model_from_dir(self.model_dir, run, device)
    
    def load_checkpoint_from_config(self, run: int, step: int, device: str) -> Union[LGInteractionNetwork, HGInteractionNetwork]:
        if self.checkpoints == None:
            return Exception("No checkpoint config defined.")
        
        return self.__load_model_from_dir(self.checkpoints.checkpoint_dir, step, run, device)
    

    def iterate_checkpoints(self, run: int, device: str, early_stopping: bool = True) -> Generator[Union[LGInteractionNetwork, HGInteractionNetwork], None, None]:
        model_dir = os.path.join(self.checkpoints.checkpoint_dir, f"run_{run}")
        num_files = self.__get_best_checkpoint_index(run) \
            if early_stopping \
            else len(list(pathlib.Path(model_dir).glob('*.pt')))

        for i in range(num_files):
            yield self.load_checkpoint_from_config(run, i, device)



class RunConfig(ConfigBase):
    def __init__(self, num_runs: int, num_steps: int) -> None:
        super().__init__()
        self.num_runs = num_runs 
        self.num_steps = num_steps

class CombinatorialSolverConfig(ConfigBase):
    def __init__(self, solver_lambda: float, margin_alpha: float, 
                 secondary_sensitive: bool) -> None:
        super().__init__()
        self.solver_lambda = solver_lambda
        self.margin_alpha = margin_alpha
        self.secondary_sensitive = secondary_sensitive

    def generate_from_config(self):
        return CombinatorialSolver(**self.__dict__)

class OptimizerConfig(ConfigBase):
    def __init__(self, lr: float, batch_size: int, patience: int, factor: float) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.factor = factor

    def generate_from_config(self, parameters):
        optimizer = torch.optim.RMSprop(parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        return (optimizer, scheduler)


def evaluate_performance_TPR_FPR(network: nn.Module, dataset: HitGraphDataset, transform: CustomGraphTransform,
                                 solver: CombinatorialSolver, indices: Optional[np.ndarray] = None):
    
    assert transform.to_line_graph == isinstance(network, LGInteractionNetwork), "Invalid configuration. Transform must match model config!"

    network.eval()
    TPs, TNs, FPs, FNs = [], [], [], []

    ds_enumerable = dataset if indices is None else subset(dataset, indices)

    for i, G in enumerate(ds_enumerable):
        if solver ==None:
            solver = CombinatorialSolver(1.0, 0.0, False)
        
        HG, LG = transform(G)

        edge_logits = network(LG) if LG != None else network(HG)
        edge_probs = torch.sigmoid(edge_logits)
        edge_probs = solver(HG, edge_probs, train=False)
        edge_labels = HG.masked_edge_labels
        
        TP = torch.sum((edge_labels==1) & (edge_probs>0.5)).item() 
        TN = torch.sum((edge_labels==0) & (edge_probs<0.5)).item()
        FP = torch.sum((edge_labels==0) & (edge_probs>0.5)).item() 
        FN = torch.sum((edge_labels==1) & (edge_probs<0.5)).item()
        TPs += [TP]; TNs += [TN]; FPs += [FP]; FNs += [FN]
        
    TPR = sum(TPs)/(sum(TPs)+sum(FNs))
    FPR = sum(FPs)/(sum(FPs)+sum(TNs))
    
    network.train()
    return TPR, FPR

            
class ExperimentBase(ConfigBase):
    def __init__(self, dataset: DatasetConfig, transforms: GraphTransfromConfig, 
                 model: ModelConfig, solver: Optional[CombinatorialSolverConfig], 
                 optimizer: OptimizerConfig, run: RunConfig) -> None:
        
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.solver = solver
        self.optimizer = optimizer
        self.model = model
        self.run = run
        
    def _get_config_dir(self):
        return os.path.join(self.model.model_dir, "experiment.json")

    def exists(self):
        f = self._get_config_dir()
        exists, matches = False, False
        
        if os.path.exists(self.model.model_dir) and \
           os.path.exists(f) :
            try: 
                comp = type(self).from_file(f, to_object=False)
                src  = delete_test_info(self.__dict__) #Do not compare test info to allow a split evaluation
                                                       #using multiple json configs with a shared underlying model   
                diff = DeepDiff(src, comp, ignore_order=True)
                matches = diff == {}

                print(diff)
                
                # Only print warning if num_runs < experiment
                if not matches and "values_changed" in diff.keys() and len(diff["values_changed"]) == 1:
                    key, values = list(diff["values_changed"].items())[0]
                    if key == "root['run']['num_runs']" and values["old_value"] < values["new_value"]:
                        print(f"Model directory contains more models than specified in experiment definition. Using first {values['old_value']}.")
                        matches = True
                
                exists = True
            except Exception as e: 
                print(e)
            
        if not exists:
            print("Starting model training")
        elif exists and matches:
            print("Skipping model training...")
        elif exists and not matches:
            raise ValueError("Trained model does not match the config.") 
    
        return exists and matches
        
    def generate_datasets(self, device: str = "cpu"):
        self.dataset.generate_from_config(device)

    def train_model(self, device: str):
        exists = self.exists()
        if exists: return

        run_logs = []
        
         
        # Lock directory to avoid starting multiple 
        # training runs for a shared model config.
        with LockDirectory(self.model.model_dir):        
            for run in tqdm(range(self.run.num_runs)):
                log_dir = self.train_model_run(run, device)
                run_logs += [log_dir]

            runs_df = pd.DataFrame(run_logs, columns=["runs"])
            runs_df.to_csv(os.path.join(self.model.model_dir, f"run_logs.csv"))
            
            #Remove test info to allow a split evaluation
            #using multiple json configs with a shared underlying model
            with open(os.path.join(self.model.model_dir, "experiment.json"), "w") as f:
                f.write(self.to_json(del_test_info=True))

    def train_model_run(self, run: int, device: str):
        set_random_seeds(run)
        train_set = self.dataset.load_train_from_config(device)
        val_set = self.dataset.load_validation_from_config(device)
        network = self.model.generate_from_config(device)
        solver = None if self.solver == None else self.solver.generate_from_config()
        transforms = self.transforms.generate_from_config()
        (optimizer, scheduler) = self.optimizer.generate_from_config(network.parameters())

        assert transforms.to_line_graph == isinstance(network, LGInteractionNetwork), "Invalid configuration. Transform must match model config!"

        N = 100
        test_indices = np.arange(len(train_set))
        test_indices = test_indices[::int(np.ceil( len(test_indices) / N ))]

        writer = SummaryWriter()
        network.train()
        
        for step in range(self.run.num_steps):
            optimizer.zero_grad()
            batch_loss = 0

            for _ in range(self.optimizer.batch_size): 
                G = train_set.sample()
                HG, LG = transforms(G)
                
                edge_logits = network(LG) if LG != None else network(HG)
                edge_probs = torch.sigmoid(edge_logits)

                if self.solver != None:
                    edge_probs = solver(HG, edge_probs, train=True)
                    #loss = hamming_loss(edge_probs, HG.masked_edge_labels)

                    nit = HG.next_is_tracker[HG.masked_edge_index].any(dim=0)
                    loss1 = hamming_loss(edge_probs[nit], HG.masked_edge_labels[nit])
                    loss2 = hamming_loss(edge_probs[~nit], HG.masked_edge_labels[~nit])
                    loss = loss1 + loss2
                else:
                    #loss = F.binary_cross_entropy(edge_probs, HG.masked_edge_labels)

                    nit = HG.next_is_tracker[HG.masked_edge_index].any(dim=0)
                    loss1 = F.binary_cross_entropy(edge_probs[nit], HG.masked_edge_labels[nit])
                    loss2 = F.binary_cross_entropy(edge_probs[~nit], HG.masked_edge_labels[~nit])
                    loss = loss1 + loss2

                loss = loss / self.optimizer.batch_size
                batch_loss += float(loss.item())

                loss.backward()
            
            optimizer.step()

            writer.add_scalar("Loss/train", batch_loss, step)

            if self.model.checkpoints != None and step % self.model.checkpoints.checkpoint_freq == 0:
                with torch.no_grad():
                    TPR, FPR = evaluate_performance_TPR_FPR(network, val_set, transforms, solver, None)
                    writer.add_scalar("Performance/tpr_val", float(TPR), step)    # 
                    writer.add_scalar("Performance/fpr_val", float(FPR), step)    # 

                    TPR, FPR = evaluate_performance_TPR_FPR(network, train_set, transforms, solver, test_indices)
                    writer.add_scalar("Performance/tpr_train", float(TPR), step)    # 
                    writer.add_scalar("Performance/fpr_train", float(FPR), step)    # 
            
                    results_df = get_metrics_for_dataset(val_set, transforms, network, solver, 100, None)
                    writer.add_scalar("Performance/pur_val", float(results_df["pur"].mean()), step)
                    writer.add_scalar("Performance/eff_val", float(results_df["eff"].mean()), step)

                    results_df = get_metrics_for_dataset(train_set, transforms, network, solver, 100, test_indices)
                    writer.add_scalar("Performance/pur_train", float(results_df["pur"].mean()), step)
                    writer.add_scalar("Performance/eff_train", float(results_df["eff"].mean()), step)

                self.model.save_model_checkpoint(network, self.model.checkpoints.checkpoint_dir, step, run)

        self.model.save_model(network, run)
        return writer.log_dir

    
    def evaluate_model(self, device: str, early_stopping: bool = True):
        print("Starting model evaluation...")
        if self.solver == None:
            print("Using default combinatorial solver for evaluation")
        
        for phantom in tqdm(self.dataset.iterate_phantoms()):
            results = dict()
            
            for events in tqdm(self.dataset.iterate_events()):
                pur_mus, pur_stds, eff_mus, eff_stds, rej_mus, rej_stds = [], [], [], [], [], []
                results[str(events)] = {"pur_mu": [], "pur_std": [], 
                                        "eff_mu": [], "eff_std": []}
                for run in tqdm(range(self.run.num_runs)):
                    network = self.model.load_from_config(run, device, early_stopping)
                    dataset = self.dataset.load_test_from_config(phantom, events, device)
                    solver = None if self.solver == None else self.solver.generate_from_config()
                    transforms = self.transforms.generate_from_config()

                    network.eval()

                    results_df = get_metrics_for_dataset(dataset, transforms, network, solver, events)
                    
                    pur_mus += [results_df["pur"].mean()]; pur_stds += [results_df["pur"].std()]
                    eff_mus += [results_df["eff"].mean()]; eff_stds += [results_df["eff"].std()]
                    
                    results_df.to_csv(os.path.join(self.model.model_dir, f"results_{phantom}_{events}_{run}.txt"))
                    
                results[str(events)]["pur_mu"] = np.array(pur_mus)
                results[str(events)]["pur_std"] = np.array(pur_stds)
                results[str(events)]["eff_mu"] = np.array(eff_mus)
                results[str(events)]["eff_std"] = np.array(eff_stds)
        
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(self.model.model_dir, f"results_{phantom}.txt"))
            df.to_pickle(os.path.join(self.model.model_dir, f"results_{phantom}.pkl"))

    
    def load_run_logs(self):
        return self.model.load_run_logs()
