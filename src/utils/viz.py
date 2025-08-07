import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from src.utils.graph import HitGraph, MaskedHitGraph
from mpl_toolkits.mplot3d import Axes3D


def create_figure() -> Axes3D:
    fig = plt.figure(figsize=(5, 10))
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.view_init(10, 60)
    ax.set_box_aspect(aspect = (1,1,2))
    return fig, ax


def plot_centroids(ax: Axes3D, graph: HitGraph, s:int = 1):
    c='black'    
    ax.scatter(graph.pos[:,0].cpu(), 
               graph.pos[:,1].cpu(), 
               graph.pos[:,2].cpu(), 
               marker='.', c=c, s=s)


def configure_plot(ax: Axes3D):
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])

    ax.set_xlabel("x-position [mm]")
    ax.set_ylabel("y-position [mm]")
    ax.set_zlabel("z-position [mm]")
    ax.margins(z=0.1)
    plt.tight_layout()


def save_figure(save_path):
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def iterate_tracks(tracks: List):
    for track in tracks:
        track_len = len(track)
        
        if track_len == 0:
            continue
        
        if isinstance(track, torch.Tensor):
            track_idx = track
        elif isinstance(track[0], torch.Tensor):
            track_idx = torch.stack(track)
        else:
            track_idx = np.array(track)
            
        yield track_idx, track_len


def plot_colored_tracks(storage: Union[HitGraph, MaskedHitGraph], tracks: List,
                        save_path: str = None, filter="all", centroids=True, ax= None) -> None:
    if ax != None and save_path != None:
        raise Exception("Cannot save plots that have an axis provided!")

    graph = storage if isinstance(storage, HitGraph) else storage.graph
    
    if ax == None:
        fig, ax = create_figure()

    if centroids:
        plot_centroids(ax, graph)
          
    for track_idx, track_len in iterate_tracks(tracks):     
        track_pos = graph.pos[track_idx].cpu()
        track_event_ids = graph.y[track_idx].cpu()
        track_event_id = track_event_ids[-1]
        
        if filter == "true" and not torch.all(track_event_ids == track_event_id):
            continue
        if filter == "false" and torch.all(track_event_ids == track_event_id):
            continue
        
        for i in reversed(range(track_len - 1)):
            f, t = track_pos[i], track_pos[i+1] 
            
            color = 'green' if track_event_id == track_event_ids[i] else 'red'
            if color == 'red' and track_event_ids[i] == track_event_ids[i+1]:
                color = 'orange'
            
            ax.plot([f[0], t[0]], [f[1], t[1]], [f[2], t[2]], color=color, linewidth=1.5)
            
    configure_plot(ax)

    if save_path != None:
        save_figure(save_path)