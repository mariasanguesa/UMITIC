import torch
from torch_geometric.nn import radius
from torch_geometric.data import Data
import numpy as np
import os 
from tqdm import tqdm

def rad_graph(cords, rad):
    """
    Constructs a graph based on spatial proximity using a variable radius for each node.
    """
    edge_indices = []
    # For each node (cell), its neighbors are searched according to its search radius.
    for index, (cord, r) in enumerate(zip(cords, rad)):
        # Compute neighbor indices using radius-based search
        neighbors = radius(torch.from_numpy(cords), torch.from_numpy(cord), r)        
        neighbor_indices = neighbors[1]
        # Mask to keep only edges where the querying node is at index 0
        mask = neighbors[0] == 0
        filtered_neighbor_indices = neighbor_indices[mask]
        # Assign correct source index (not 0) to represent the true node index
        edge_indices.append(torch.stack([torch.full_like(filtered_neighbor_indices, index), filtered_neighbor_indices]))

    edge_index = torch.cat(edge_indices, dim=1)
    return edge_index

def remove_and_sort_edges(edge_index):
    """
    Removes duplicate and unordered edges from the graph.
    """
    unique_edges = set()
    filtered_edges = [[], []]

    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        
        edge = tuple(sorted((u, v)))
        
        if edge not in unique_edges:
            unique_edges.add(edge)
            filtered_edges[0].append(u)
            filtered_edges[1].append(v)

    unique_edge_index = torch.tensor(filtered_edges)
    return unique_edge_index

def remove_self_edges(edge_index):
    """
    Removes self-loops (edges from a node to itself)
    """
    mask = edge_index[0] != edge_index[1]
    filtered_edge_index = edge_index[:, mask]
    return filtered_edge_index

def graph_generation(args_path):
    path = args_path+'Patch_Contrastive_Learning/Image_Patch_Representation_Norm/'
    output_path = args_path+'Spatial_Contrastive_Learning/Spatial_Graphs_2/'
    os.makedirs(output_path,exist_ok=True)

    embs = [f for f in os.listdir(path) if f.endswith('.npy')]
    for emb_name in tqdm(embs, desc="Processing graphs"):
        # Load the embedding: assumes columns [x, y, radius, features...]
        emb = np.load(os.path.join(path,emb_name), allow_pickle=True) 
        edge_index = rad_graph(emb[:,[0,1]].astype(float),1*(emb[:,2].astype(float)+1))
        # edge_index = rad_graph(emb[:,[0,1]].astype(float), np.full((emb.shape[0],), 20.0))
        # unique_edge_index = remove_and_sort_edges(edge_index)
        filtered_edge_index = remove_self_edges(edge_index)
        # Extract node features (excluding x, y, radius)
        # Create torch_geometric Data object
        data = Data(edge_index=filtered_edge_index, x=emb[:,3:].astype(float) , y=emb_name)
        torch.save(data, os.path.join(output_path, 'data_{}.pt'.format(emb_name.split('.tiff.npy')[0],0)))
