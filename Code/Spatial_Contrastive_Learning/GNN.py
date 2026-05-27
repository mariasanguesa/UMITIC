import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import negative_sampling
from tqdm import tqdm 

class GraphSAGE(nn.Module):
    """
    GraphSAGE model for node representation learning
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        # Normalize embeddings
        # x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-5)
        return x

def load_graphs(data_path, device, cnn_features):
    graphs = [torch.load(os.path.join(data_path, f), weights_only=False).to(device)
              for f in os.listdir(data_path) if f.endswith('.pt')]
    num_ch = graphs[0].x.shape[1]-cnn_features-6 
    return graphs, num_ch

# def unsupervised_loss(pos_sim, neg_sim, temp):
#     """
#     Computes contrastive loss between positive and negative pairs using temperature scaling.
#     """
#     pos_exp = torch.exp(pos_sim / temp)
#     neg_exp = torch.exp(neg_sim / temp)
#     return -torch.log(pos_exp / neg_exp.sum(dim=0)).mean()

# def unsupervised_loss(pos_sim, neg_sim, temp):
#     # pos_sim: (N,)  neg_sim: (N, K) — K negativos por positivo
#     logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temp  # (N, K+1)
#     labels = torch.zeros(pos_sim.shape[0], dtype=torch.long, device=pos_sim.device)
#     return F.cross_entropy(logits, labels)

# def unsupervised_loss(pos_sim, neg_sim, temp):
#     """
#     InfoNCE/NT-Xent loss para contrastive learning.
#     pos_sim: (N,) similitud coseno de pares positivos
#     neg_sim: (M,) similitud coseno de pares negativos
#     """
#     eps = 1e-8
#     # Concatenar positivos y negativos
#     logits = torch.cat([pos_sim, neg_sim], dim=0) / temp
#     labels = torch.zeros(pos_sim.shape[0], dtype=torch.long, device=pos_sim.device)  # positivos en la posición 0

#     # El primer elemento de cada fila es el positivo, el resto son negativos
#     logits = torch.stack([torch.cat([pos_sim[i:i+1], neg_sim]) for i in range(pos_sim.shape[0])])
#     labels = torch.zeros(pos_sim.shape[0], dtype=torch.long, device=pos_sim.device)

#     loss = F.cross_entropy(logits, labels)
#     return loss

# def find_negative_edge_index(edge_index, num_negatives):
#     """
#     Generates negative samples for contrastive learning.
#     """
#     neg_edge_index = negative_sampling(
#         edge_index=edge_index,
#         num_neg_samples=min(num_negatives * len(torch.unique(edge_index[0])), edge_index.size(1) * 2)
#         #num_neg_samples=num_negatives*len(torch.unique(edge_index[0]))
#     )
#     return neg_edge_index

def find_negative_edge_index(edge_index, num_negatives):
    N = edge_index.size(1)
    target = N * num_negatives
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_neg_samples=target
    )
    if neg_edge_index.size(1) == 0:
        return None  # señal explícita para saltar el batch
    if neg_edge_index.size(1) < target:
        rep = (target // neg_edge_index.size(1)) + 1
        neg_edge_index = neg_edge_index.repeat(1, rep)
    return neg_edge_index[:, :target]


# def unsupervised_loss(pos_sim, neg_sim, temp, num_negatives):
#     """
#     InfoNCE loss.
#     pos_sim : (N,)          — similitud coseno de pares positivos
#     neg_sim : (N*K,)        — K negativos por cada positivo, en orden
#     Reordena neg_sim a (N, K) para tener un denominador propio por muestra.
#     """
#     N = pos_sim.shape[0]
#     K = num_negatives

#     # Ajustar longitud por si negative_sampling devolvió menos
#     neg_sim = neg_sim[:N * K]
#     neg_sim = neg_sim.reshape(N, K)                            # (N, K)

#     logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temp  # (N, K+1)
#     labels = torch.zeros(N, dtype=torch.long, device=pos_sim.device)   # positivo en col 0
#     return F.cross_entropy(logits, labels)

def unsupervised_loss_inbatch(out, edge_index, temp):
    """
    InfoNCE con in-batch negatives.
    Todos los nodos del batch actúan como negativos entre sí.
    out: (N, D) embeddings ya normalizados L2
    edge_index: aristas positivas del batch
    """
    N = out.size(0)
    
    # Matriz de similitud completa (N x N)
    sim_matrix = torch.mm(out, out.T) / temp          # (N, N)
    
    # Excluir diagonal (auto-similitud)
    eye_mask = torch.eye(N, dtype=torch.bool, device=out.device)
    sim_matrix = sim_matrix.masked_fill(eye_mask, float('-inf'))
    
    # Para cada arista positiva, el denominador son todos los demás nodos
    src, dst = edge_index[0], edge_index[1]
    pos_sim = sim_matrix[src, dst]                     # (E,) similitud de cada arista positiva
    
    # Loss: para cada positivo (src, dst), denominador = fila src de sim_matrix
    loss = -pos_sim + torch.logsumexp(sim_matrix[src], dim=1)
    
    return loss.mean()


def train_unsupervised(model, optimizer, graph_list, num_neighbors, device, scheduler, temperature, batch_size, num_ch, num_negatives):
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_edges = 0
    skipped_batches = 0
    for graph in graph_list:
        data = Data(x=torch.tensor(graph.x[:, num_ch+6:], dtype=torch.float).to(device),
                    edge_index=graph.edge_index.long().to(device))
        # NeighborLoader samples node neighborhoods dynamically
        train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=True)
        for batch in train_loader:
            # Skip small batches with too few edges
            if batch.edge_index.size(1) < 2:  # omitir batches sin aristas
                skipped_batches += 1
                continue
            optimizer.zero_grad()
            # Forward pass
            out = model(batch.x.to(device), batch.edge_index.to(device))
            # Positive similarities: between real connected nodes
            out = F.normalize(out, dim=1)
            # pos_sim = F.cosine_similarity(out[batch.edge_index[0]], out[batch.edge_index[1]])
            # Negative samples: randomly sampled unconnected node pairs
            # neg_edge_index = find_negative_edge_index(batch.edge_index, 2)
            # neg_sim = F.cosine_similarity(out[neg_edge_index[0]], out[neg_edge_index[1]])
            # loss = unsupervised_loss(pos_sim, neg_sim, temperature)
            # neg_edge_index = find_negative_edge_index(batch.edge_index, num_negatives)
            # # If no negatives, skip this batch
            # if neg_edge_index is None:
            #     skipped_batches += 1
            #     continue
            # neg_sim = F.cosine_similarity(out[neg_edge_index[0]], out[neg_edge_index[1]])
            # loss = unsupervised_loss(pos_sim, neg_sim, temperature, num_negatives)
            loss = unsupervised_loss_inbatch(out, batch.edge_index.to(device), temperature)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                src, dst = batch.edge_index[0], batch.edge_index[1]
                last_pos_sim = F.cosine_similarity(out[src], out[dst]).mean().item()
                # Negativos aleatorios para comparar (muestra rápida)
                perm = torch.randperm(out.size(0), device=out.device)
                last_neg_sim = F.cosine_similarity(out, out[perm]).mean().item()

            total_loss += loss.item()
            total_batches += 1
            total_edges += batch.edge_index.size(1)
    # Update learning rate
    scheduler.step()

    avg_loss = total_loss / max(total_batches, 1)

    return {
        'loss': avg_loss,
        'batches': total_batches,
        'skipped': skipped_batches,
        'edges': total_edges,
        'pos_sim': last_pos_sim,
        'neg_sim': last_neg_sim,
    }

def inference(model, graph, num_neighbors, batch_size, device, output_path, num_ch, out_dim):
    model.eval()
    # Extrae las features a usar para el GNN
    x = torch.tensor(graph.x[:, num_ch+6:], dtype=torch.float)
    data = Data(
        x=x,
        edge_index=graph.edge_index.long(),
        input_id=torch.arange(x.shape[0])
    )

    infer_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False
    )

    out = np.zeros([data.x.shape[0], out_dim + num_ch], dtype=np.float32)
    out[:, :num_ch] = graph.x[:, :num_ch]
    emb = np.zeros((data.x.shape[0], out_dim), dtype=np.float32)

    for batch in infer_loader:
        if batch.edge_index.size(1) == 0:
            # Si el batch no tiene aristas, puedes saltarlo o asignar ceros
            continue
        with torch.no_grad():
            batch_emb_t = model(batch.x.to(device), batch.edge_index.to(device))
            batch_emb_t = F.normalize(batch_emb_t, dim=1)
            batch_emb = batch_emb_t.cpu().numpy()
            # NeighborLoader returns embeddings for all sampled nodes; keep only seed nodes.
            seed_size = int(batch.batch_size) if hasattr(batch, 'batch_size') else batch.input_id.shape[0]
            seed_ids = batch.input_id[:seed_size].cpu().numpy()
            emb[seed_ids] = batch_emb[:seed_size]
            if np.isnan(batch_emb).any():
                print(f"NaN encontrado en el embedding del batch con input_id {batch.input_id}")

    out[:, num_ch:] = emb
    filename = os.path.join(output_path, graph.y.split('.')[0] + '.npy')
    np.save(filename, out)

def GNN(args):
    graphs_path = args['path'] +'Spatial_Contrastive_Learning/Spatial_Graphs_2/'
    graphs, num_ch = load_graphs(graphs_path, args['GPU_INDEX'], args['CNN_Out_Dimensions'])

    model = GraphSAGE(in_channels=args['CNN_Out_Dimensions'], hidden_channels=128, out_channels=args['GNN_Out_Dimensions'], dropout=args['GNN_Dropout']).to(args['GPU_INDEX'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['GNN_Learning_Rate'], weight_decay=args['GNN_Weight_Decay'])
    scheduler = StepLR(optimizer, step_size=args['GNN_Step_Size'], gamma=args['GNN_Gamma'])

    # Training loop 
    print(f"{'Epoch':>5} {'Loss':>8} {'pos_sim':>8} {'neg_sim':>8} {'gap':>8} {'batches':>8} {'skipped':>7}")
    print("-" * 60)

    # for epoch in tqdm(range(args['GNN_N_Epochs']), desc="Training Epochs"):
    for epoch in range(args['GNN_N_Epochs']):
        stats = train_unsupervised(
            model, optimizer, graphs, args['GNN_N_Neighbors'],
            args['GPU_INDEX'], scheduler, args['GNN_Temperature'],
            args['GNN_Batch_Size'], num_ch, 4
        )
        gap = stats['pos_sim'] - stats['neg_sim']
        print(
            f"{epoch:>5} "
            f"{stats['loss']:>8.4f} "
            f"{stats['pos_sim']:>8.4f} "
            f"{stats['neg_sim']:>8.4f} "
            f"{gap:>8.4f} "
            f"{stats['batches']:>8} "
            f"{stats['skipped']:>7}"
        )
        
    # Inference loop with tqdm
    gnn_output_path = args['path'] +'Spatial_Contrastive_Learning/Spatial_Representations_2/'
    os.makedirs(gnn_output_path, exist_ok=True)
    for graph in tqdm(graphs, desc="Inference"):
        inference(model, graph, args['GNN_N_Neighbors'], args['GNN_Batch_Size'], args['GPU_INDEX'], gnn_output_path,num_ch, args['GNN_Out_Dimensions'])


# if __name__ == "__main__":
#     GNN()
