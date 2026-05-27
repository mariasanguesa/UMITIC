import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import gc
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from collections import Counter
from scipy.stats import mode
from Code.utils.heatMAPS import showHeatmapNeighbors,showNumCellsPerCommunity,showHeatmap,cluster_area_percentage,plot_aligned_heatmap
from Code.utils.Spatial_Maps import spatial_distr
from Code.utils import utilz
import faiss
from sklearn.decomposition import PCA

plt.switch_backend('agg')

def neighborhood_assignment(args):
    """
    Performs neighborhood assignment using contrastive patch embeddings and Leiden clustering.
    """

    # Load embedding files
    embeddings_path = os.path.join(args['path'], 'Spatial_Contrastive_Learning/Spatial_Representations_Norm_2/')
    embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('.npy')]

    output_dir = os.path.join(args['path'], 'Clustering/Neighborhood_Assignment_2_res_' + str(args['Leiden_Neighborhood_Resolution']) + '/')
    os.makedirs(output_dir, exist_ok=True)

    # Processing parameters
    sampling_percentage = args['Leiden_Percentage']
    clustering_resolution = args['Leiden_Neighborhood_Resolution']

    # Define channel names
    _, channels = utilz.load_channels(os.path.join(args['path'], 'Raw_Data/'))

    all_embeddings = {}
    embeddings_subset = []
    neighborhoods_per_node = {}
    sampled_node_indices = {}

    #Load and sample embeddings
    for file_name in tqdm(embedding_files, desc="Loading and sampling embeddings for neighborhood assignment"):
        embedding = np.load(os.path.join(embeddings_path, file_name), allow_pickle=True).astype('float32')
        all_embeddings[file_name] = embedding
        num_nodes = embedding.shape[0]
        sampled_indices = np.random.choice(num_nodes, size=int(num_nodes * sampling_percentage), replace=False)
        embeddings_subset.append(embedding[sampled_indices])
        neighborhoods_per_node[file_name.replace('.npy', '.tiff.npy')] = np.full(num_nodes, -1, dtype=int)
        sampled_node_indices[file_name] = sampled_indices

    # Combine all sampled embeddings
    embeddings_subset = np.vstack(embeddings_subset)

    # # Visualización rápida
    # X = embeddings_subset[:, len(channels):]
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    # plt.figure(figsize=(6,6))
    # plt.scatter(X_pca[:,0], X_pca[:,1], s=1, alpha=0.3)
    # plt.title("PCA de los embeddings GNN")
    # plt.savefig(os.path.join(output_dir, 'PCA'))

    #  Clustering using Scanpy 
    adata = sc.AnnData(X=embeddings_subset[:, len(channels):])
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca', metric='cosine')
    # sc.pp.neighbors(adata, n_neighbors=15, use_rep='X', metric='cosine')
    sc.tl.leiden(adata, resolution=clustering_resolution, use_weights=True, n_iterations=-1)
    community_labels = adata.obs['leiden'].astype(int).values

    #  Load phenotype assignments
    # phenotype_path = os.path.join(args['path']+'Clustering/Cell_Type_Assignment_resolution_2_'+str(args['Leiden_Cell_Type_Resolution']), 'final_labels_all_nodes.npz')
    # phenotype_assignments = np.load(phenotype_path)

    neighborhood_names = [f'NB {nb + 1}' for nb in np.unique(community_labels)]
    phenotype_names = []
    phenotype_labels = []
    index = 0

    # Assign neighborhood labels to sampled nodes 
    for file_name, indices in sampled_node_indices.items():
        key = file_name.replace('.npy', '.tiff.npy')
        neighborhoods_per_node[key][indices] = community_labels[index:index + len(indices)]
        key = file_name.replace('.npy', '.tif.npy')
        # phenotype_labels.append(phenotype_assignments[key][indices])
        index += len(indices)

    # phenotype_labels = np.concatenate(phenotype_labels)
    # phenotype_names = [f'T {ph + 1}' for ph in np.unique(phenotype_labels)]

    # Save neighborhood assignments
    np.savez(os.path.join(output_dir, 'leiden_neighborhoods_per_node.npz'), **neighborhoods_per_node)

    # Visualization
    # showHeatmapNeighbors(phenotype_labels, community_labels, neighborhood_names, phenotype_names,os.path.join(output_dir, 'nb_leiden_phenotype'))

    showNumCellsPerCommunity(dict(Counter(community_labels)),os.path.join(output_dir, 'leiden_num_cells_per_nb'), 1)

    channel_intensity = embeddings_subset[:, :len(channels)]
    showHeatmap(channels, neighborhood_names, community_labels, channel_intensity,os.path.join(output_dir, 'leiden_nb_per_channel'), True)

    # Process remaining (non-sampled) nodes
    new_node_indices = {}
    new_embeddings = []

    for file_name in embedding_files:
        embedding = all_embeddings[file_name]
        key = file_name.replace('.npy', '.tiff.npy')
        remaining_indices = np.where(neighborhoods_per_node[key] == -1)[0]
        new_node_indices[file_name] = remaining_indices
        new_embeddings.append(embedding[remaining_indices])

    new_embeddings = np.vstack(new_embeddings)

    # Fit k-NN using original community embeddings with Faiss
    d = adata.X.shape[1]
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(adata.X.astype('float32'))

    # Predict community label for new nodes using mode of neighbors
    assigned_communities = []
    batch_size = 1000
    for i in range(0, new_embeddings.shape[0], batch_size):
        batch = new_embeddings[i:i + batch_size, len(channels):].astype('float32')
        _, neighbors = index.search(batch, 30)  # search k nearest neighbors
        batch_communities = mode(community_labels[neighbors], axis=1, keepdims=True)[0]
        assigned_communities.extend(batch_communities)

    # New assignments 
    new_phenotype_labels = []
    idx = 0
    for file_name, indices in new_node_indices.items():
        key = file_name.replace('.npy', '.tiff.npy')
        neighborhoods_per_node[key][indices] = np.array(assigned_communities[idx:idx + len(indices)]).flatten()  # Convertir a array de NumPy
        key = file_name.replace('.npy', '.tif.npy.npy')
        # new_phenotype_labels.extend(phenotype_assignments[key][indices])
        idx += len(indices)

    # Save updated neighborhoods
    np.savez(os.path.join(output_dir, 'neighborhoods_per_node_complete.npz'), **neighborhoods_per_node)

    # Visualization: updated distributions
    community_labels = np.array(community_labels).flatten()
    assigned_communities = np.array(assigned_communities).flatten()
    total_communities = np.concatenate((community_labels, assigned_communities))
    showNumCellsPerCommunity(dict(Counter(total_communities)),os.path.join(output_dir, 'new_embeddings_num_cells_per_nb'), 1)
    new_channel_intensity = new_embeddings[:, :len(channels)]
    all_channel_intensity = np.concatenate((channel_intensity, new_channel_intensity), axis=0)
    normalized_intensity, _ = showHeatmap(channels, neighborhood_names, total_communities,all_channel_intensity,os.path.join(output_dir, 'new_embeddings_nb_per_channel'), True)

    # Percentage per cluster
    cluster_area_percentage([Counter(total_communities)[k] for k in sorted(Counter(total_communities).keys())],'Neighborhood',output_dir)

    # Final aligned heatmap
    plot_aligned_heatmap(normalized_intensity, channels, neighborhood_names, showNumCellsPerCommunity(Counter(total_communities), "", 0), os.path.join(output_dir, 'aligned_heatmap'))

    # Spatial distribution plots
    # embedding_input_path = os.path.join(args['path'], 'Patch_Contrastive_Learning/Image_Patch_Representation/')
    # spatial_distr(neighborhoods_per_node, len(np.unique(community_labels)), output_dir, embedding_input_path)


