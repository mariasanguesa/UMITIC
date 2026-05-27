import os

from sklearn.decomposition import PCA
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import scanpy as sc
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from Code.utils.heatMAPS import showHeatmap, showNumCellsPerCommunity, plot_heatmap_with_bars, plot_aligned_heatmap
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import mode
from Code.utils import utilz

plt.switch_backend('agg')


def cell_type_assignment(args):
    """
    Performs cell type assignment using contrastive patch embeddings and Leiden clustering.
    """

    embeddings_path = os.path.join(args['path'], 'Patch_Contrastive_Learning/Image_Patch_Representation_Norm/')
    embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('.npy')]

    _, marker_channels = utilz.load_channels(os.path.join(args['path'], 'Raw_Data/'))

    sampled_embeddings = []
    node_labels_per_file = {}
    sampled_node_indices = {}
    full_embeddings_dict = {}

    sampling_percentage = args['Leiden_Percentage']
    clustering_resolution = args['Leiden_Cell_Type_Resolution']

    output_dir = os.path.join(args['path'], 'Clustering/Cell_Type_Assignment_resolution_'+str(args['Leiden_Cell_Type_Resolution']) + '/')
    os.makedirs(output_dir, exist_ok=True)

    # Load and subsample embeddings from each file
    for file_name in tqdm(embedding_files, desc="Loading and sampling embeddings"):
        embedding = np.load(os.path.join(embeddings_path, file_name), allow_pickle=True).astype('float32')[:, 3:]
        full_embeddings_dict[file_name] = embedding

        num_nodes = embedding.shape[0]
        sampled_indices = np.random.choice(num_nodes, size=int(num_nodes * sampling_percentage), replace=False)
        sampled_embeddings.append(embedding[sampled_indices])
        node_labels_per_file[file_name] = np.full(num_nodes, -1, dtype=int)
        sampled_node_indices[file_name] = sampled_indices

    # Stack sampled embeddings for clustering
    sampled_embeddings = np.vstack(sampled_embeddings)

    # X = sampled_embeddings[:, len(marker_channels)+6:]

    # Visualización rápida
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    # plt.figure(figsize=(6,6))
    # plt.scatter(X_pca[:,0], X_pca[:,1], s=1, alpha=0.3)
    # plt.title("PCA de los embeddings GNN")
    # plt.savefig(os.path.join(output_dir, 'PCA'))

    # Run Leiden clustering using Scanpy
    adata = sc.AnnData(X=sampled_embeddings[:, len(marker_channels):])
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
    sc.tl.leiden(adata, resolution=clustering_resolution, use_weights=True, n_iterations=-1)
    cluster_labels = adata.obs['leiden'].astype(int).values

    # Assign cluster labels back to sampled nodes
    index = 0
    for file_name, sampled_indices in sampled_node_indices.items():
        node_labels_per_file[file_name][sampled_indices] = cluster_labels[index:index + len(sampled_indices)]
        index += len(sampled_indices)

    # Save intermediate cluster assignments
    np.savez(os.path.join(output_dir, 'leiden_labels_sampled_nodes.npz'), **node_labels_per_file)

    # Visualization for the initial clustering
    phenotype_names = ['Ph ' + str(label + 1) for label in np.unique(cluster_labels)]
    marker_intensities = sampled_embeddings[:, :len(marker_channels)]

    norm_intensities, _ = showHeatmap(marker_channels, phenotype_names, cluster_labels, marker_intensities, os.path.join(output_dir, 'leiden_phenotypes'), True)
    cell_counts = showNumCellsPerCommunity(dict(Counter(cluster_labels)), os.path.join(output_dir, 'leiden_cell_counts'), 0)
    plot_heatmap_with_bars(norm_intensities, marker_channels, phenotype_names, cell_counts, os.path.join(output_dir, 'leiden_intensity_barplot'))

    # Prepare for KNN-based label propagation
    unsampled_node_indices = {}
    unsampled_embeddings = []

    for file_name in tqdm(embedding_files, desc="Processing unsampled nodes"):
        embedding = full_embeddings_dict[file_name]
        indices = np.where(node_labels_per_file[file_name] == -1)[0]
        unsampled_node_indices[file_name] = indices
        unsampled_embeddings.append(embedding[indices])

    unsampled_embeddings = np.vstack(unsampled_embeddings)

    # KNN fit on clustered nodes
    knn_model = NearestNeighbors(n_neighbors=30, metric='cosine', n_jobs=-1)
    knn_model.fit(adata.X)

    predicted_labels = []
    batch_size = 1000
    for i in tqdm(range(0, unsampled_embeddings.shape[0], batch_size), desc="Propagating labels with KNN"):
        batch = unsampled_embeddings[i:i + batch_size]
        _, neighbors = knn_model.kneighbors(batch[:, len(marker_channels):])
        neighbor_clusters = cluster_labels[neighbors]
        batch_labels = mode(neighbor_clusters, axis=1, keepdims=False)[0]
        predicted_labels.extend(batch_labels)

    # Combine results for final visualization
    # Ensure both arrays are 1D
    predicted_labels = np.array(predicted_labels).flatten()
    cluster_labels = np.array(cluster_labels).flatten()

    combined_labels = np.concatenate((predicted_labels, cluster_labels))

    combined_marker_data = np.concatenate((unsampled_embeddings[:, :len(marker_channels)], marker_intensities), axis=0)

    norm_intensities_all, _ = showHeatmap(marker_channels, phenotype_names, combined_labels, combined_marker_data, os.path.join(output_dir, 'all_nodes'), False)
    cell_counts_all = showNumCellsPerCommunity(dict(Counter(combined_labels)), os.path.join(output_dir, 'all_nodes_cell_counts'), 0)
    plot_aligned_heatmap(norm_intensities_all, marker_channels, phenotype_names, cell_counts_all, os.path.join(output_dir, 'final_heatmap'))

    # Final label assignment to unsampled nodes
    index = 0
    for file_name, indices in unsampled_node_indices.items():
        node_labels_per_file[file_name][indices] = predicted_labels[index:index + len(indices)]
        index += len(indices)

    # Save complete label assignments
    np.savez(os.path.join(output_dir, 'final_labels_all_nodes.npz'), **node_labels_per_file)

    # (Optional) t-SNE visualization of embeddings (commented out)
    # tsne = TSNE(n_components=2, random_state=42).fit_transform(unsampled_embeddings[:, -128:])
    # plt.figure(figsize=(8, 6))
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=predicted_labels, cmap='tab20')
    # plt.title('t-SNE on Remaining Nodes')
    # plt.xlabel('t-SNE 1')
    # plt.ylabel('t-SNE 2')
    # plt.colorbar(label='Leiden Communities')
    # plt.savefig(os.path.join(output_dir, 'tsne_remaining_nodes.png'), dpi=300)
    # plt.close()
    # gc.collect()
