import os
from tqdm import tqdm
import numpy as np

def z_score_norm(args_path, args_cnn_out_dimensions):
    path = args_path+'Patch_Contrastive_Learning/Image_Patch_Representation/'
    embs_list = [f for f in os.listdir(path) if f.endswith('.npy')]

    all_embeddings = []
    for emb_name in tqdm(embs_list, desc="Processing feature vectors"):
        embedding = np.load(os.path.join(path, emb_name), allow_pickle=True).astype('float32')
        all_embeddings.append(embedding)

    output_path = args_path+'Patch_Contrastive_Learning/Image_Patch_Representation_Norm/'
    os.makedirs(output_path, exist_ok=True)

    # the first three features and the mean intensity per channel will not be normalized 
    combined_features = np.concatenate([emb[:, -args_cnn_out_dimensions-6:] for emb in all_embeddings], axis=0)

    global_mean = np.mean(combined_features, axis=0)
    global_std = np.std(combined_features, axis=0)

    for i, emb_name in enumerate(tqdm(embs_list, desc="Saving normalized feature vectors")):
        embedding = all_embeddings[i]
        z_score_embedding = np.copy(embedding)
        z_score_embedding[:, -args_cnn_out_dimensions-6:] = (embedding[:, -args_cnn_out_dimensions-6:] - global_mean) / global_std
        np.save(os.path.join(output_path, emb_name), z_score_embedding)

def z_score_norm_spatial(args_path, args_gnn_out_dimensions):
    path = args_path +'Spatial_Contrastive_Learning/Spatial_Representations_2/'
    embs_list = [f for f in os.listdir(path) if f.endswith('.npy')]

    all_embeddings = []
    for emb_name in tqdm(embs_list, desc="Processing feature vectors"):
        embedding = np.load(os.path.join(path, emb_name), allow_pickle=True).astype('float32')
        all_embeddings.append(embedding)

    output_path = args_path+'Spatial_Contrastive_Learning/Spatial_Representations_Norm_2/'
    os.makedirs(output_path, exist_ok=True)

    # the mean intensity per channel will not be normalized 
    combined_features = np.concatenate([emb[:, -args_gnn_out_dimensions:] for emb in all_embeddings], axis=0)

    global_mean = np.mean(combined_features, axis=0)
    global_std = np.std(combined_features, axis=0)

    for i, emb_name in enumerate(tqdm(embs_list, desc="Saving normalized feature vectors")):
        embedding = all_embeddings[i]
        z_score_embedding = np.copy(embedding)
        z_score_embedding[:, -args_gnn_out_dimensions:] = (embedding[:, -args_gnn_out_dimensions:] - global_mean) / global_std
        np.save(os.path.join(output_path, emb_name), z_score_embedding)
