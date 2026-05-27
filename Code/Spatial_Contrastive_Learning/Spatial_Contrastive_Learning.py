from Code.Spatial_Contrastive_Learning.graph_generation import graph_generation
from Code.Spatial_Contrastive_Learning.z_score import z_score_norm, z_score_norm_spatial
from Code.Spatial_Contrastive_Learning.GNN import GNN

def spatial_contrastive_learning(args):
    # Compute z-score normalization
    z_score_norm(args['path'], args['CNN_Out_Dimensions'])

    # Create graphs with spatial information
    graph_generation(args['path'])

    # Train GNN and generate spatial-aware embeddings
    GNN(args )

    z_score_norm_spatial(args['path'], args['GNN_Out_Dimensions'])