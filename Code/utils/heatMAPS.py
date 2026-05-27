import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
from collections import Counter
import pandas as pd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib as mpl

def showHeatmap(channels, pheno_names, ph_assign, ch_intensity,name_to_save, save_excel):

    ch_intensity_per_ph = np.zeros((len(pheno_names), len(channels)))
         
    for ph in np.unique(ph_assign):
        indices = np.where(ph_assign == ph)[0]
        ch_intensity_per_ph[ph, :] = np.mean(ch_intensity[indices, :], axis=0)
        
    plt.figure(figsize=(18, 12))
    plt.rcParams.update({'font.size': 14})
    sns.heatmap(data=ch_intensity_per_ph, xticklabels=channels, yticklabels=pheno_names,cmap='bwr',linewidths=0.5,linecolor='gray',cbar_kws={'shrink': 0.75})  
    plt.title('Raw heatmap', fontsize=16)
    plt.savefig(name_to_save+'_raw.png', dpi=300, bbox_inches='tight')
    plt.clf()

    if save_excel:        
        df = pd.DataFrame(ch_intensity_per_ph, columns=channels, index=pheno_names)

        excel_filename = name_to_save + '_data.xlsx'
        df.to_excel(excel_filename)


    ch_intensity_per_ph_norm_col = np.zeros(ch_intensity_per_ph.shape)
    for j in range(len(channels)):
        min_val = np.min(ch_intensity_per_ph[:, j])
        max_val = np.max(ch_intensity_per_ph[:, j])
        if max_val - min_val > 0:
            ch_intensity_per_ph_norm_col[:, j] = ((ch_intensity_per_ph[:, j] - min_val) / (max_val - min_val)) * 100
        else:
            ch_intensity_per_ph_norm_col[:, j] = 0  

    sns.heatmap(data=ch_intensity_per_ph_norm_col, xticklabels=channels, yticklabels=pheno_names,cmap='bwr',linewidths=0.5,linecolor='gray',cbar_kws={'shrink': 0.75}) 
    plt.title('Normalized heatmap per channels', fontsize=16) 
    plt.savefig(name_to_save+'_norm_per_ch.png', dpi=300, bbox_inches='tight')
    plt.clf()

    ch_intensity_per_ph_norm_row = np.zeros(ch_intensity_per_ph.shape)
    for i in np.unique(ph_assign):
        min_val = np.min(ch_intensity_per_ph[i, :])
        max_val = np.max(ch_intensity_per_ph[i, :])
        if max_val - min_val > 0:
            ch_intensity_per_ph_norm_row[i, :] = ((ch_intensity_per_ph[i, :] - min_val) / (max_val - min_val)) * 100
        else:
            ch_intensity_per_ph_norm_row[i, :] = 0  

    sns.heatmap(data=ch_intensity_per_ph_norm_row, xticklabels=channels, yticklabels=pheno_names,cmap='bwr',linewidths=0.5,linecolor='gray',cbar_kws={'shrink': 0.75}) 
    plt.title('Normalized heatmap per phenotype', fontsize=16) 
    plt.savefig(name_to_save+'_norm_per_ph.png', dpi=300, bbox_inches='tight')
    plt.clf()

    return ch_intensity_per_ph_norm_col, ch_intensity_per_ph_norm_row
    

def showNumCellsPerCommunity(dic, name_to_save, cluster):
    name_short_cluster = 'Ph' if cluster == 0 else 'Nb'
    name_cluster = 'phenotype' if cluster == 0 else 'neighbor'
    x = [name_short_cluster +str(key+1) for key in np.sort(list(dic.keys()))]
    y = [float(dic[k]) for k in np.sort(list(dic.keys()))]
    sns.barplot(x=x, y=y)
    plt.title(f'Num cells per {name_cluster}')
    plt.ylabel('Num cells')
    plt.savefig(name_to_save+'.png')

    return y

def showHeatmapNeighbors(ph_assignment, nb_assignment,nb_names,pheno_names,name_to_save):
    ph_in_nb = np.zeros((len(np.unique(nb_assignment)),len(np.unique(ph_assignment))))

    for nb in np.unique(nb_assignment):
        indices = np.where(nb_assignment == nb)[0]
        ph_per_nb = np.zeros(len(np.unique(ph_assignment)))
        for i in indices:
            ph_per_nb[ph_assignment[i]] += 1
        ph_in_nb[nb, :] = ph_per_nb
    
    sns.heatmap(data=ph_in_nb, xticklabels=pheno_names, yticklabels=nb_names,cmap='bwr')  
    plt.title('Raw heatmap') 
    plt.savefig(name_to_save+'_raw.png')
    plt.clf()

    ph_in_nb_norm_col = np.zeros(ph_in_nb.shape)
    for j in np.unique(ph_assignment):
        min_val = np.min(ph_in_nb[:, j])
        max_val = np.max(ph_in_nb[:, j])
        if max_val - min_val > 0:
            ph_in_nb_norm_col[:, j] = ((ph_in_nb[:, j] - min_val) / (max_val - min_val)) * 100
        else:
            ph_in_nb_norm_col[:, j] = 0  

    sns.heatmap(data=ph_in_nb_norm_col, xticklabels=pheno_names, yticklabels=nb_names,cmap='bwr')  
    plt.title('Normalized heatmap') 
    plt.savefig(name_to_save+'_norm_per_ph.png')
    plt.clf()

    ph_in_nb_norm_row = np.zeros(ph_in_nb.shape)
    for i in np.unique(nb_assignment):
        min_val = np.min(ph_in_nb[i, :])
        max_val = np.max(ph_in_nb[i, :])
        if max_val - min_val > 0:
            ph_in_nb_norm_row[i, :] = ((ph_in_nb[i, :] - min_val) / (max_val - min_val)) * 100
        else:
            ph_in_nb_norm_row[i, :] = 0  

    sns.heatmap(data=ph_in_nb_norm_row, xticklabels=pheno_names, yticklabels=nb_names,cmap='bwr')  
    plt.title('Normalized heatmap') 
    plt.savefig(name_to_save+'_norm_per_nb.png')
    plt.clf()

def plot_heatmap_with_bars(ch_intensity_per_ph_norm_row, channels, pheno_names, num_cells_per_ph, name_to_save):
    fig, (ax1, ax2) = plt.subplots(ncols=2, 
                                   figsize=(10, 6), 
                                   sharey=True, 
                                   gridspec_kw={'width_ratios': [4, 1]})

    heatmap = sns.heatmap(ch_intensity_per_ph_norm_row,
                          xticklabels=channels,
                          yticklabels=pheno_names,
                          cmap='bwr',
                          linewidths=0.5,
                          linecolor='gray',
                          cbar=False,
                          ax=ax1)

    ax1.set_ylabel("Phenotypes")
    ax1.set_xlabel("Markers")
    ax1.set_title("Normalized heatmap per phenotype")

    cbar_ax = fig.add_axes([0.03, 0.05, 0.1, 0.03])
    cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0,50,100])
    cbar_ax.set_title("Mean expression", fontsize=8)

    ax2.barh(range(len(pheno_names)), num_cells_per_ph, color='blue', align='edge')
    ax2.set_xlabel("Num cells")
    ax2.set_title("Num cells per phenotype")
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False)

    # plt.savefig(name_to_save + '_combined.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()

def plot_aligned_heatmap(ch_intensity_per_ph_norm_row, channels, pheno_names, num_cells_per_ph, name_to_save):
    pheno_names = [i.replace("Ph","T") for i in pheno_names]
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'figure.dpi': 300
    })
    
    cmap = plt.get_cmap('RdBu_r')
    
    fig = plt.figure(figsize=(12, 8))
    
    gs = fig.add_gridspec(nrows=1, ncols=13, wspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, :10])  # Heatmap
    ax2 = fig.add_subplot(gs[0, 10:12])  # Bar chart
    
    heatmap = sns.heatmap(ch_intensity_per_ph_norm_row,
                         xticklabels=channels,
                         yticklabels=pheno_names,
                         cmap=cmap,
                         linewidths=0.8,
                         linecolor='white',
                         cbar=False, 
                         ax=ax1,
                         square=False)
    
    ax1.set_ylabel("Cell types", fontweight='bold', labelpad=15)
    ax1.set_xlabel("Markers", fontweight='bold', labelpad=15)
    ax1.set_title("Mean marker expression by cell type", fontweight='bold', pad=15)
    
    ax1.set_xticklabels(channels, rotation=45, ha='right', fontweight='bold')
    ax1.set_yticklabels(pheno_names, fontweight='bold')
    
    cbar_ax = fig.add_subplot(gs[0, 12])
    cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=12, width=1, length=6)
    cbar.ax.set_ylabel("Expression level", fontweight='bold', labelpad=15, fontsize=14)
    
    norm = plt.Normalize(min(num_cells_per_ph), max(num_cells_per_ph))
    colors = plt.cm.Greens(norm(num_cells_per_ph))
    
    y_centers = np.arange(len(pheno_names))
    
    ax2.barh(y_centers, num_cells_per_ph, color=colors, align='edge', height=0.9, edgecolor='black', linewidth=0.8)
    
    ax2.set_ylim(ax1.get_ylim())
    
    ax2.set_xlabel("Cell count", fontweight='bold', labelpad=15)
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    for i, v in enumerate(num_cells_per_ph):
        if v > max(num_cells_per_ph) * 0.3:  # Only label bars that are at least 30% of the max value
            color='black'
            if v > max(num_cells_per_ph) * 0.7:
                color='white'
            ax2.text(v*0.92, y_centers[i]+0.45, f'{int(v):,}', 
                    va='center', ha='right', 
                    fontsize=10, fontweight='bold', color=color)
                
        else:
            ax2.text(v*1.05, y_centers[i]+0.45, f'{int(v):,}', 
                    va='center', 
                    fontsize=10, fontweight='bold')
    
    ax2.set_title("Cells per cell type", fontweight='bold', pad=15)
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.2)
    
    plt.savefig(f"{name_to_save}_publication.png", dpi=600, bbox_inches='tight')
    # plt.savefig(f"{name_to_save}_publication.pdf", format='pdf', bbox_inches='tight')

def cluster_area_percentage(num_cells_per_ph,cluster_name,name_to_save):
    total_cells = sum(num_cells_per_ph)
    percentages = [round((cells / total_cells) * 100, 1) for cells in num_cells_per_ph]

    if cluster_name =='Phenotype':
        clusters = [f"Ph{i+1}" for i in range(len(num_cells_per_ph))]
    elif cluster_name =='Neighborhood':
        clusters = [f"Nb{i+1}" for i in range(len(num_cells_per_ph))]
    plt.figure(figsize=(10, 6), dpi=300)

    colors = plt.cm.tab20(np.linspace(0, 1, len(num_cells_per_ph)))
    bars = plt.bar(clusters, percentages, color=colors, width=0.7, edgecolor='black', linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}%', ha='center', va='bottom', fontsize=9)

    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xlabel(cluster_name, fontsize=12)
    plt.title('Total Area by '+cluster_name, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(percentages) * 1.15)  

    plt.tight_layout()
    plt.savefig(name_to_save +'clustering_distribution.png', dpi=300, bbox_inches='tight')

    # plt.show()
if __name__ == "__main__":
    channels = ["DAPI", "CD20", "Ki67", "CD21"]  # Nombres de los canales
    pheno_names = ["Ph 1", "Ph 2", "Ph 3", "Ph 4", "Ph 5", "Ph 6", "Ph 7", "Ph 8"]  # Nombres de fenotipos
    num_cells_per_ph = [17000, 15000, 14500, 14000, 7500, 7000, 6500, 100]  # Número de células por fenotipo
    ch_intensity_per_ph_norm_row = np.random.rand(8, 4) * 100  # Datos de ejemplo del heatmap
    # plot_aligned_heatmap(ch_intensity_per_ph_norm_row, channels, pheno_names, 
                                #  num_cells_per_ph, "immune_profile", 
                                #  cmap_name='RdBu_r')
    # plot_heatmap_with_bars(ch_intensity_per_ph_norm_row, channels, pheno_names, num_cells_per_ph, "output")
    # cluster_area_percentage(num_cells_per_ph,"Phenotype")
    # plot_bubble_chart(ch_intensity_per_ph_norm_row, channels, pheno_names, num_cells_per_ph, "output")
    
    # Create the visualisation
    # plot_dotplot_with_expression(ch_intensity_per_ph_norm_row, percent_positive_cells, 
    #                            channels, pheno_names, num_cells_per_ph, "output")    

    # emb = np.load('C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Image_Patch_Representation/Panel33C_Scan2_12_14.tiff.npy', allow_pickle=True)
    # ph_assignment = np.load('C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/fenotipos_12_14.npy')
    # ch = ['CD21','CD23','CD20','CD4','CK','CD8','DAPI']
    # ph_names = ['Ph '+str(ph+1) for ph in np.unique(ph_assignment)]
    # ch_int = emb[:,3:3+len(ch)].astype('float')

    # showHeatmap(ch, ph_names, ph_assignment, ch_int,'heatmap_try')
    
    # file = np.load('C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/fenotipos_por_nodo_total_64.npz')
    # fenotipo_tamanos= dict(Counter({np.int64(0): 19292, np.int64(1): 14519, np.int64(2): 11430, np.int64(3): 11105, np.int64(4): 10299, np.int64(5): 9987, np.int64(6): 8979, np.int64(7): 4882, np.int64(8): 131}))

    # showNumCellsPerPh( dict(Counter(file['Panel33C_Scan2_17_10.tiff.npy.npy'])))