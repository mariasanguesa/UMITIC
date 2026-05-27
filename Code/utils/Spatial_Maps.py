import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import os
import random 
import matplotlib.gridspec as gridspec
import colorsys
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from PIL import Image
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib import rcParams
import matplotlib.patches as patches
from PIL import Image

def setup_nature_style():
    """Configuración global para todas las figuras estilo Nature"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 7,
        'axes.linewidth': 0.5,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'figure.titlesize': 9,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'xtick.minor.size': 1,
        'ytick.minor.size': 1,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'text.usetex': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def spatial_distr(ph_assign, num_fenotipos, path_to_save, embs_path):
    X_coords = []
    Y_coords = []
    phenotypes = []
    
    if num_fenotipos <= 10:
        # Para pocos fenotipos, usar una paleta más distintiva
        colors = sns.color_palette("tab10", num_fenotipos)
    elif num_fenotipos <= 20:
        # Para más fenotipos, usar una paleta extendida
        colors = sns.color_palette("tab20", num_fenotipos)
    else:
        # Para muchos fenotipos, crear una paleta personalizada
        base_colors = sns.color_palette("hsv", num_fenotipos)
        colors = []
        for i, color in enumerate(base_colors):
            # Ajustar saturación y brillo para mejor visualización
            hsv = mcolors.rgb_to_hsv(color)
            hsv[1] = min(1.0, 0.7 + (i % 3) * 0.15)  # Variar saturación
            hsv[2] = min(1.0, 0.8 + (i % 2) * 0.2)   # Variar brillo
            colors.append(mcolors.hsv_to_rgb(hsv))
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=num_fenotipos)
    
    for cell_type in range(num_fenotipos):
        for key in ph_assign.keys():
            indices_with_value = np.where(ph_assign[key] == cell_type)[0]
            key = key.split(".")[0]

            pcl_repr = np.load(embs_path+"/"+key+".tiff.npy", allow_pickle=True)
            
            for i in indices_with_value:
                row = (int(key.split("_")[1]))
                col = int(key.split("_")[2])              
                
                Y_coords.append((-1000 * row) - pcl_repr[i, 1])
                X_coords.append(pcl_repr[i, 0] + (1000 * col))
                phenotypes.append(cell_type)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.weight': 'bold',
        'figure.dpi': 500
    })

    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    scatter = plt.scatter(
        X_coords, 
        Y_coords, 
        c=phenotypes, 
        cmap=custom_cmap, 
        s=2,
        alpha=1,
        edgecolors='none',
        rasterized=True 
    )

    ax.margins(0.01, 0.01)
    
    legend_labels = [f'Nb {i+1}' for i in range(num_fenotipos)]
    
    legend_circles = [mlines.Line2D([], [], color=scatter.cmap(scatter.norm(i)), 
                                   marker='o', linestyle='None',
                                   markersize=10, label=legend_labels[i]) 
                     for i in range(num_fenotipos)]
    
    plt.legend(
        handles=legend_circles, 
        loc='upper right',
        bbox_to_anchor=(1.16, 0.95), 
        title="Neighborhoods",
        framealpha=0.7,
        edgecolor='#444444',
        title_fontsize=14,
        fontsize=12,
        markerscale=1.2 
    )
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    plt.savefig(
        path_to_save + 'spatial_distribution_new_nb.png', 
        dpi=500, 
        bbox_inches='tight',
        # facecolor=fig.get_facecolor()
    )
    
    # plt.savefig(
    #     path_to_save + 'spatial_distribution.pdf', 
    #     format='pdf',
    #     bbox_inches='tight'
    # )
    
    # plt.clf()

def spatial_distr_exp_7plex(ph_assign, num_communities, path_to_save, embs_path, cell_jer):
    colors = plt.cm.tab20(np.linspace(0, 1, num_communities))
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=num_communities)
    
    unique_images = set(key.split("_")[0] for key in ph_assign.keys())
    
    for image_id in unique_images:
        X_coords = []
        Y_coords = []
        phenotypes = []
        
        for cell_type in range(num_communities):
            for key in ph_assign.keys():
                if key.startswith(image_id):
                    indices_with_value = np.where(ph_assign[key] == cell_type)[0]
                    key_base = key.split(".")[0]
                                        
                    pcl_repr = np.load(embs_path+"/"+key, allow_pickle=True)
                    
                    for i in indices_with_value:
                        row = int(key_base.split("_")[1])
                        col = int(key_base.split("_")[2])
                        
                        Y_coords.append((-1000 * row) - pcl_repr[i, 1])
                        X_coords.append(pcl_repr[i, 0] + (1000 * col))
                        phenotypes.append(cell_type)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.style.use('dark_background')
        plt.gca().set_facecolor('black')
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.xticks([])
        plt.yticks([])
        
        scatter = plt.scatter(
            X_coords, 
            Y_coords, 
            c=phenotypes, 
            cmap=custom_cmap, 
            s=3,  
            alpha=0.8, 
            edgecolors='none'
        )
        legend_name = 'T' if cell_jer=='phenotypes' else 'Nb'
        legend_labels = [f'{legend_name} {i+1}' for i in range(num_communities)]
        
        legend_circles = [mlines.Line2D([], [], color=scatter.cmap(scatter.norm(i)), 
                                        marker='o', linestyle='None',
                                        markersize=10, label=legend_labels[i]) 
                        for i in range(num_communities)]
        
        plt.legend(
            handles=legend_circles, 
            loc='upper right',
            bbox_to_anchor=(1.1, 0.95), 
            title= 'Cell types' if cell_jer=='phenotypes' else 'Neighborhoods',
            title_fontsize=12,
            fontsize=11,
            framealpha=False  
        )
        
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(path_to_save, f'spatial_distribution_{image_id}.png'), 
            dpi=500, 
            bbox_inches='tight'
        )
    
def spatial_distr_syn(ph_assign, num_ch, num_fenotipos, path_to_save, embs_path):
    X_coords = []
    Y_coords = []
    phenotypes = []
    
    colors = plt.cm.tab20(np.linspace(0, 1, num_fenotipos))
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=num_fenotipos)
    
    num = random.randint(1,25)
    pcl_repr = np.load(embs_path+"/Multiplex_referencia_exp_2_nImage_"+str(num)+".tiff.npy", allow_pickle=True)
    
    for cell_type in range(num_fenotipos):
        indices_with_value = np.where(ph_assign["Multiplex_referencia_exp_2_nImage_"+str(num)+".tiff.npy"] == cell_type)[0]

        for i in indices_with_value:
            Y_coords.append(-pcl_repr[i, 1])
            X_coords.append(pcl_repr[i, 0])
            phenotypes.append(cell_type)

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.style.use('dark_background')
    plt.gca().set_facecolor('black')
    ax.set_facecolor('black')

    plt.xticks([])
    plt.yticks([])

    scatter = plt.scatter(
        X_coords, 
        Y_coords, 
        c=phenotypes, 
        cmap=custom_cmap, 
        s=10,  
        alpha=0.8, 
        edgecolors='none'
    )
    
    legend_labels = [f'T {i+1}' for i in range(num_fenotipos)]
    
    legend_circles = [mlines.Line2D([], [], color=scatter.cmap(scatter.norm(i)), 
                                   marker='o', linestyle='None',
                                   markersize=10, label=legend_labels[i]) 
                     for i in range(num_fenotipos)]
    
    plt.legend(
        handles=legend_circles, 
        loc='upper right',
        bbox_to_anchor=(1.1, 0.95), 
        title="Cell types",
        title_fontsize=12,
        fontsize=11,
        framealpha=False  
    )
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    plt.savefig(
        path_to_save + "spatial_distribution_referencia_" + str(num) + ".png", 
        dpi=500, 
        bbox_inches='tight',
        # facecolor=fig.get_facecolor()
    )

def compare_resolutions(base_path, embs_path, output_filename="comparison_enhanced.png"):
    resolutions = {
        'res07': 'Resolution 0.7',
        'res03': 'Resolution 0.3',
        'res05': 'Resolution 0.5',
        'res09': 'Resolution 0.9',
        'res12': 'Resolution 1.2'
    }
    
    # Generate perceptually distinct colors for better visualization
    def generate_distinct_colors(n):
        HSV_tuples = [(x/n, 0.8, 0.9) for x in range(n)]
        return list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    
    # Create figure with black background
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'savefig.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'legend.facecolor': 'black',
        'legend.edgecolor': '#333333',
        'legend.framealpha': 0.7
    })
    
    # Adjust figure size and aspect ratio to reduce empty space
    fig = plt.figure(figsize=(20, 9))
    
    # Create grid layout with better proportions and reduced spacing
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.01)
    
    # --- Main image (res_07) ---
    ax_main = fig.add_subplot(outer_gs[0])
    path_res07 = f"{base_path}/exp_43ch/43ch_res07/"
    ph_assign = np.load(path_res07 + 'nuevos_nb.npz')
    
    all_labels = np.concatenate([ph_assign[key] for key in ph_assign])
    num_fenotipos_07 = len(np.unique(all_labels))
    
    # Create custom color palette
    colors_07 = generate_distinct_colors(num_fenotipos_07)
    cmap_07 = LinearSegmentedColormap.from_list('cmap_07', colors_07, N=num_fenotipos_07)
    
    X_coords, Y_coords, phenotypes = [], [], []
    
    for cell_type in range(num_fenotipos_07):
        for key in ph_assign.keys():
            indices = np.where(ph_assign[key] == cell_type)[0]
            key_base = key.split(".")[0]
            pcl_repr = np.load(os.path.join(embs_path, key), allow_pickle=True)
            
            for i in indices:
                row = int(key_base.split("_")[1])
                col = int(key_base.split("_")[2])
                Y_coords.append((-1000 * row) - pcl_repr[i, 1])
                X_coords.append(pcl_repr[i, 0] + (1000 * col))
                phenotypes.append(cell_type)
    
    # Plot with improved aesthetics
    scatter = ax_main.scatter(X_coords, Y_coords, c=phenotypes, cmap=cmap_07, s=2, alpha=0.9)
    ax_main.text(0.95, 0.95, 'Res 0.7',
        transform=ax_main.transAxes, fontsize=15, fontweight='bold',
        ha='right', va='top', color='white',
        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    
    # Remove spines for cleaner look
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    
    # Create compact legend for main plot - position it inside the main plot to save space
    legend_handles = [
        mlines.Line2D([], [], color=cmap_07(i / num_fenotipos_07), marker='o', 
                     linestyle='None', markersize=6, label=f'Nb {i + 1}')
        for i in range(num_fenotipos_07)
    ]
    
    # Create a legend with optimal number of columns inside the plot
    ncols = min(5, num_fenotipos_07)
    leg = ax_main.legend(handles=legend_handles, loc='lower center', 
               bbox_to_anchor=(0.5, -0.02), frameon=True, ncol=ncols,
               fontsize=13, labelspacing=0.2, columnspacing=0.8, handletextpad=0.4)
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_edgecolor('#444444')
    
    # --- Four smaller images with more compact layout ---
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1], hspace=0.1, wspace=0.05)
    other_resolutions = ['res03', 'res05', 'res09', 'res12']
    
    for idx, res in enumerate(other_resolutions):
        ax = fig.add_subplot(inner_gs[idx // 2, idx % 2])
        path = f"{base_path}/exp_43ch/43ch_{res}/"
        ph_assign = np.load(path + 'nuevos_nb.npz')
        
        all_labels = np.concatenate([ph_assign[key] for key in ph_assign])
        num_fenotipos = len(np.unique(all_labels))
        
        # Create custom color palette for each resolution
        colors = generate_distinct_colors(num_fenotipos)
        cmap = LinearSegmentedColormap.from_list(f'cmap_{res}', colors, N=num_fenotipos)
        
        X_coords, Y_coords, phenotypes = [], [], []
        
        for cell_type in range(num_fenotipos):
            for key in ph_assign.keys():
                indices = np.where(ph_assign[key] == cell_type)[0]
                key_base = key.split(".")[0]
                pcl_repr = np.load(os.path.join(embs_path, key), allow_pickle=True)
                
                for i in indices:
                    row = int(key_base.split("_")[1])
                    col = int(key_base.split("_")[2])
                    Y_coords.append((-1000 * row) - pcl_repr[i, 1])
                    X_coords.append(pcl_repr[i, 0] + (1000 * col))
                    phenotypes.append(cell_type)
        
        # Plot with improved aesthetics
        scatter = ax.scatter(X_coords, Y_coords, c=phenotypes, cmap=cmap, s=0.5, alpha=0.9)
                
        ax.text(0.95, 0.95, resolutions[res].replace('Resolution', 'Res'),
            transform=ax.transAxes, fontsize=15, fontweight='bold',
            ha='right', va='top', color='white',
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Calculate optimal number of columns based on phenotype count
        ncols = min(max(4, (num_fenotipos + 3) // 2), 4)
        
        # Create a more compact legend for each resolution
        legend_handles = [
            mlines.Line2D([], [], color=cmap(i / num_fenotipos), marker='o', 
                         linestyle='None', markersize=6, label=f'Nb {i + 1}')
            for i in range(num_fenotipos)
        ]
        
        # Position legends at the bottom to save vertical space
        # Use slightly smaller font and tighter spacing for better compactness
        leg = ax.legend(handles=legend_handles, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.11), frameon=True, ncol=ncols,
                   fontsize=13, title=None, labelspacing=0.1, 
                   columnspacing=0.2, handletextpad=0.1)
        leg.get_frame().set_alpha(0.7)
        leg.get_frame().set_edgecolor('#444444')
    
    
    # Save figure with high resolution
    save_path = f"{base_path}/comparison/"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, output_filename),
                dpi=600, bbox_inches='tight', pad_inches=0.1,
                facecolor='black', edgecolor='none')
    
    # Also save a version optimized for journal requirements
    plt.savefig(os.path.join(save_path, f"journal_{output_filename}"),
                dpi=600, bbox_inches='tight', pad_inches=0.1,
                facecolor='black', edgecolor='none', format='tiff')
    
    plt.close(fig)
    
    print(f"Enhanced visualization saved to {save_path}")

def spatial_distr_exp_7plex_tmi(ph_assign, num_communities, path_to_save, embs_path, cell_jer):
    colors = plt.cm.Paired(np.linspace(0, 1, num_communities))
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=num_communities)
    
    unique_images = sorted(set(key.split("_")[0] for key in ph_assign.keys()), reverse=True)

    fig = plt.figure(figsize=(12, 5), dpi=600, facecolor="#000000")
    gs = gridspec.GridSpec(2, 5, figure=fig, wspace=0, hspace=0.)

    axs = []

    # Primer subplot ocupa 2 filas y 2 columnas
    axs.append(fig.add_subplot(gs[:, :2]))

    # Siguientes 4 subplots
    axs.append(fig.add_subplot(gs[0, 2]))
    axs.append(fig.add_subplot(gs[0, 3]))
    axs.append(fig.add_subplot(gs[1, 2]))
    axs.append(fig.add_subplot(gs[1, 3]))
    axs.append(fig.add_subplot(gs[0, 4]))
    axs.append(fig.add_subplot(gs[1, 4]))

    vecindarios_path = os.path.join(path_to_save, 'vecindarios_por_nodo_total.npz')
    fenotipos_path = 'C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Results_fenotipos/exp_7ch/exp6_7plex/res_03_28/fenotipos_por_nodo_total.npz'
    bar_image = plot_vecindario_fenotipo_bar(vecindarios_path, fenotipos_path)
    axs[6].imshow(bar_image)
    axs[6].axis('off')

    for i, image_id in enumerate(unique_images[:6]):
        X_coords = []
        Y_coords = []
        phenotypes = []
        
        for cell_type in range(num_communities):
            for key in ph_assign.keys():
                if key.startswith(image_id):
                    indices_with_value = np.where(ph_assign[key] == cell_type)[0]
                    key_base = key.split(".")[0]
                    pcl_repr = np.load(os.path.join(embs_path, key), allow_pickle=True)
                    
                    for idx in indices_with_value:
                        row = int(key_base.split("_")[1])
                        col = int(key_base.split("_")[2])
                        
                        Y_coords.append((-1000 * row) - pcl_repr[idx, 1])
                        X_coords.append(pcl_repr[idx, 0] + (1000 * col))
                        phenotypes.append(cell_type)

        ax = axs[i]
        ax.set_facecolor('#000000')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        scatter = ax.scatter(
            X_coords, 
            Y_coords, 
            c=phenotypes, 
            cmap=custom_cmap, 
            s=0.5 if i==0 else 0.2,  
            alpha=1, 
            edgecolors='none'
        )

    legend_name = 'T' if cell_jer == 'phenotypes' else 'NB'
    legend_labels = [f'{legend_name} {i+1}' for i in range(num_communities)]
    legend_circles = [
        mlines.Line2D([], [], color=custom_cmap(i / num_communities),
                      marker='o', linestyle='None',
                      markersize=10, label=legend_labels[i])
        for i in range(num_communities)
    ]
    
    fig.legend(
        handles=legend_circles,
        loc='lower center',
        ncol=num_communities,
        bbox_to_anchor=(0.5, 0.95),  # Ajusta para que quede justo a la izquierda y centrado verticalmente
        prop={'weight': 'bold', 'size':10},
        labelcolor='white',
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, 'spatial_distribution_figure_all.png'), bbox_inches='tight')

def plot_vecindario_fenotipo_bar(vecindarios_path, fenotipos_path):
    vecindarios = np.load(vecindarios_path, allow_pickle=True)
    fenotipos = np.load(fenotipos_path, allow_pickle=True)

    all_vec = np.concatenate([vecindarios[key] for key in vecindarios])
    all_fen = np.concatenate([fenotipos[key] for key in fenotipos])
    num_vecindarios = len(np.unique(all_vec))
    num_fenotipos = len(np.unique(all_fen))

    conteo = np.zeros((num_fenotipos, num_vecindarios), dtype=int)
    for key in vecindarios:
        vec = vecindarios[key]
        fen = fenotipos[key]
        for v, f in zip(vec, fen):
            conteo[f, v] += 1

    totales_vecindario = np.sum(conteo, axis=0)
    porcentajes = conteo / totales_vecindario

    colores = ['#00ff00', '#008fff', '#00ffff', '#ffff00', '#ff0000','#ff00ff', '#ffffff', '#ff7b00', '#ff7b00']

    fig, ax = plt.subplots(figsize=(2, 2), dpi=600, facecolor='black')
    fig.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.2)

    x = np.arange(num_vecindarios)
    bottom = np.zeros(num_vecindarios)

    ax.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(0.3)

    for i in range(num_fenotipos):
        heights = porcentajes[i] * 100
        bars = ax.bar(x, heights, bottom=bottom,
                      color=colores[i % len(colores)],
                      edgecolor='black', linewidth=0.2)
        
        # Añadir etiquetas solo si el porcentaje es > 20%
        for j, height in enumerate(heights):
            if height > 15:
                y_pos = bottom[j] + height / 2
                ax.text(x[j], y_pos, f'T{i+1}', ha='center', va='center',
                        color='black', fontsize=6, fontweight='bold')
        
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels([f'NB{i+1}' for i in range(num_vecindarios)], color='white', fontweight='bold', fontsize=7)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels([0, 50, 100], color='white', fontsize=7, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title("Cell types per neighborhood", color='white', fontsize=7, fontweight='bold')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # Renderizar a una imagen numpy
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()
    plt.close(fig)

    # return image

def spatial_distr_43plex(ph_assign, num_fenotipos, path_to_save, embs_path): 
    setup_nature_style()

    X_coords = []
    Y_coords = []
    phenotypes = []

    # Improved color palette - more professional and colorblind-friendly
    base_colors = ['#9467bd', '#17becf', '#FB8072', '#e377c2', 
                   '#7f7f7f', '#bcbd22', '#A6CEE3', '#B2DF8A',
                   '#FDBF6F', '#CAB2D6', '#FFFF99', '#B15928']
    colors = base_colors[:num_fenotipos]
    custom_cmap = mcolors.ListedColormap(colors)

    # Data processing (same as original)
    for cell_type in range(num_fenotipos):
        for key in ph_assign.keys():
            indices_with_value = np.where(ph_assign[key] == cell_type)[0]
            key_clean = key.split(".")[0]
            pcl_repr = np.load(f"{embs_path}/{key_clean}.tiff.npy", allow_pickle=True)

            for i in indices_with_value:
                parts = key_clean.split("_")
                row, col = (int(parts[1]), int(parts[2])) if len(parts) >= 3 else (0, 0)
                Y_coords.append((-1000 * row) - pcl_repr[i, 1])
                X_coords.append(pcl_repr[i, 0] + (1000 * col))
                phenotypes.append(cell_type)

    X_coords = np.array(X_coords)
    Y_coords = np.array(Y_coords)       
    phenotypes = np.array(phenotypes)

    # Zoom region calculation
    x_min, x_max = X_coords.min(), X_coords.max()
    y_min, y_max = Y_coords.min(), Y_coords.max()
    zoom_width = (x_max - x_min) * 0.25
    zoom_height = (y_max - y_min) * 0.25
    center_x = 7000
    center_y = -5000

    zoom_x_min = center_x - zoom_width / 2
    zoom_x_max = center_x + zoom_width / 2
    zoom_y_min = center_y - zoom_height / 2
    zoom_y_max = center_y + zoom_height / 2

    # Load image
    image_path = "C:/Users/maria.sanguesa/Desktop/43plex_rgb.tif"
    image = Image.open(image_path)
    image_np = np.array(image)
    # Obtener dimensiones de la imagen para empatar proporción
    img_height, img_width = image_np.shape[:2]
    img_aspect = img_height / img_width

    # Calcular centro de la nube de puntos
    x_min, x_max = X_coords.min(), X_coords.max()
    y_min, y_max = Y_coords.min(), Y_coords.max()
    x_center = (x_max + x_min) / 2

    # Ajustar rango X como está
    data_width = x_max - x_min
    # Ajustar Y en base a aspecto de imagen
    data_height = data_width * img_aspect

    # Calcular nuevos límites Y centrados
    y_center = (y_max + y_min) / 2
    y_min = y_center - data_height / 2
    y_max = y_center + data_height / 2
    
    # Improved figure layout - con espacio para leyendas reducido
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 0.03, 0.12], width_ratios=[1, 1], hspace=0.05, wspace=0.15)

    # Main plots
    ax1 = fig.add_subplot(gs[0, 0])  # Original image
    ax3 = fig.add_subplot(gs[0, 1])  # Full scatter
    
    ax_legend_A = fig.add_axes([0.20, 0.15, 0.05, 0.1])  # [x, y, width, height]
    ax_legend_A.axis('off')
    ax_legend_B = fig.add_axes([0.60, 0.15, 0.05, 0.1])
    ax_legend_B.axis('off')

    # Panel A: Original image with zoom box and inset
    ax1.imshow(image_np, aspect='equal')  # Force equal aspect    
    # Improved zoom box styling
    rect = FancyBboxPatch((1121, 2247), 2869 - 1121, 3745 - 2247,
                         boxstyle="round,pad=5", 
                         linewidth=2.5, edgecolor='black', 
                         facecolor='none', alpha=0.9,
                         linestyle='-')
    ax1.add_patch(rect)
    ax1.axis('off')

    # Add scale bar to original image
    scale_bar_length = 500  # pixels, adjust based on your image scale
    scale_bar_y = image_np.shape[0] - 150
    scale_bar_x = 150
    ax1.plot([scale_bar_x, scale_bar_x + scale_bar_length], 
             [scale_bar_y, scale_bar_y], 'w-', linewidth=4)

    # Create inset for zoomed image (top-right corner)
    zoom_img = image_np[2247:3745, 1121:2869]
    
    # Position inset in top-right corner
    inset_ax1 = ax1.inset_axes([0.45,-0.3, 0.7, 0.7/1.17])
    inset_ax1.imshow(zoom_img, aspect='equal')
    inset_ax1.axis('off')
    inset_ax1.set_aspect('equal')
    
    # Add scale bar to zoomed inset
    zoom_scale_length = 100  # adjust based on zoom level
    zoom_scale_y = zoom_img.shape[0] - 80
    zoom_scale_x = 80
    inset_ax1.plot([zoom_scale_x, zoom_scale_x + zoom_scale_length], 
                   [zoom_scale_y, zoom_scale_y], 'w-', linewidth=3)

    # Panel B: Full spatial distribution with inset - FONDO NEGRO
    ax3.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax3.transAxes, facecolor='white'))

    # Make sure both plots have the same physical size by setting equal aspect and limits
    scatter = ax3.scatter(X_coords, Y_coords, c=phenotypes, cmap=custom_cmap, 
                         s=0.7, alpha=1, rasterized=True, edgecolors='none')
    data_width = x_max - x_min
    data_height = y_max - y_min
    ax3.set_box_aspect(data_height / data_width)
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.patch.set_visible(True)
    ax3.set_aspect('equal')
    
    # Improved zoom box for scatter plot
    rect = FancyBboxPatch((zoom_x_min, zoom_y_min), 
                         zoom_x_max - zoom_x_min, zoom_y_max - zoom_y_min,
                         boxstyle="round,pad=50", 
                         linewidth=2.5, edgecolor='BLACK', 
                         facecolor='none', alpha=1,
                         linestyle='-')
    ax3.add_patch(rect)
    ax3.axis('off')
    ax3.set_aspect('equal')

    # Create inset for zoomed scatter (top-right corner) - FONDO NEGRO
    inset_ax2 = ax3.inset_axes([0.45,-0.3, 0.7, 0.7/1.17])
    
    inset_ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=inset_ax2.transAxes, facecolor='white'))

    inset_ax2.patch.set_visible(True)
    mask = ((X_coords >= zoom_x_min) & (X_coords <= zoom_x_max) & 
            (Y_coords >= zoom_y_min) & (Y_coords <= zoom_y_max))
    inset_ax2.scatter(X_coords[mask], Y_coords[mask], 
                     c=phenotypes[mask], cmap=custom_cmap, 
                     s=3, alpha=1, rasterized=True, edgecolors='none')
    inset_ax2.set_xlim(zoom_x_min, zoom_x_max)
    inset_ax2.set_ylim(zoom_y_min, zoom_y_max)
    inset_ax2.set_aspect('equal', adjustable='box')
    inset_ax2.axis('off')

    # Leyendas en subplots separados pero más cerca
    unique_phenotypes = np.unique(phenotypes)
    
    # Legend A: Simple scale information - en subplot separado
    legend_labels = ['CD21', 'CD31', 'CD20', 'CD4', 'PanCK', 'CD8', 'SMA', 'DAPI']
    legend_colors = ['#00ffff', '#00ff00', '#ff7d00', '#ff0000', '#ffffff', '#ffff00','#ff00ff', '#0000ff']
    legend_elements_A = []
    for i, marker in enumerate(legend_labels):
        legend_elements_A.append(Patch(facecolor=legend_colors[i], edgecolor='black', label=marker, linewidth=0.5))

    ax_legend_A.legend(handles=legend_elements_A, 
                    loc='upper center', 
                    ncol=2,
                    frameon=False, 
                    columnspacing=1,
                    handletextpad=0.4,
                    prop={'weight': 'bold', 'size':8})
    
    # Legend B: Phenotype color coding - en subplot separado
    legend_elements = []
    phenotype_labels = [f'NB {i+1}' for i in unique_phenotypes]  # Customize these labels
    for i, pheno in enumerate(unique_phenotypes):
        color = colors[pheno % len(colors)]
        legend_elements.append(plt.Line2D([], [], marker='o', color='w', 
                                        markerfacecolor=color, markersize=9,
                                        markeredgewidth=0.5, linestyle='None', markeredgecolor='black', label=phenotype_labels[i]))
    
    ax_legend_B.legend(handles=legend_elements, 
                                 loc='upper center', 
                                 ncol=min(3, len(unique_phenotypes)),
                                 frameon=False, 
                                 columnspacing=0.7,
                                 handletextpad=0.2,
                                 prop={'weight': 'bold', 'size':8})
    

    # Save with high quality settings
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    plt.savefig(path_to_save + 'spatial_zoom_improved.png', 
                bbox_inches='tight', 
                edgecolor='none', 
                dpi=600, 
                pad_inches=0)
    plt.close()

def plot_vecindario_fenotipo_bar_43plex(vecindarios_path, fenotipos_path):
    setup_nature_style()

    vecindarios = np.load(vecindarios_path, allow_pickle=True)
    fenotipos = np.load(fenotipos_path, allow_pickle=True)

    all_vec = np.concatenate([vecindarios[key] for key in vecindarios])
    all_fen = np.concatenate([fenotipos[key] for key in fenotipos])
    num_vecindarios = len(np.unique(all_vec))
    num_fenotipos = len(np.unique(all_fen))

    conteo = np.zeros((num_fenotipos, num_vecindarios), dtype=int)
    for key in vecindarios:
        vec = vecindarios[key]
        fen = fenotipos[key]
        for v, f in zip(vec, fen):
            conteo[f, v] += 1

    totales_vecindario = np.sum(conteo, axis=0)
    porcentajes = conteo / totales_vecindario

    base_colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    additional_colors = sns.color_palette("Set3", 46)  # Para completar hasta 56
    colores_fenotipos = base_colors + list(additional_colors)
    color_otros = "#CCCCCCCA"

    top_k = 10
    conteo_agrupado = np.zeros((top_k + 1, num_vecindarios), dtype=int)
    fenotipo_indices_top = []

    for v in range(num_vecindarios):
        conteo_v = conteo[:, v]
        top_idx = np.argsort(conteo_v)[::-1][:top_k]
        fenotipo_indices_top.append(top_idx)
        
        conteo_agrupado[:top_k, v] = conteo_v[top_idx]
        conteo_agrupado[top_k, v] = np.sum(conteo_v) - np.sum(conteo_v[top_idx])

    porcentajes_agrupado = conteo_agrupado / np.sum(conteo_agrupado, axis=0)

    # Crear figura con proporciones más elegantes
    fig, ax = plt.subplots(figsize=(1.5, 3.5), dpi=600)
    fig.patch.set_facecolor('white')
    
    x = np.arange(num_vecindarios)
    bottom = np.zeros(num_vecindarios)
    width =0.85  # Barras más estrechas y elegantes

    
    for i in range(top_k + 1):
        heights = porcentajes_agrupado[i] * 100
        
        bar_colors = []
        for j in range(num_vecindarios):
            if i < top_k:
                fen_idx = fenotipo_indices_top[j][i]
                bar_colors.append(colores_fenotipos[fen_idx])
            else:
                bar_colors.append(color_otros)

        bars = ax.bar(x, heights, bottom=bottom, width=width,
                     color=bar_colors, edgecolor='white', linewidth=0.5,
                     alpha=0.85)
        
        # Añadir etiquetas solo para segmentos grandes (>3%)
        for j, height in enumerate(heights):
            if height > 1.0:  # Solo etiquetar segmentos significativos
                y_pos = bottom[j] + height / 2
                label = f'T{fenotipo_indices_top[j][i]+1}' if i < top_k else 'Other'
                
                # Color del texto adaptativo
                # text_color = 'white' if height > 10 else 'black'
                text_color = 'black'
                ax.text(x[j], y_pos, label, ha='center', va='center',
                       color=text_color, fontsize=4, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='none', 
                                edgecolor='none', alpha=0.7))

        bottom += heights

    # Configurar ejes con estilo profesional
    ax.set_xticks(x)
    ax.set_xticklabels([f'NB {i+1}' for i in range(num_vecindarios)],
                      color='#333333', fontweight='bold')
    
    # Mejorar los ticks del eje Y
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100'], 
                      color='#333333', fontsize=5, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Grid sutil para mejor lectura
    ax.grid(True, axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.set_ylabel("Cell Type Abundance (%)", fontweight='bold', fontsize=5)
    
    # Configurar ticks
    ax.tick_params(axis='both', which='major', labelsize=5, 
                   colors='#333333', width=0.3, length=1)
    ax.tick_params(axis='both', which='minor', width=0.3, length=1)
    
    plt.savefig('C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Results_fenotipos/exp_43ch/43ch_res25/bar_per_ph_enhanced.png', 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none', 
                dpi=600, 
                format='png')

    plt.close()

def spatial_distr_7plex_ph_nb(ph_assign, num_fenotipos, path_to_save, embs_path, nb_assign, num_vecindarios): 
    setup_nature_style()

    X_coords, Y_coords, phenotypes, neighborhoods = [], [], [], []

    base_colors_ph = ['#00ff00','#008fff' ,'#00ffff' , '#ffff00','#ff0000' , '#ff00ff', '#ffffff', '#ff7b00']
    colors_ph = base_colors_ph[:num_fenotipos]
    custom_cmap_ph = mcolors.ListedColormap(colors_ph)

    base_colors_nb = ['#A6CEE3', '#33A02C', '#FFC874', '#61388E', '#B15928']
    colors_nb = base_colors_nb[:num_vecindarios]
    custom_cmap_nb = mcolors.ListedColormap(colors_nb)

    for cell_type in range(num_fenotipos):
        for key in ph_assign.keys():
            if 'Ctrl' in key:
                indices = np.where(ph_assign[key] == cell_type)[0]
                key_clean = key.split(".")[0]
                pcl_repr = np.load(f"{embs_path}/{key_clean}.tiff.npy", allow_pickle=True)
                for i in indices:
                    parts = key_clean.split("_")
                    row, col = (int(parts[1]), int(parts[2])) if len(parts) >= 3 else (0, 0)
                    Y_coords.append((-1000 * row) - pcl_repr[i, 1])
                    X_coords.append(pcl_repr[i, 0] + (1000 * col))
                    phenotypes.append(cell_type)
                    neighborhoods.append(nb_assign[key][i])

    X_coords, Y_coords = np.array(X_coords), np.array(Y_coords)
    phenotypes, neighborhoods = np.array(phenotypes), np.array(neighborhoods)

    x_min, x_max = X_coords.min(), X_coords.max()
    y_min, y_max = Y_coords.min(), Y_coords.max()
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

    zoom_factor = 0.4
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_zoom_min = x_center - (x_range * 0.25 )
    x_zoom_max = x_center + (x_range * 0.25 )
    y_zoom_min = y_center - (y_range * 0.25 )
    y_zoom_max = y_center + (y_range * 0.25 )

    rgb_image = Image.open("C:/Users/maria.sanguesa/Desktop/Ctrl_RGB.tif")
    rgb_array = np.array(rgb_image)

    img_height, img_width = rgb_array.shape[:2]
    crop_x_min = int(img_width * 0.25)
    crop_x_max = int(img_width * 0.75)
    crop_y_min = int(img_height * 0.25)
    crop_y_max = int(img_height * 0.75)
    rgb_cropped = rgb_array[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    fig = plt.figure(figsize=(16, 8), facecolor='black')
    spec = gridspec.GridSpec(ncols=3, nrows=1, wspace=0.02)

    ax_rgb = fig.add_subplot(spec[0, 0], facecolor='black')
    ax_rgb.imshow(rgb_cropped)
    ax_rgb.axis('off')

    rgb_labels = ['CD21','CD23','CD20','CD4','CK','CD8','DAPI']
    rgb_colors = ['#00ffff','#00ff00','#ff7d00','#ff0000','#ffffff','#ffff00','#0000ff']
    legend_patches = [Patch(color=c, label=l) for c, l in zip(rgb_colors, rgb_labels)]
    ax_rgb.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=7, fontsize=8, frameon=False, labelcolor='white', labelspacing=0.01, columnspacing=0.4)

    ax_main = fig.add_subplot(spec[0,1], facecolor='black')
    x_min, x_max = X_coords.min(), X_coords.max()
    y_min, y_max = Y_coords.min(), Y_coords.max()
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

    ax_main.set_xlim(x_zoom_min, x_zoom_max)
    ax_main.set_ylim(y_zoom_min, y_zoom_max)
    ax_main.set_aspect('equal')
    ax_main.axis('off')

    scatter_nb = ax_main.scatter(X_coords, Y_coords, c=neighborhoods, cmap=custom_cmap_nb, s=2, alpha=1, rasterized=True, edgecolors='none', zorder=1)

    ph_width = (x_zoom_max - x_zoom_min) * 0.5
    ph_height = (y_zoom_max - y_zoom_min) * 0.5
    ph_x = x_zoom_min + (x_zoom_max - x_zoom_min) * 0.1
    ph_y = y_zoom_min + (y_zoom_max - y_zoom_min) * 0.05

    mask_ph = (X_coords > ph_x) & (X_coords < ph_x + ph_width) & \
              (Y_coords > ph_y) & (Y_coords < ph_y + ph_height)

    scatter_ph = ax_main.scatter(X_coords[mask_ph], Y_coords[mask_ph], c=phenotypes[mask_ph],
                                 cmap=custom_cmap_ph, s=2, alpha=1, rasterized=True, edgecolors='none', zorder=2)

    rect = Rectangle((ph_x, ph_y), ph_width, ph_height, linewidth=1, edgecolor='white', facecolor='none', zorder=10)
    ax_main.add_patch(rect)

    nb_names = [f"NB {i+1}" for i in range(num_vecindarios)]
    nb_patches = [Patch(color=c, label=l) for c, l in zip(colors_nb, nb_names)]
    ax_nb_leg = fig.add_subplot(spec[0, 1])
    ax_nb_leg.axis('off')
    ax_nb_leg.legend(handles=nb_patches, loc='lower center', bbox_to_anchor=(0.5, 0.065), ncol=num_vecindarios, fontsize=8, frameon=False, labelcolor='white')

    pheno_names = [f"T {i+1}" for i in range(num_fenotipos)]
    ph_patches = [Patch(color=c, label=l) for c, l in zip(base_colors_ph[:num_fenotipos], pheno_names)]
    ax_ph_leg = fig.add_subplot(spec[0, 1])
    ax_ph_leg.axis('off')
    ax_ph_leg.legend(handles=ph_patches,loc='center', bbox_to_anchor=(1.06, 0.5), columnspacing=5, labelspacing=4, fontsize=8, frameon=False, labelcolor='white')

    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    plt.savefig(path_to_save + 'spatial_ph_nb_Ctrl.png', bbox_inches='tight', edgecolor='none', dpi=600, pad_inches=0)
    plt.close()
 
if __name__ == "__main__":
    embs_path = "C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/embs_exp6_7plex"
    num_fenotipos = 8
    num_vecindarios = 5
    path = "C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Results_vecindarios/exp_7ch/exp6_7plex/res_05/"
    ph_assign = np.load('C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Results_fenotipos/exp_7ch/exp6_7plex/res_03_28/fenotipos_por_nodo_total.npz')
    nb_assign = np.load('C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Results_vecindarios/exp_7ch/exp6_7plex/res_05/vecindarios_por_nodo_total.npz')
    spatial_distr_7plex_ph_nb(ph_assign, num_fenotipos, path, embs_path, nb_assign, num_vecindarios)
    # plot_vecindario_fenotipo_bar_43plex('C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Results_vecindarios/exp_43ch/res_05_2/vecindarios_por_nodo_total.npz', 'C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Results_fenotipos/exp_43ch/43ch_res25/fenotipos_por_nodo_total.npz')

    # embs_path = "C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/embs_43ch_Norm"
    # num_fenotipos = 9
    # base_path = "C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Results_vecindarios" 
    # compare_resolutions(base_path, embs_path)
    # cluster =1

    # cell_jer = 'phenotypes' if cluster==0 else 'neighborhoods'
    # embs_path = "C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/embs_exp6_7plex"
    # num_clusters = 5
    # path = f"C:/Users/maria.sanguesa/OneDrive - UPNA/Nb_GNN/Results_vecindarios/exp_7ch/exp6_7plex/res_05/"
    # ph_assign = np.load(path+f'vecindarios_por_nodo_total.npz')

    # # # spatial_distr_syn(ph_assign, num_ch, num_fenotipos,path, embs_path)
    # # spatial_distr_exp_7plex(ph_assign, num_clusters, path, embs_path, cell_jer)
    # spatial_distr_exp_7plex_tmi(ph_assign, num_clusters, path, embs_path, cell_jer)





    