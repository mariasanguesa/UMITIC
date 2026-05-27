import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Code.Patch_Contrastive_Learning.utils import save_config_file, accuracy
import pandas as pd
import numpy as np
from PIL import Image
import gc 
from openTSNE import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import sklearn
gc.set_threshold(0,0,0)

torch.manual_seed(0)

def visualize_using_clustering_ALL(dataset, out_dir, features, features_idx, patch_images):

    # Cluster image using all information.
    clusters_ALL = sklearn.cluster.KMeans(n_clusters=5).fit(features[:,2:])
    
    # Calculate image size
    for im_idx, name in enumerate(patch_images):
        
        # # CLusters all
        sel_feats = features[features_idx==im_idx]
        image_size = sel_feats[:,:2].max(0)+sel_feats[:,:2].min(0)
        patch_size = int((sel_feats[:,:2].min(0)*2)[0])
        sel_clusters_all = clusters_ALL.labels_[features_idx==im_idx]

        # CLustering of this image
        clusters_im = sklearn.cluster.KMeans(n_clusters=5).fit(sel_feats[:,2:])
        
        # Create patch Image
        Patch_im = np.zeros((int(image_size[0]),int(image_size[1])))                        
        # Create superPatch Label image using the Superpatch size. 
        division_rows = np.floor(Patch_im.shape[0]/patch_size)
        division_cols = np.floor(Patch_im.shape[1]/patch_size)
        lins = np.repeat(list(range(int(division_cols))), patch_size)
        lins = np.repeat(np.expand_dims(lins,axis=1), patch_size, axis=1)
        for row_indx, row in enumerate(range(int(division_rows))):
            Patch_im[row_indx*patch_size:(row_indx+1)*patch_size,:int(division_cols*patch_size)] = np.transpose(lins+int(row_indx*division_cols))
        Patch_im = Patch_im.astype(int)

        # Cluster into distinct phenotypes
        Cluster_image_ALL = sel_clusters_all[Patch_im]
        Cluster_image = clusters_im.labels_[Patch_im]

        # Save cluster image
        imtosave = Image.fromarray(np.uint8(Cluster_image_ALL))
        imtosave.save(out_dir+name+'_cluster_ALL.png') 

        # Save cluster image
        imtosave = Image.fromarray(np.uint8(Cluster_image))
        imtosave.save(out_dir+name+'_cluster.png') 

def visualize_batch_effect_Ph(out_dir,features,features_idx,count):
    # Select n subjects randomly
    num_subjects = 5
    sel_subjects = sum(([features_idx==c for c in np.random.randint(0, high=features_idx.max(),size=num_subjects)]))>0
    features_idx = features_idx[sel_subjects]
    features = features[sel_subjects]
    
    # Select number of patches
    num_val = min(100000,features_idx.shape[0]*0.9)
    features_idx = features_idx[::int(features_idx.shape[0]/num_val)]
    features = features[::int(features.shape[0]/num_val),:]

    # Reorder samples
    sel_vals = np.random.randint(0,high=features.shape[0],size=features.shape[0])
    features = features[sel_vals]
    features_idx = features_idx[sel_vals]

    # Normalize features
    scaled_features = StandardScaler().fit_transform(features[:,2:])

    # Create UMAP
    embedding = TSNE(n_jobs=8,random_state=3, verbose=True).fit(scaled_features)

    # Create and save scatter plot figure
    plt.close('all')
    fig, ax = plt.subplots()   
    scatter = ax.scatter(embedding[:, 0],embedding[:, 1],s=0.5,c=features_idx,cmap='jet',edgecolors='none')
    legend1 = ax.legend(*scatter.legend_elements(),loc="upper right", title="Subjects")
    ax.add_artist(legend1)
    plt.title('TSNE of the dataset - one pat', fontsize=24)    
    plt.savefig(out_dir+'/TSNE_embeddings_Patients_'+str(count)+'.png',dpi=600)

    return

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args['GPU_INDEX'])
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']            
        self.model_dir = self.args['path']+'Patch_Contrastive_Learning/Model/'        
        self.writer = SummaryWriter(log_dir=self.model_dir)    
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)  
        logging.basicConfig(filename=os.path.join(self.model_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args['GPU_INDEX'])

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(features.shape[0]/2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args['GPU_INDEX'])

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args['GPU_INDEX'])
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args['GPU_INDEX'])

        logits = logits / self.args['CNN_Temperature']
        return logits, labels

    def train(self, train_loader):
    
        scaler = GradScaler(enabled=True)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        # Train from previous checkpoint
        text_files = [f for f in os.listdir(self.model_dir) if f.endswith('.tar')]        
        if len(text_files)>0:
            text_files.sort()
            self.model.load_state_dict(torch.load(os.path.join(self.model_dir, text_files[-1]), weights_only=True))
            self.model.train()
            done_epochs = len(text_files)*self.args['CNN_Epochs']/10
        else:
            done_epochs = 0

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args['CNN_Epochs']} epochs.")        

        for epoch_counter in tqdm(range(self.args['CNN_Epochs']),initial=int(done_epochs)):
            epoch_counter += int(done_epochs) 
            for images in train_loader:
                images[0] = torch.cat(torch.tensor_split(images[0], images[0].shape[0]),1).squeeze()
                images[1] = torch.cat(torch.tensor_split(images[1], images[1].shape[0]),1).squeeze()

                images = torch.cat((images[0],images[1]),dim=0)

                images = images.to(torch.float32)

                images = images.to(self.args['GPU_INDEX'])

                images = torch.moveaxis(images, 3, 1)

                with autocast(device_type='cuda', enabled=True):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # Log info
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                self.writer.add_scalar('loss', loss, global_step=n_iter)
                self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1                
            
                del images, features
                gc.collect(0)
                gc.collect(1)
                gc.collect(2)
                
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tTop5 accuracy: {top5[0]}")

            # warmup for the first 10 epochs
            if epoch_counter >= 11*(self.args['CNN_Epochs']/10):
                self.scheduler.step()
            
            # save model checkpoints
            if epoch_counter % int(self.args['CNN_Epochs']/10) == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, checkpoint_name))        
                logging.info(f"Model checkpoint and metadata has been saved at {self.model_dir}.")

            if epoch_counter>=self.args['CNN_Epochs']:
                break

        logging.info("Training has finished.")
        

    def infer(self,infer_loader, out_dir):
        
        # Open last saved model checkpoint          
        text_files = [f for f in os.listdir(self.model_dir) if f.endswith('.tar')]
        text_files.sort()
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, text_files[-1])))
        del self.model.backbone.fc._modules['4']
        del self.model.backbone.fc._modules['3']
        self.model.eval()       

        self.folder = out_dir 

        # Order Images 
        infer_loader.dataset.image_list.sort()
        
        # Extract patches and save it into a graph
        for patches, patch_pos, filename, diags, ecc, comp, pol, nucl, nucl_size, cell_size, mean_ch in tqdm(infer_loader):
        # for patches, patch_pos, filename, diags in tqdm(infer_loader):                                     
            # Locate model to device.
            self.model = self.model.to(self.args['GPU_INDEX'])
            
            # Minibatch to extract features
            step = self.args['N_Crops_per_Image']*self.args['CNN_Batch_Size']
            features = []
            patch_pos = np.squeeze(patch_pos)
            mean_ch = np.squeeze(mean_ch)
            for p in range(0,len(patches),step):
                if len(patches[p:p+step])>1:
                    features.append(self.model(torch.moveaxis(torch.squeeze(torch.tensor(np.stack(patches[p:p+step],0),device=self.args['GPU_INDEX'])),3,1).to(torch.float32)).to('cpu').detach().numpy()) 
                elif len(patches[p:p+step])==1:
                    patch_pos = np.delete(patch_pos,-1,axis=0)
                    diags = np.delete(diags,-1)
                    ecc = np.delete(ecc,-1)
                    comp = np.delete(comp,-1)
                    pol = np.delete(pol,-1)
                    nucl = np.delete(nucl,-1)
                    nucl_size = np.delete(nucl_size,-1)
                    cell_size = np.delete(cell_size,-1)
                    mean_ch = np.delete(mean_ch,-1,axis=0)

            features = np.concatenate(features,axis=0)

            ecc = np.array([t.item() for t in ecc]).reshape(len(ecc),1)
            features = np.concatenate((ecc,features),axis=1)

            comp = np.array([t.item() for t in comp]).reshape(len(comp),1)
            features = np.concatenate((comp,features),axis=1)

            pol = np.array([t.item() for t in pol]).reshape(len(pol),1)
            features = np.concatenate((pol,features),axis=1)                        

            nucl = np.array([t.item() for t in nucl]).reshape(len(nucl),1)
            features = np.concatenate((nucl,features),axis=1)

            nucl_size = np.array(nucl_size).reshape(len(nucl_size),1)
            features = np.concatenate((nucl_size.astype('float32'),features),axis=1)

            cell_size = np.array(cell_size).reshape(len(cell_size),1)
            features = np.concatenate((cell_size.astype('float32'),features),axis=1)

            features = np.concatenate((mean_ch, features),axis=1)

            diags = np.array([t.item() for t in diags]).reshape(len(diags),1)
            features = np.concatenate((diags,features),axis=1)

            features = np.concatenate((patch_pos,features),axis=1)    

            # Join several images to one numpy structure
            patient_to_image = [i for i in os.listdir(self.args['path']+'Raw_Data/Experiment_Information/') if 'Patient_to_Image.xlsx'==i]
            if len(patient_to_image)>0:
                
                # Find subject name and subject file
                patient_to_image_excel = pd.read_excel(self.args['path']+'Raw_Data/Experiment_Information/'+patient_to_image[0])
                index = list(patient_to_image_excel['Image_Name']).index(filename[0])
                NPYname = self.folder+patient_to_image_excel['Subject_Name'][index]+'.npy'
                
                # if already exists...                 
                if os.path.isfile(NPYname):
                    
                    # Load previous concatenate with actual data and save it.
                    prevFeats = np.load(NPYname, allow_pickle=True)
                    features[:,:2] = features[:,:2] + prevFeats[:,:2].max() + self.args['Crop_Size']*10 
                    np.save(NPYname,np.concatenate((prevFeats,features),axis=0))      
                    del prevFeats           

                else:
                    # Create new file
                    np.save(NPYname,features)    
                
            # Save one image is one subject
            else:
                np.save(self.folder+filename[0],features)    

            # Eliminate data from garbage collector
            del patches, patch_pos, filename, features, diags 
            gc.collect(0)
            gc.collect(1)      
            gc.collect(2)      

        return 


    def cluster_patches(self, patch_dir, out_dir):
        # Patches from subjects
        patch_images = os.listdir(patch_dir)        

        features = []
        features_idx = []

        # Extract patches and save it into a graph
        for subj_idx, filename in enumerate(patch_images):                                    
            
            # Load image 
            subject_feats = np.load(patch_dir+filename, allow_pickle=True)

            # Store feats
            features.append(subject_feats)
            
            # Store subject idx
            features_idx.append([subj_idx]*subject_feats.shape[0])

        # Store in one file all features
        features = np.concatenate(features,axis=0)
        features_idx = np.concatenate(features_idx,axis=0)        

        # Show batch effect        
        visualize_batch_effect_Ph(out_dir,features,features_idx,0)

        # Cluster image 
        # visualize_using_clustering_ALL(self, out_dir, features, features_idx, patch_images)    

        return 