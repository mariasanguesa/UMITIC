import xlsxwriter
from Code.utils.plot_confusion_matrix import plot_confusion_matrix_from_data
from sklearn import metrics, linear_model
from Code.utils.modules import *
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import copy
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations 
import pandas as pd

def load_channels(base_path):
    # User-defined channels to use
    with open(base_path+'Experiment_Information/Channels.txt','r') as f:
        Channels = f.read().split('\n')
    
    Channels = [m for n,m in enumerate(Channels) if m!='']

    Marker_Names = copy.deepcopy(Channels)

    # If the user defines one of the values to None, it is not included.
    Channels = [n for n,m in enumerate(Channels) if m!='None']
    Marker_Names = [m for n,m in enumerate(Marker_Names) if m!='None']
    return Channels, Marker_Names


def survival_grid(IndexAndClass,num_clusters):
    indexForSurvival=[[i[0],i[1],i[2]] for i in IndexAndClass if i[1]>-1]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.reshape(np.asarray([i[2] for i in indexForSurvival]),(len(indexForSurvival),1)))
    for n,i in enumerate(indexForSurvival):
        if i[2]>-1:
            IndexAndClass[n].append(kmeans.labels_[n])
            IndexAndClass[n][2] = kmeans.cluster_centers_[IndexAndClass[n][3]][0]
        else:
            IndexAndClass[n].append(-1)
    return IndexAndClass # Last Value is cluster Center, and second-to-last is label.

def nll_loss(hazards, Y, c, device, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    
    Y = Y.unsqueeze(dim=1)# ground truth bin, 1,2,...,k Y.view(batch_size, 1) 
    if c is None:
        c = torch.zeros(Y.shape[0],device=device)
    c = c.unsqueeze(dim=1)# censorship status, 0 for alive or 1 for dead    
    # c = c.view(batch_size, 1).float() 
    S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # without padding, S(0) = S[0], h(0) = h[0]
    # S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss    
    return loss.mean()


def MSE_loss(out,y,device,meanSTD_ylabel):    
    y_zsco = (y-torch.tensor(meanSTD_ylabel[0]).to(device))/torch.tensor(meanSTD_ylabel[1]).to(device)
    out = (torch.exp((y_zsco - out).abs())-1)
    out = torch.log(1+out.sum())    
    return out

def MSE_pred(out,y,device,meanSTD_ylabel):    
    out = (out.cpu().detach().numpy()*meanSTD_ylabel[1])+meanSTD_ylabel[0] # RE-normalize
    return out

# def survival_loss_(out,y,device):
#     alpha=0.2
#     sigma=0.1
#     nll = nll_pc_hazard_loss(out, y[:,0].long(), torch.ones(out.shape[0],device=device), y[:,1], 'mean')
#     rank_loss = rank_loss_deephit_single(out, y[:,0].long(), torch.ones(out.shape[0],device=device), torch.ones(out.shape[0],out.shape[0],device=device), sigma,'mean')
#     return alpha * nll + (1. - alpha) * rank_loss

def nearestNeighbor_loss(training, args, out, y, NNList,index, indices, labels,device,model):
    # Here we select whether we are training the clusters in a unsupervised/supervised way    
    # Normalize 
    out=out-out.mean(-2)
    out=out/out.norm(p=2)
    # KNN
    out = out*10
    # Update positions:
    NNList=NNList.detach()    
    try:
        NNList[indices[index:min(index+args['batch_size'],len(indices))]]=out
    except:
        print('Error somewhere')
    PredictedLabels = np.zeros((out.shape[0],args['num_classes']), dtype=np.uint)
    labels=torch.tensor(labels,device=device)
    yCentroid=out.clone()
    yNOTCentroid=out.clone()
    pred = np.zeros((out.shape[0]))
    for b in range(out.shape[0]):
        topkVal = torch.topk(-(out[b,:]-NNList).pow(2).sum(-1), k=min(args['KinNearestNeighbors'],NNList.shape[0]))[1]            
        aux = torch.argsort((labels[topkVal[0]]-labels[topkVal[1:]]).abs()) #out[torch.nonzero(topkVal[0]-topkVal[1:])+1,:]
        # auxx = out[torch.nonzero(topkVal[0]==topkVal[1:])+1,:]
        yCentroid[b,:] = NNList[topkVal[aux[0]+1],:] # yCentroid[b,:] if len(aux)==0 else aux.mean(dim=0)
        yNOTCentroid[b,:] = NNList[topkVal[aux[-1]+1],:] # yNOTCentroid[b,:] if len(auxx)==0 else auxx.mean(dim=0)
        countVal = np.asarray([list(labels[topkVal[1:]].cpu().numpy()).count(i) for i in range(args['num_classes'])])
        pred[b] = countVal.argmax()
        PredictedLabels[b] = countVal/countVal.sum()
    loss = F.mse_loss(out,yCentroid) - F.mse_loss(out,yNOTCentroid)    
    loss=torch.exp(-model.NNClsfctionMTLearning)*(loss**2)+model.NNClsfctionMTLearning-1
    print('Nearest Neighbor, Y Same Class',F.mse_loss(out,yCentroid).item(),'Y different class:',F.mse_loss(out,yNOTCentroid).item())
    return loss, pred, PredictedLabels, NNList

    # # Here we select whether we are training the clusters in a unsupervised/supervised way    
    # # Normalize 
    # out=out-out.mean(-2)
    # out=out/out.norm(p=2)
    # # KNN
    # out = out*10
    # # Update positions:
    # # NNList[indices[index:min(index+args['batch_size'],len(NNList))]]=out
    # PredictedLabels = np.zeros((out.shape[0],args['num_classes']), dtype=np.uint)
    # labels=torch.tensor(labels,device=device)
    # yCentroid=out.clone()
    # yNOTCentroid=out.clone()
    # pred = np.zeros((out.shape[0]))
    # for b in range(out.shape[0]):
    #     topkVal = torch.topk(-(out[b,:]-out).pow(2).sum(-1), k=min(args['KinNearestNeighbors'],out.shape[0]))[1]            
    #     aux = torch.argsort((labels[topkVal[0]]-labels[topkVal[1:]]).abs()) #out[torch.nonzero(topkVal[0]-topkVal[1:])+1,:]
    #     # auxx = out[torch.nonzero(topkVal[0]==topkVal[1:])+1,:]
    #     yCentroid[b,:] = out[topkVal[aux[0]+1],:] # yCentroid[b,:] if len(aux)==0 else aux.mean(dim=0)
    #     yNOTCentroid[b,:] = out[topkVal[aux[-1]+1],:] # yNOTCentroid[b,:] if len(auxx)==0 else auxx.mean(dim=0)
    #     countVal = np.asarray([list(labels[topkVal[1:]].cpu().numpy()).count(i) for i in range(args['num_classes'])])
    #     pred[b] = countVal.argmax()
    #     PredictedLabels[b] = countVal/countVal.sum()
    # loss = F.mse_loss(out,yCentroid) - F.mse_loss(out,yNOTCentroid)
    # print(F.mse_loss(out,yCentroid))
    # print(F.mse_loss(out,yNOTCentroid))
    # return loss, pred, PredictedLabels, NNList

def cross_entropy_loss(training, Train_indices_for_pred, args, out, y, dataset, device,model):
    # Here we select whether we are training the clusters in a unsupervised/supervised way
    
    loss = []
    pred = []
    PredictedLabels = []
    for i in range(len(out)): 
      
        loss.append(F.cross_entropy(out[i][Train_indices_for_pred,:], y[Train_indices_for_pred,i].long()))
        pred.append(out[i][Train_indices_for_pred,:].max(1)[1].detach().cpu().numpy())
        PredictedLabels.append(F.softmax(out[i][Train_indices_for_pred,:],dim=1).detach().cpu().numpy())
  
    return loss, pred, PredictedLabels

def gather_apply_loss(training, contrast_loss_Ph, PatchEntropy_loss_Ph, contrast_loss_Nb, PatchEntropy_loss_Nb, contrast_loss_Ar, PatchEntropy_loss_Ar, Cross_entropy, optimizer, model, args):
    
    Cross_entropy_loss = 0
    for i in range(len(Cross_entropy)):        
        Cross_entropy_loss += Cross_entropy[i]

    loss  = contrast_loss_Ph*args['ContrastACC_Pheno'] + PatchEntropy_loss_Ph*args['PatchEntropy_Pheno']
    loss += contrast_loss_Nb*args['ContrastACC_Neigh'] + PatchEntropy_loss_Nb*args['PatchEntropy_Neigh']
    loss += contrast_loss_Ar*args['ContrastACC_Area']  + PatchEntropy_loss_Ar*args['PatchEntropy_Area']
    loss += Cross_entropy_loss*args['CrossEntropy_Loss'] 

    if training:
        loss.backward()    

        # print(model.pheno_clust.lin.weight.grad)
        optimizer.step() 
        optimizer.zero_grad()
    

def cov_matrix(m, rowvar=True):
    '''Estimate a covariance matrix given data.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m = m - m.mean(dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def pearson_corr(x,y): 
    ''' Obtain the pearson coefficient between two vectors.'''
    mean_x = x.mean()
    mean_y = y.type(torch.Tensor).mean()
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / (r_den+1e-16)
    # # Obtain confidence interval 0.95
    # pearson_coef, p_value = stats.pearsonr(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    return r_val

# def apply_adaptive_loss(loss, optimizer, hyper_optim, net):    
#     first_grad = ag.grad(loss, net.parameters(),allow_unused=True, retain_graph=True)
#     hyper_optim.compute_hg(net, first_grad)    
#     loss.backward()
#     optimizer.step()    
#     hyper_optim.hyper_step(first_grad)
    
def transformToInt(args):    
    args["clusters"] = [int(args["N_Phenotypes"]), int(args["N_Neighborhoods"]), int(args["N_Areas"])]
    args["batch_size"] = int(args["batch_size"])
    args["lr_decay_step_size"] = int(args["lr_decay_step_size"])
    args["epochs"] = int(args["epochs"])    

    return args

def initialization_params(dataset,args,num_features,model):
    '''
    Function to save the parameters used to initialize this model.
    '''
    
    # Obtain number of parameters
    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    args['n_Params_in_model'] = get_n_params(model)
    
    # Create an excel file with the initialization parameters.
    workbook = xlsxwriter.Workbook(dataset.processed_dir_cross_validation+"NaroNet_parameters.xlsx")
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for key in args.keys():
        row += 1
        col=0
        worksheet.write(row, 0, key)
        if ("list" in str(type(args[key]))):
            for col, i in enumerate(args[key]):                
                worksheet.write(row, col+1, i)    
        elif ("tuple" in str(type(args[key]))) or ("ndarray" in str(type(args[key]))): 
            worksheet.write(row, 1, 'None')            
        else:
            worksheet.write(row, 1, str(args[key]))
    workbook.close()

def showAndSaveEpoch(self,train_results,validation_results,fold,epoch,time,t_start_epoch):
    
    # Save training statistic
    if self.dict_epoch==None:
        self.dict_epoch={}
        self.dict_epoch['train_acc']=[[train_results[2]]]        
        self.dict_epoch['train_cross_ent_loss']=[[train_results[3]]]
        self.dict_epoch['train_contrast_loss_Ph_loss']=[[train_results[4]]]
        self.dict_epoch['train_patch_entropy_Ph_loss']=[[train_results[5]]]
        self.dict_epoch['train_contrast_loss_Nb_loss']=[[train_results[6]]]
        self.dict_epoch['train_patch_entropy_Nb_loss']=[[train_results[7]]]
        self.dict_epoch['train_contrast_loss_Ar_loss']=[[train_results[8]]]
        self.dict_epoch['train_patch_entropy_Ar_loss']=[[train_results[9]]]        
        self.dict_epoch['validation_acc']=[[validation_results[2]]]        
        self.dict_epoch['validation_cross_ent_loss']=[[validation_results[3]]]
        self.dict_epoch['validation_contrast_loss_Ph_loss']=[[validation_results[4]]]
        self.dict_epoch['validation_patch_entropy_Ph_loss']=[[validation_results[5]]]
        self.dict_epoch['validation_contrast_loss_Nb_loss']=[[validation_results[6]]]
        self.dict_epoch['validation_patch_entropy_Nb_loss']=[[validation_results[7]]]
        self.dict_epoch['validation_contrast_loss_Ar_loss']=[[validation_results[8]]]
        self.dict_epoch['validation_patch_entropy_Ar_loss']=[[validation_results[9]]]                
        self.dict_epoch['time/epoch']=[[time-t_start_epoch]]
        self.dict_epoch['epoch']=epoch
    else:
        self.dict_epoch['train_acc'].append([train_results[2]])        
        self.dict_epoch['train_cross_ent_loss'].append([train_results[3]])
        self.dict_epoch['train_contrast_loss_Ph_loss'].append([train_results[4]])
        self.dict_epoch['train_patch_entropy_Ph_loss'].append([train_results[5]])
        self.dict_epoch['train_contrast_loss_Nb_loss'].append([train_results[6]])
        self.dict_epoch['train_patch_entropy_Nb_loss'].append([train_results[7]])
        self.dict_epoch['train_contrast_loss_Ar_loss'].append([train_results[8]])
        self.dict_epoch['train_patch_entropy_Ar_loss'].append([train_results[9]])        
        self.dict_epoch['validation_acc'].append([validation_results[2]])
        self.dict_epoch['validation_cross_ent_loss'].append([validation_results[3]])
        self.dict_epoch['validation_contrast_loss_Ph_loss'].append([validation_results[4]])
        self.dict_epoch['validation_patch_entropy_Ph_loss'].append([validation_results[5]])
        self.dict_epoch['validation_contrast_loss_Nb_loss'].append([validation_results[6]])
        self.dict_epoch['validation_patch_entropy_Nb_loss'].append([validation_results[7]])
        self.dict_epoch['validation_contrast_loss_Ar_loss'].append([validation_results[8]])
        self.dict_epoch['validation_patch_entropy_Ar_loss'].append([validation_results[9]])
        self.dict_epoch['time/epoch'].append([time-t_start_epoch])
        self.dict_epoch['epoch']=epoch
    
    eval_info = {}
    for dict_i in self.dict_epoch:
        eval_info[dict_i] = round(self.dict_epoch[dict_i][-1][-1],4) if type(self.dict_epoch[dict_i]) is list else self.dict_epoch[dict_i]

    # Save epoch information into a log file.
    fn = self.dataset.processed_dir_cross_validation+'epochs_info.txt'
    if epoch==1 and fold==0:
        with open(fn, "w") as myfile:
            myfile.write(str(eval_info)+"\n")        
    else:
        with open(fn, "a") as myfile:
            myfile.write(str(eval_info)+"\n")        
   
    return self.dict_epoch

def showAndSaveFold(self,train_results,validation_results,test_results,time,t_start_fold): 
    # Save Fold statistics
    if self.dict_fold==None:
        self.dict_fold={}
        self.dict_fold['train_acc']=self.dict_epoch['train_acc']
        self.dict_fold['train_cross_ent_loss']=self.dict_epoch['train_cross_ent_loss']
        self.dict_fold['train_contrast_loss_Ph_loss']=self.dict_epoch['train_contrast_loss_Ph_loss']
        self.dict_fold['train_patch_entropy_Ph_loss']=self.dict_epoch['train_patch_entropy_Ph_loss']
        self.dict_fold['train_contrast_loss_Nb_loss']=self.dict_epoch['train_contrast_loss_Nb_loss']
        self.dict_fold['train_patch_entropy_Nb_loss']=self.dict_epoch['train_patch_entropy_Nb_loss']
        self.dict_fold['train_contrast_loss_Ar_loss']=self.dict_epoch['train_contrast_loss_Ar_loss']
        self.dict_fold['train_patch_entropy_Ar_loss']=self.dict_epoch['train_patch_entropy_Ar_loss']
        self.dict_fold['validation_acc']=self.dict_epoch['validation_acc']
        self.dict_fold['validation_cross_ent_loss']=self.dict_epoch['validation_cross_ent_loss']
        self.dict_fold['validation_contrast_loss_Ph_loss']=self.dict_epoch['validation_contrast_loss_Ph_loss']
        self.dict_fold['validation_patch_entropy_Ph_loss']=self.dict_epoch['validation_patch_entropy_Ph_loss'] 
        self.dict_fold['validation_contrast_loss_Nb_loss']=self.dict_epoch['validation_contrast_loss_Nb_loss']
        self.dict_fold['validation_patch_entropy_Nb_loss']=self.dict_epoch['validation_patch_entropy_Nb_loss']
        self.dict_fold['validation_contrast_loss_Ar_loss']=self.dict_epoch['validation_contrast_loss_Ar_loss']
        self.dict_fold['validation_patch_entropy_Ar_loss']=self.dict_epoch['validation_patch_entropy_Ar_loss']       
        self.dict_fold['train_subject_indices']=[train_results[10]]
        self.dict_fold['validation_subject_indices']=[validation_results[10]]
        self.dict_fold['time/epoch']=self.dict_epoch['time/epoch']        
        self.dict_fold['acc_train']=[train_results[2]]
        self.dict_fold['cross_ent_train']=[train_results[3]]
        self.dict_fold['acc_validation']=[validation_results[2]]
        self.dict_fold['cross_ent_validation']=[validation_results[3]]
        self.dict_fold['GroundTruthValues']=test_results[0]
        self.dict_fold['PredictedValues']=test_results[1]
        self.dict_fold['acc_test']=[test_results[2]]
        self.dict_fold['cross_ent_test']=[test_results[3]]
        self.dict_fold['AUC_test']=[0]
        self.dict_fold['test_subject_indices']=[test_results[10]]
        
    else:
        self.dict_fold['train_acc'] = [self.dict_fold['train_acc'][i]+d for i,d in enumerate(self.dict_epoch['train_acc'])]
        self.dict_fold['train_cross_ent_loss'] = [self.dict_fold['train_cross_ent_loss'][i]+d for i,d in enumerate(self.dict_epoch['train_cross_ent_loss'])]
        self.dict_fold['train_contrast_loss_Ph_loss'] = [self.dict_fold['train_contrast_loss_Ph_loss'][i]+d for i,d in enumerate(self.dict_epoch['train_contrast_loss_Ph_loss'])]
        self.dict_fold['train_patch_entropy_Ph_loss'] = [self.dict_fold['train_patch_entropy_Ph_loss'][i]+d for i,d in enumerate(self.dict_epoch['train_patch_entropy_Ph_loss'])]
        self.dict_fold['train_contrast_loss_Nb_loss'] = [self.dict_fold['train_contrast_loss_Nb_loss'][i]+d for i,d in enumerate(self.dict_epoch['train_contrast_loss_Nb_loss'])]
        self.dict_fold['train_patch_entropy_Nb_loss'] = [self.dict_fold['train_patch_entropy_Nb_loss'][i]+d for i,d in enumerate(self.dict_epoch['train_patch_entropy_Nb_loss'])]
        self.dict_fold['train_contrast_loss_Ar_loss'] = [self.dict_fold['train_contrast_loss_Ar_loss'][i]+d for i,d in enumerate(self.dict_epoch['train_contrast_loss_Ar_loss'])]
        self.dict_fold['train_patch_entropy_Ar_loss'] = [self.dict_fold['train_patch_entropy_Ar_loss'][i]+d for i,d in enumerate(self.dict_epoch['train_patch_entropy_Ar_loss'])]
        self.dict_fold['validation_acc'] = [self.dict_fold['validation_acc'][i]+d for i,d in enumerate(self.dict_epoch['validation_acc'])]
        self.dict_fold['validation_cross_ent_loss'] = [self.dict_fold['validation_cross_ent_loss'][i]+d for i,d in enumerate(self.dict_epoch['validation_cross_ent_loss'])]
        self.dict_fold['validation_contrast_loss_Ph_loss'] = [self.dict_fold['validation_contrast_loss_Ph_loss'][i]+d for i,d in enumerate(self.dict_epoch['validation_contrast_loss_Ph_loss'])]
        self.dict_fold['validation_patch_entropy_Ph_loss'] = [self.dict_fold['validation_patch_entropy_Ph_loss'][i]+d for i,d in enumerate(self.dict_epoch['validation_patch_entropy_Ph_loss'])]        
        self.dict_fold['validation_contrast_loss_Nb_loss'] = [self.dict_fold['validation_contrast_loss_Nb_loss'][i]+d for i,d in enumerate(self.dict_epoch['validation_contrast_loss_Nb_loss'])]
        self.dict_fold['validation_patch_entropy_Nb_loss'] = [self.dict_fold['validation_patch_entropy_Nb_loss'][i]+d for i,d in enumerate(self.dict_epoch['validation_patch_entropy_Nb_loss'])]
        self.dict_fold['validation_contrast_loss_Ar_loss'] = [self.dict_fold['validation_contrast_loss_Ar_loss'][i]+d for i,d in enumerate(self.dict_epoch['validation_contrast_loss_Ar_loss'])]
        self.dict_fold['validation_patch_entropy_Ar_loss'] = [self.dict_fold['validation_patch_entropy_Ar_loss'][i]+d for i,d in enumerate(self.dict_epoch['validation_patch_entropy_Ar_loss'])]
        self.dict_fold['train_subject_indices'].append(train_results[10])    
        self.dict_fold['validation_subject_indices'].append(validation_results[10])
        self.dict_fold['test_subject_indices'].append(test_results[10])
        self.dict_fold['time/epoch'] = [self.dict_fold['time/epoch'][i]+d for i,d in enumerate(self.dict_epoch['time/epoch'])]            
        self.dict_fold['acc_train'].append(train_results[2])
        self.dict_fold['cross_ent_train'].append(train_results[3])
        self.dict_fold['acc_validation'].append(validation_results[2])
        self.dict_fold['cross_ent_validation'].append(validation_results[3])       
        self.dict_fold['AUC_test'].append(0) 
        self.dict_fold['GroundTruthValues']=np.concatenate((self.dict_fold['GroundTruthValues'],test_results[0]),axis=0)
        for i in range(len(self.dict_fold['PredictedValues'])):
            self.dict_fold['PredictedValues'][i] = np.concatenate((self.dict_fold['PredictedValues'][i],test_results[1][i]),axis=0)        
        self.dict_fold['acc_test'].append(test_results[2])
        self.dict_fold['cross_ent_test'].append(test_results[3])
        
        # Save Best Model.        
        if any([self.dict_fold['acc_test'][-1]>i for i in self.dict_fold['acc_test'][:-1]]) and ("self.model.lin1_1" in locals()):
            torch.save([self.model.lin1_1,self.model.BNLast_1,self.model.lin2_1], self.dataset.processed_dir_cross_validation+"/model.pt")

    return self.dict_fold


def recall_precision_curve(name_labels,GTValues,Pred_Values,processed_dir_cross_validation,exp_name,type_eval):
    comb = combinations(list(range(len(name_labels))), 2)
    for c in list(comb): 
        val_sel = np.logical_or(GTValues==c[0],GTValues==c[1])
        Sel_Pred_Values = Pred_Values[val_sel,:][:,c]
        Sel_Pred_Values = Sel_Pred_Values/Sel_Pred_Values.sum(-1,keepdims=True)
        GT_val = GTValues[val_sel]  
        if all(GT_val==c[0]) or all(GT_val==c[1]):                
            average_precision=0
            continue # do not display ROC_AUC for now.
        else:
            GT_val = GT_val-GT_val.min()
            GT_val = GT_val/GT_val.max()
            average_precision = metrics.average_precision_score(GT_val, Sel_Pred_Values[:,1])
            precision, recall, thresholds_ = metrics.precision_recall_curve(GT_val, Sel_Pred_Values[:,1])
            f1_score = metrics.f1_score(GT_val, (Sel_Pred_Values[:,1]>thresholds_[np.argmax(recall - precision)])*1)                
            h = plot_prec_recall(recall,precision,average_precision,f1_score)                
            h.savefig(processed_dir_cross_validation+"/Recall_Precision_"+str(exp_name)+'_'+str(c)+"_"+type_eval+".png",dpi=900)
            h.close()  


def auc_roc_curve(name_labels,GTValues,Pred_Values,processed_dir_cross_validation,exp_name,type_eval):
    comb = combinations(list(range(len(name_labels))), 2)
    thresholds_all = []
    fpr_all = []
    tpr_all = []
    roc_auc_all = []
    for c in list(comb): 
        val_sel = np.logical_or(GTValues==c[0],GTValues==c[1])
        Sel_Pred_Values = Pred_Values[val_sel,:][:,c]
        Sel_Pred_Values = Sel_Pred_Values/Sel_Pred_Values.sum(-1,keepdims=True)
        GT_val = GTValues[val_sel]        
        if all(GT_val==c[0]) or all(GT_val==c[1]):                
            roc_auc=0
            thresholds='None'
            fpr=0
            tpr=0.5
            continue # do not display ROC_AUC for now.
        else:
            GT_val = GT_val-GT_val.min()
            GT_val = GT_val/GT_val.max()
            fpr, tpr, thresholds = metrics.roc_curve(GT_val,Sel_Pred_Values[:,1], pos_label=1)
            thresholds_all.append(thresholds)
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            roc_auc = metrics.roc_auc_score(GT_val, Sel_Pred_Values[:,1]) 
            ci = confidence_interval_AUC(GT_val, Sel_Pred_Values[:,1])
            h = plot_roc(fpr,tpr,roc_auc)
            h.savefig(processed_dir_cross_validation+"/ROC_AUC_"+str(exp_name)+'_'+str(c)+"_"+type_eval+"_CI{}_{}_.png".format(ci[0],ci[1]),dpi=900)
            h.close()  
            roc_auc_all.append(roc_auc)
    return thresholds_all,fpr_all,tpr_all, sum(roc_auc_all)/len(list(combinations(list(range(len(name_labels))), 2)))

def confusion_matrix(name_labels,GTValues,Pred_Values,processed_dir_cross_validation,thresholds,fpr,tpr,exp_name,type_eval):
    plt.figure()
    if np.unique(GTValues).shape[0]>1:
        if len(name_labels)==2 and (not isinstance(thresholds, str)):
            h = plot_confusion_matrix_from_data(GTValues, (Pred_Values[:,1]>thresholds[0][np.argmax(tpr[0] - fpr[0])])*1, columns=name_labels,fz=24)
            h.savefig(processed_dir_cross_validation+'ConfusionMatrix_'+str(exp_name)+'_'+str(name_labels)+"_"+type_eval+".png",dpi=900)
        else:
            for n_c, c in enumerate(list(combinations(list(range(len(name_labels))), 2))): 
                val_sel = np.logical_or(GTValues==c[0],GTValues==c[1])
                Sel_Pred_Values = Pred_Values[val_sel,:][:,c]
                Sel_Pred_Values = Sel_Pred_Values/Sel_Pred_Values.sum(-1,keepdims=True)
                GT_val = GTValues[val_sel]
                GT_val = GT_val-GT_val.min()
                GT_val = GT_val/GT_val.max() if GT_val.max()>0 else GT_val+1
                try:
                    h = plot_confusion_matrix_from_data(GT_val, (Sel_Pred_Values[:,1]>thresholds[n_c][np.argmax(tpr[n_c] - fpr[n_c])])*1, columns=[name_labels[cc] for cc in c],fz=24)
                    h.savefig(processed_dir_cross_validation+'ConfusionMatrix_'+str(exp_name)+'_'+str([name_labels[cc] for cc in c])+"_"+type_eval+".png",dpi=900)
                    h.close()
                except:
                    continue            
            try:
                h = plot_confusion_matrix_from_data(GTValues, Pred_Values.argmax(-1), columns=name_labels,fz=24)
                h.savefig(processed_dir_cross_validation+'ConfusionMatrix_'+str(exp_name)+'_'+str(name_labels)+"_"+type_eval+".png",dpi=900)
            except:
                return
        h.close()

def Correlation_to_GT(Pred_Val, GT_Val,processed_dir_cross_validation,type_eval):
    regr = linear_model.LinearRegression()
    regr.fit(Pred_Val.reshape(-1,1), GT_Val.reshape(-1,1))
    y_pred = regr.predict(Pred_Val.reshape(-1,1))
    r2_sc = metrics.r2_score(GT_Val,Pred_Val)
    mse_sc = metrics.mean_squared_error(GT_Val,Pred_Val)
    plt.figure()
    plt.scatter(Pred_Val.reshape(-1,1), GT_Val.reshape(-1,1),  color='black')
    plt.plot(Pred_Val.reshape(-1,1), y_pred, color='blue', linewidth=3)
    plt.savefig(processed_dir_cross_validation+"CorrelationGT_Pred_R2{}_MSE{}_{}.png".format(str(round(r2_sc,3)),str(round(mse_sc,3)),type_eval),dpi=900)

def confidence_interval_AUC(y_true, y_pred): 

    from scipy.stats import sem
    from sklearn.metrics import roc_auc_score 
    from numpy import random as rng
    n_bootstraps = 1000   
    bootstrapped_scores = []   
   
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
       
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)   
 
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

   # 90% c.i.
   # confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
   # confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
 
    # 95% c.i.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
   
    return confidence_lower,confidence_upper

def showAndSaveEndOfTraining(self):                
    # Reinitialize
    plt.clf()
    
    # Save epoch information into a log file.
    eval_info = {'train_accuracy_mean': np.mean(self.dict_fold['acc_train']),'train_accuracy_std': np.std(self.dict_fold['acc_train']), 
    'validation_accuracy_mean': np.mean(self.dict_fold['acc_validation']),'validation_accuracy_std': np.std(self.dict_fold['acc_validation']),
    'test_accuracy_mean': np.mean(self.dict_fold['acc_test']),'tests_accuracy_std': np.std(self.dict_fold['acc_test'])}
    fn = self.dataset.processed_dir_cross_validation+"folds_information.txt"                        
    with open(fn, "a") as myfile:
        myfile.write(str(eval_info)+"\n")        

    # Save test info in excel
    for label_i in range(len(self.dict_fold['PredictedValues'])):
        table_prediction = {'Subject_name':[],'Prediction':[],'Label':[]}
        for i_s, subject in enumerate(np.concatenate(self.dict_fold['test_subject_indices'])):
            table_prediction['Subject_name'].append(self.IndexAndClass[subject][0])            
            table_prediction['Prediction'].append(self.dict_fold['PredictedValues'][label_i][i_s,:])
            table_prediction['Label'].append(self.dict_fold['GroundTruthValues'][:,label_i][i_s])
        table_prediction = pd.DataFrame.from_dict(table_prediction)
        table_prediction.to_excel(self.dataset.processed_dir_cross_validation+"/Prediction_values_"+str(self.args['experiment_Label'][label_i])+'_Fold'+str(self.fold)+".xlsx")    

    for i in range(len(self.dict_fold['PredictedValues'])):               
        
        if len(set([iac[2][i] for iac in self.IndexAndClass]))<6:        
            # AUC-ROC curves
            thresholds,fpr,tpr, auc = auc_roc_curve(self.name_labels[i],self.dict_fold['GroundTruthValues'][:,i],
            self.dict_fold['PredictedValues'][i],self.dataset.processed_dir_cross_validation,self.args['experiment_Label'][i],'record_wise')
            self.dict_fold['AUC_test'][-1] = auc
            
            # Recall-Precision curve
            recall_precision_curve(self.name_labels[i],self.dict_fold['GroundTruthValues'][:,i],
            self.dict_fold['PredictedValues'][i],self.dataset.processed_dir_cross_validation,self.args['experiment_Label'][i],'record_wise')
            
            # Confusion matrix        
            confusion_matrix(self.name_labels[i],self.dict_fold['GroundTruthValues'][:,i],self.dict_fold['PredictedValues'][i],
            self.dataset.processed_dir_cross_validation,thresholds,fpr,tpr,self.args[''][i],'record_wise')
            
        else:  
            # Obtain correlation of predicted values to the ground-truth
            Correlation_to_GT(self.dict_fold['PredictedValues'][i].argmax(1),self.dict_fold['GroundTruthValues'][:,i],
            self.dataset.processed_dir_cross_validation,'record_wise')        

        # Subject wise analysis
        patient_to_image_excel = pd.read_excel(self.dataset.root+'Raw_Data/Experiment_Information/Image_Labels.xlsx')  
        # In case there is a column named 'Subject_Names' we have to classify subject-wise patients
        if any([p=='Subject_Names' for p in patient_to_image_excel.columns]):
            if any([len(pat.split('.'))>1 for pat in patient_to_image_excel['Image_Names']]): 
                lst_excl = ['.'.join(pat.split('.')[:-1]) for pat in patient_to_image_excel['Image_Names']]
            else:
                lst_excl = [pat for pat in patient_to_image_excel['Image_Names']]
            
            image_ind = [lst_excl.index(self.IndexAndClass[ind][0]) for ind in np.concatenate(self.dict_fold['test_subject_indices'])]
            subjects = [patient_to_image_excel['Subject_Names'][ii] for ii in image_ind]
            
            Pred_values = []
            GT_values = []
            for subj in set(subjects):
                subj_images_ind = [ii for ii in image_ind if patient_to_image_excel['Subject_Names'][ii]==subj]
                subj_images_names = [lst_excl[i] for i in subj_images_ind]
                subj_images_ind_test = [[i[0] for i in self.IndexAndClass].index(sii) for sii in subj_images_names]
                test_ind = [list(np.concatenate(self.dict_fold['test_subject_indices'])).index(siit) for siit in subj_images_ind_test]
                Pred_values.append(np.mean(self.dict_fold['PredictedValues'][i][test_ind,:],axis=0))
                GT_values.append(np.mean(self.dict_fold['GroundTruthValues'][test_ind,i],axis=0))

            if len(set([iac[2][i] for iac in self.IndexAndClass]))<6: 
                # AUC-ROC curves
                thresholds,fpr,tpr,auc = auc_roc_curve(self.name_labels[i],np.stack(GT_values),
                np.stack(Pred_values),self.dataset.processed_dir_cross_validation,self.args['experiment_Label'][i],'subject_wise')
                # Recall-Precision curve
                recall_precision_curve(self.name_labels[i],np.stack(GT_values),
                np.stack(Pred_values),self.dataset.processed_dir_cross_validation,self.args['experiment_Label'][i],'subject_wise')
                # Confusion matrix        
                confusion_matrix(self.name_labels[i],np.stack(GT_values),np.stack(Pred_values),
                self.dataset.processed_dir_cross_validation,thresholds,fpr,tpr,self.args['experiment_Label'][i],'subject_wise')
            else:
                # Obtain correlation of predicted values to the ground-truth
                Correlation_to_GT(np.stack(Pred_values).argmax(1),np.stack(GT_values),
                self.dataset.processed_dir_cross_validation,'subject_wise')   
            
            self.dict_fold['AUC_test'][-1] = auc

    # Save Training/Validation Loss and accuracy
    plt.figure()
    h = plot_training_loss_acc(train_info=self.dict_fold['train_cross_ent_loss'],val_info=self.dict_fold['validation_cross_ent_loss'], title='Training/Validation Cross entropy Loss Test:'+str(round(np.mean(self.dict_fold['validation_cross_ent_loss']),4))+'±'+str(round(np.std(self.dict_fold['validation_cross_ent_loss']),4)), label=['Training cross entropy Loss', 'Validation cross entropy Loss'], ylabel='Loss')
    h.savefig(self.dataset.processed_dir_cross_validation+"/TrainingValidation_Loss.png",dpi=900)
    h.close()  
    plt.figure()  
    h = plot_training_loss_acc(train_info=self.dict_fold['train_acc'],val_info=self.dict_fold['validation_acc'], title='Training/Validation Loss Test:'+str(round(np.mean(self.dict_fold['validation_acc']),4))+'±'+str(round(np.std(self.dict_fold['validation_acc']),4)), label=['Training Accuracy', 'Validation Accuracy'], ylabel='Accuracy')
    h.savefig(self.dataset.processed_dir_cross_validation+"/TrainingValidation_Acc.png",dpi=900)
    h.close()

    # Save Training/Validation contrast Loss
    plt.figure()  
    h = plot_training_loss_acc(train_info=self.dict_fold['train_contrast_loss_Ph_loss'],val_info=self.dict_fold['validation_contrast_loss_Ph_loss'], title='Training/Validation Contrast Loss Ph', label=['Training Contrast Loss Ph', 'Validation Contrast Loss Ph'], ylabel='Contrast Loss Ph')
    h.savefig(self.dataset.processed_dir_cross_validation+"/TrainingValidation_ContrastLoss_Ph.png",dpi=900)
    h.close()
    plt.figure()  
    h = plot_training_loss_acc(train_info=self.dict_fold['train_contrast_loss_Nb_loss'],val_info=self.dict_fold['validation_contrast_loss_Nb_loss'], title='Training/Validation Contrast Loss Nb', label=['Training Contrast Loss Nb', 'Validation Contrast Loss Nb'], ylabel='Contrast Loss Nb')
    h.savefig(self.dataset.processed_dir_cross_validation+"/TrainingValidation_ContrastLoss_Nb.png",dpi=900)
    h.close()
    plt.figure()  
    h = plot_training_loss_acc(train_info=self.dict_fold['train_contrast_loss_Ar_loss'],val_info=self.dict_fold['validation_contrast_loss_Ar_loss'], title='Training/Validation Contrast Loss Ar', label=['Training Contrast Loss Ar', 'Validation Contrast Loss Ar'], ylabel='Contrast Loss Ar')
    h.savefig(self.dataset.processed_dir_cross_validation+"/TrainingValidation_ContrastLoss_Ar.png",dpi=900)
    h.close()

    # Save Training/Validation Patch Entropy Loss
    plt.figure()  
    h = plot_training_loss_acc(train_info=self.dict_fold['train_patch_entropy_Ph_loss'],val_info=self.dict_fold['validation_patch_entropy_Ph_loss'], title='Training/Validation Patch Entropy Loss Ph', label=['Training Patch Entropy Loss Ph', 'Validation Patch Entropy Loss Ph'], ylabel='Patch Entropy Ph')
    h.savefig(self.dataset.processed_dir_cross_validation+"/TrainingValidation_PatchEntropy_Ph.png",dpi=900)
    h.close()
    plt.figure()  
    h = plot_training_loss_acc(train_info=self.dict_fold['train_patch_entropy_Nb_loss'],val_info=self.dict_fold['validation_patch_entropy_Nb_loss'], title='Training/Validation Patch Entropy Loss Nb', label=['Training Patch Entropy Loss Nb', 'Validation Patch Entropy Loss Nb'], ylabel='Patch Entropy Nb')
    h.savefig(self.dataset.processed_dir_cross_validation+"/TrainingValidation_PatchEntropy_Nb.png",dpi=900)
    h.close()
    plt.figure()  
    h = plot_training_loss_acc(train_info=self.dict_fold['train_patch_entropy_Ar_loss'],val_info=self.dict_fold['validation_patch_entropy_Nb_loss'], title='Training/Validation Patch Entropy Loss Ar', label=['Training Patch Entropy Loss Ar', 'Validation Patch Entropy Loss Ar'], ylabel='Patch Entropy Ar')
    h.savefig(self.dataset.processed_dir_cross_validation+"/TrainingValidation_PatchEntropy_Ar.png",dpi=900)
    h.close()
    
    return self.dict_fold

def checkIfDebugging(args):
    args['epochs'] = args['epochs'] if np.isscalar(args['epochs']) else np.asscalar(args['epochs'])
    args['batch_size'] = args['batch_size'] if np.isscalar(args['batch_size']) else np.asscalar(args['batch_size'])
    args['lr'] = args['lr'] if np.isscalar(args['lr']) else np.asscalar(args['lr'])
    args['clusters1'] = args['clusters1'] if np.isscalar(args['clusters1']) else np.asscalar(args['clusters1'])
    args['clusters2'] = args['clusters2'] if np.isscalar(args['clusters2']) else np.asscalar(args['clusters2'])
    args['clusters3'] = args['clusters3'] if np.isscalar(args['clusters3']) else np.asscalar(args['clusters3'])     
    args['isAttentionLayer'] = args['isAttentionLayer'] if np.isscalar(args['isAttentionLayer']) else np.asscalar(args['isAttentionLayer'])     
    args['orthoColor'] = args['orthoColor'] if np.isscalar(args['orthoColor']) else np.asscalar(args['orthoColor'])             
    args['ortho'] = args['ortho'] if np.isscalar(args['ortho']) else np.asscalar(args['ortho'])             
    args['MinCut'] = args['MinCut'] if np.isscalar(args['MinCut']) else np.asscalar(args['MinCut'])             
    args['dropoutRate'] = args['dropoutRate'] if np.isscalar(args['dropoutRate']) else np.asscalar(args['dropoutRate'])             
    args['AttntnSparsenss'] = args['AttntnSparsenss'] if np.isscalar(args['AttntnSparsenss']) else np.asscalar(args['AttntnSparsenss'])             
    args['normalizeFeats'] = args['normalizeFeats'] if np.isscalar(args['normalizeFeats']) else np.asscalar(args['normalizeFeats'])             
    args['normalizeCells'] = args['normalizeCells'] if np.isscalar(args['normalizeCells']) else np.asscalar(args['normalizeCells'])             
    args['useOptimizer'] = args['useOptimizer'] if np.isscalar(args['useOptimizer']) else np.asscalar(args['useOptimizer'])             
    args['lr_decay_step_size'] = args['lr_decay_step_size'] if np.isscalar(args['lr_decay_step_size']) else np.asscalar(args['lr_decay_step_size'])             
    return args