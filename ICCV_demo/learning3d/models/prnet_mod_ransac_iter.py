#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import copy
import math
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. ops import transform_functions as transform
from .. utils import Transformer, Identity

from sklearn.metrics import r2_score


# +
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# -

def pairwise_distance(src, tgt):
    inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), tgt)
    xx = torch.sum(src**2, dim=1, keepdim=True)
    yy = torch.sum(tgt**2, dim=1, keepdim=True)
    distances = xx.transpose(2, 1).contiguous() + inner + yy
    return torch.sqrt(distances)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    device=x.device
    # x = x.squeeze()
    x = x.view(*x.size()[:3])
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


def cycle_consistency(rotation_ab, translation_ab, rotation_ba, translation_ba):
    batch_size = rotation_ab.size(0)
    identity = torch.eye(3, device=rotation_ab.device).unsqueeze(0).repeat(batch_size, 1, 1)
    return F.mse_loss(torch.matmul(rotation_ab, rotation_ba), identity) + F.mse_loss(translation_ab, -translation_ba)


class PointNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x1 = x.max(dim=-1, keepdim=True)[0]
 
        x = get_graph_feature(x1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x3)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2).view(batch_size, -1, num_points)
        return x


class MLPHead(nn.Module):
    def __init__(self, emb_dims):
        super(MLPHead, self).__init__()
        n_emb_dims = emb_dims
        self.n_emb_dims = n_emb_dims
        self.nn = nn.Sequential(nn.Linear(n_emb_dims*2, n_emb_dims//2),
                                nn.BatchNorm1d(n_emb_dims//2),
                                nn.ReLU(),
                                nn.Linear(n_emb_dims//2, n_emb_dims//4),
                                nn.BatchNorm1d(n_emb_dims//4),
                                nn.ReLU(),
                                nn.Linear(n_emb_dims//4, n_emb_dims//8),
                                nn.BatchNorm1d(n_emb_dims//8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(n_emb_dims//8, 4)
        self.proj_trans = nn.Linear(n_emb_dims//8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


# ML based Heat Kernel
class TemperatureNet(nn.Module):
    def __init__(self, emb_dims, temp_factor):
        super(TemperatureNet, self).__init__()
        self.n_emb_dims = emb_dims
        self.temp_factor = temp_factor
        self.nn = nn.Sequential(nn.Linear(self.n_emb_dims, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128, 1),
                                nn.ReLU())
        self.feature_disparity = None

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src_embedding = src_embedding.mean(dim=2)
        tgt_embedding = tgt_embedding.mean(dim=2)
        residual = torch.abs(src_embedding-tgt_embedding)

        self.feature_disparity = residual

        return torch.clamp(self.nn(residual), 1.0/self.temp_factor, 1.0*self.temp_factor), residual

# ## Rotation code from SVD


class SVDHead(nn.Module):
    def __init__(self, emb_dims, cat_sampler):
        super(SVDHead, self).__init__()
        self.n_emb_dims = emb_dims
        self.cat_sampler = cat_sampler
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.temperature = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
        self.my_iter = torch.ones(1)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size, num_dims, num_points = src.size()
        temperature = input[4].view(batch_size, 1, 1)

        if self.cat_sampler == 'softmax':
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
            scores = torch.softmax(temperature*scores, dim=2)
        elif self.cat_sampler == 'gumbel_softmax':
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
            scores = scores.view(batch_size*num_points, num_points)
            temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
            scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
            scores = scores.view(batch_size, num_points, num_points)
        else:
            raise Exception('not implemented')

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        R = []

        for i in range(src.size(0)):
#             u, s, v = torch.svd(H[i])
            L=H[i]
            try:
                u, s, v = torch.svd(L)
            except Exception as e:                     # torch.svd may have convergence issues for GPU and CPU.
                print(src.shape)
#                 print(e)
#                 print(L)
#                 print(src_corr)
                u, s, v = torch.svd(L + 1e-4*L.mean()*torch.rand(L.shape[0], L.shape[1]).to(L.device))

            # Kabst
            r = torch.matmul(v, u.transpose(1, 0)).contiguous()
            r_det = torch.det(r).item()
            diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                              [0, 1.0, 0],
                                              [0, 0, r_det]]).astype('float32')).to(v.device)
            r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
            R.append(r)

        R = torch.stack(R, dim=0).to(src.device)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        if self.training:
            self.my_iter += 1
        return R, t.view(batch_size, 3)


# +
class SVDHead_mod(nn.Module):
    def __init__(self, emb_dims, cat_sampler):
        super(SVDHead_mod, self).__init__()
        self.n_emb_dims = emb_dims
        self.cat_sampler = cat_sampler
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.temperature = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
        self.my_iter = torch.ones(1)
        
        # mem
        self.mem={}

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size, num_dims, num_points = src.size()
        temperature = input[4].view(batch_size, 1, 1)
        
        ransac_inliers=True
        if (ransac_inliers):
            ""
            # RANSAC on TOPK 128?
            t_inliers=[]
            for it in range(src.size(0)):
#                 print(it)
                src_i=src[it].unsqueeze(0)
                tgt_i=tgt[it].unsqueeze(0)
                pred_trans_i, inliers_i=ransac_align(src_i.permute(0,2,1),tgt_i.permute(0,2,1))
            
#                 print(pred_trans_i.shape)
#                 print('pred_trans',pred_trans_i)
#                 print('inliers')
#                 print(inliers_i.shape)
#                 print(inliers_i)
                
                # Create 512x512 matrix
#                 print()
                na_ones=np.zeros((num_points,num_points))
                for r in range(inliers_i.shape[0]):
                    na_ones[inliers_i[r,0],inliers_i[r,1]]=1
                
#                 print(na_ones)
#                                 inliers = np.array(reg_result.correspondence_set)
#                 pred_labels = torch.zeros_like(gt_labels)
#                 pred_labels[0, inliers[:, 0]] = 1
                #print('PRED_i')
                #print(pred_trans_i)
                #print(inliers_i.shape)
                t_inliers.append(torch.from_numpy(na_ones))
            t_inliers=torch.stack(t_inliers)
#             print('t_inliers',t_inliers.shape)
            
            
        

        # Keypoints Inlier selection and 
        if self.cat_sampler == 'softmax':
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
            scores = torch.softmax(temperature*scores, dim=2)
        elif self.cat_sampler == 'gumbel_softmax':
            d_k = src_embedding.size(1)
            
#             print(d_k.shape)
            
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)

            scores = scores.view(batch_size*num_points, num_points)
          
            temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
            
            scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
         
            scores = scores.view(batch_size, num_points, num_points)
            # Binary Scores - doubly stochastic   
            
        else:
            raise Exception('not implemented')

        ### View scores
        if (ransac_inliers):
            scores_b=t_inliers.to(scores).detach()
            self.mem['scores']=scores
            self.mem['scores_b']=scores_b
        else:
            scores_b=scores
        scores_b=scores
        #
                  
            
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
#         print('scores')
#         print(scores.shape)
#         print(scores[0].count_nonzero())
#         print(scores[0].nonzero())
#         idx=scores[0].nonzero().cpu().T.numpy()
#         print(scores.cpu()[0][idx])
        
# #         [[  0, 156],
# #         [  1, 158],
        
#         print(src_corr.shape)
#         print(src_corr[:2])

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

#         H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous()).cpu()
        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        R = []

        for i in range(src.size(0)):
#             u, s, v = torch.svd(H[i])
            L=H[i]
            try:
                u, s, v = torch.svd(L)
            except Exception as e:                     # torch.svd may have convergence issues for GPU and CPU.
                print(src.shape)
#                 print(e)
#                 print(L)
#                 print(src_corr)
                u, s, v = torch.svd(L + 1e-4*L.mean()*torch.rand(L.shape[0], L.shape[1]).to(L.device))
                
            
            r = torch.matmul(v, u.transpose(1, 0)).contiguous()
            r_det = torch.det(r).item()
            diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                              [0, 1.0, 0],
                                              [0, 0, r_det]]).astype('float32')).to(v.device)
            r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
            R.append(r)

        R = torch.stack(R, dim=0).to(src.device)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        if self.training:
            self.my_iter += 1
            
#         print(R,t)
        return R, t.view(batch_size, 3)

# +
from jakteristics import compute_features
feature_names=['eigenvalue_sum', 'omnivariance', 'eigenentropy', 'anisotropy', 'planarity', 'linearity', 'PCA1', 'PCA2', 'surface_variation', 'sphericity', 'verticality', 'nx', 'ny', 'nz']
# # feature_names=['nx', 'ny', 'nz','eigenvalue_sum']
# features_pcd_src = compute_features(t_pcd_src0[0].permute(1,0).cpu().numpy().astype(float), search_radius=0.15,max_k_neighbors=20,feature_names=feature_names)
# # t_pcd_src0
# # t_pcd_src0[0].shape, features.shape
# features_pcd_tgt_part = compute_features(t_pcd_tgt_part0[0].permute(1,0).cpu().numpy().astype(float), search_radius=0.15,max_k_neighbors=20,feature_names=feature_names)

def get_geometric_features(pcd,K=20):
#     print('pcd',pcd.shape)
    features=compute_features(pcd.permute(1,0).detach().cpu().numpy().astype(float), search_radius=1,max_k_neighbors=K,feature_names=feature_names)
    eps=1e-6
    mat_det=features[:,1]
    mat_trace=features[:,0]

#     lam=(mat_det+eps)/(mat_trace+eps)**(1/3)
#     sharpness=(mat_det+eps) - (mat_trace+eps)**(3)
    mat_det=mat_det**(3)
    mat_trace=mat_trace**(1/3)
    lam=(mat_det+eps)/(mat_trace+eps)
    sharpness=(mat_det+eps) - (mat_trace+eps)
#     lam, sharpness, H_S_src, H_min_src
    return lam, sharpness

def tensor_batch_geo_features(t_pcd_src0,K=20):
    H_min=[]
    H_sharpness=[]
    for it in range(t_pcd_src0.shape[0]):
        pcd=t_pcd_src0[it]
#         print('pcd',pcd.shape)
        lam,sharpness=get_geometric_features(pcd,K=K)
        H_min.append(torch.from_numpy(lam))
        H_sharpness.append(torch.from_numpy(sharpness))
    H_min=torch.stack(H_min,dim=0)
    H_sharpness=torch.stack(H_sharpness,dim=0)

    return H_min, H_sharpness

# t_features_src_0,t_features_src_1=tensor_batch_geo_features(t_pcd_src0)


# +

def geometric_matching(t_pcd_src,t_pcd_tgt_part,K=20,C=50,th1=0.8,th2=0.0001, renormalize=False):
#     C=50
    verbose=False
    
    if (renormalize):
        print('normalize input')
        
        
    # SRC    
#     H_min_src,H_S_src=tensor_batch_geo_features(t_pcd_src,K=K)
#     H_min_src,H_S_src=tensor_batch_geo_features(t_pcd_src,K=K)
    H_S_src,H_min_src=tensor_batch_geo_features(t_pcd_src,K=K)
    # H_S=H_S
    
    # TGT
#     H_min_tgt_part,H_S_tgt_part=tensor_batch_geo_features(t_pcd_tgt_part,K=K)
    H_S_tgt_part,H_min_tgt_part=tensor_batch_geo_features(t_pcd_tgt_part,K=K)
    
    if verbose:
        print("H_min")
        print(H_min_src.shape,H_min_src.min(),H_min_src.max())
        print(H_min_tgt_part.shape,H_min_tgt_part.min(),H_min_tgt_part.max())
        print(H_S_src.shape,H_S_src.min(),H_S_src.max())
        print(H_S_tgt_part.shape,H_S_tgt_part.min(),H_S_tgt_part.max())
    
    
    
    # SRC
    offset1=(H_min_src.cpu().max() - H_min_src.cpu().min())*th1
#     thresh2=(H_min_tgt_part.cpu().min() + offset)    
    min1=H_min_src.cpu().min()
    thresh1=(H_min_src.cpu().min() + offset1)        
    thidx=(H_min_src >= thresh1)  
#     sel_idx_src=((H_min_src >= thresh) & (H_min_src < 1)).nonzero(as_tuple=True)
    sel_idx_src=((H_min_src >= thresh1)).nonzero(as_tuple=True)

    sel_idx_src=torch.topk(H_min_src,C)[1]
    
    
    # TGT
    offset2=(H_min_tgt_part.cpu().max() - H_min_tgt_part.cpu().min())*th2
    min2=H_min_tgt_part.cpu().min()
    thresh2=(H_min_tgt_part.cpu().min() + offset2)
    thidx=(H_min_tgt_part >= thresh2)    
#     sel_idx_tgt_part=((H_min_tgt_part >= thresh2) & (H_min_tgt_part < 1)).nonzero(as_tuple=True)
    sel_idx_tgt_part=((H_min_tgt_part >= thresh2)).nonzero(as_tuple=True)
    
    sel_idx_tgt_part=torch.topk(H_min_tgt_part,C)[1]

    if verbose:
        print('src th',thresh1)
        print('th1',th1,thresh1,offset1,min1)    
        print('tgt th',thresh2)
        print('th2',th2,thresh2,offset2,min2)    
    
#     print('indices',sel_idx_src[0].shape,sel_idx_tgt_part[0].shape)
#     print(sel_idx_src[0])
    
#     # TOP-select
#     # SRC
#     thresh=0
#     thidx=(H_min_src >= thresh) & (H_min_src < 1)
#     print(thidx.nonzero())
# #     thresh=H_min_src.cpu()[thidx.T.cpu()].max()*th1
#     thresh=H_min_src.cpu()[thidx.cpu()].max()*th1
# #     print(thresh)
#     sel_idx_src=((H_min_src >= thresh) & (H_min_src < 1)).nonzero(as_tuple=True)
    
#     # TGT
#     thresh=0
#     thidx=(H_min_tgt_part >= thresh) & (H_min_tgt_part < 1)
# #     print(thidx.T.nonzero())
# #     thresh=H_min_tgt_part.cpu()[thidx.T.cpu()].max()*th2
#     thresh=H_min_tgt_part.cpu()[thidx.cpu()].max()*th2
#     sel_idx_tgt_part=((H_min_tgt_part >= thresh) & (H_min_tgt_part < 1)).nonzero(as_tuple=True)

    # TOPK selection
    

    device=t_pcd_src.device
#     print(device)
#     print(t_pcd_src.shape)
    # SRC
#     print(sel_idx_src[0])
    t_src_th_vec_topk=t_pcd_src.index_select(dim=2,index=sel_idx_src[0].to(device))
    t_src_th_vec_topk2=t_src_th_vec_topk.permute(0,2,1)   
    
    
#     TGT
#     print(sel_idx_tgt_part[0])
    t_tgt_part_vec_topk=t_pcd_tgt_part.index_select(dim=2,index=sel_idx_tgt_part[0].to(device))
    t_tgt_part_vec_topk2=t_tgt_part_vec_topk.permute(0,2,1)
#     print('vec shape')
#     print(t_tgt_part_vec_topk2)
#     print(t_src_th_vec_topk2.shape,t_tgt_part_vec_topk2.shape)
    
#     print

    R_ransac_part=None

    return R_ransac_part, t_src_th_vec_topk2, t_tgt_part_vec_topk2, sel_idx_src, sel_idx_tgt_part, H_min_src, H_min_tgt_part


# +
class KeyPointNet_mod(nn.Module):
    def __init__(self, num_keypoints):
        super(KeyPointNet_mod, self).__init__()
        self.num_keypoints = num_keypoints

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        batch_size, num_dims, num_points = src_embedding.size()
        src_norm = torch.norm(src_embedding, dim=1, keepdim=True)
        tgt_norm = torch.norm(tgt_embedding, dim=1, keepdim=True)
        
#         print('keys')
#         print(self.num_keypoints)
#         print(src_norm.shape)
#         print(src.shape,src_embedding.shape)
        
        src_topk_idx = torch.topk(src_norm, k=self.num_keypoints, dim=2, sorted=False)[1]
        tgt_topk_idx = torch.topk(tgt_norm, k=self.num_keypoints, dim=2, sorted=False)[1]
        
        print('keypoint topk')
#         print(src.shape,tgt.shape)
#         print(src_topk_idx.shape,tgt_topk_idx.shape)
        
        harris_idx=True
        device=src.device
        if harris_idx:
            n_subsamples=self.num_keypoints
            T_estN, data1N,data2N, topk_H_src_idxN, topk_H_tgt_idxN, H_min_srcN, H_min_tgtN \
            =geometric_matching(src,tgt,th1=0.3,th2=0.3,C=n_subsamples)
            
#             print(topk_H_src_idxN.shape)
#             print(topk_H_tgt_idxN.shape)
            
            src_topk_idx=topk_H_src_idxN.to(device).unsqueeze(1)
            tgt_topk_idx=topk_H_tgt_idxN.to(device).unsqueeze(1)
            
            print('harris done',src_topk_idx.shape,tgt_topk_idx.shape)
        else:
            ""
            
        
        
        src_keypoints_idx = src_topk_idx.repeat(1, 3, 1)
        tgt_keypoints_idx = tgt_topk_idx.repeat(1, 3, 1)
        src_embedding_idx = src_topk_idx.repeat(1, num_dims, 1)
        tgt_embedding_idx = tgt_topk_idx.repeat(1, num_dims, 1)
        
#         print(src.shape)
#         print(src_topk_idx.shape,src_keypoints_idx.shape,src_embedding_idx.shape)

        src_keypoints = torch.gather(src, dim=2, index=src_keypoints_idx)
        tgt_keypoints = torch.gather(tgt, dim=2, index=tgt_keypoints_idx)
        
#         print(src_keypoints.shape)
        
#         print(src[:,:,0],src[:,:,1])
#         print(src_keypoints_idx[0])
#         print(src_embedding_idx[0])
#         print(src_topk_idx[0])
#         print(src_keypoints[0])
        
        src_embedding = torch.gather(src_embedding, dim=2, index=src_embedding_idx)
        tgt_embedding = torch.gather(tgt_embedding, dim=2, index=tgt_embedding_idx)
        return src_keypoints, tgt_keypoints, src_embedding, tgt_embedding


# +
class KeyPointNet(nn.Module):
    def __init__(self, num_keypoints):
        super(KeyPointNet, self).__init__()
        self.num_keypoints = num_keypoints

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        batch_size, num_dims, num_points = src_embedding.size()
        src_norm = torch.norm(src_embedding, dim=1, keepdim=True)
        tgt_norm = torch.norm(tgt_embedding, dim=1, keepdim=True)
        
#         print('keys')
#         print(self.num_keypoints)
#         print(src_norm.shape)
#         print(src.shape,src_embedding.shape)
        
        src_topk_idx = torch.topk(src_norm, k=self.num_keypoints, dim=2, sorted=False)[1]
        tgt_topk_idx = torch.topk(tgt_norm, k=self.num_keypoints, dim=2, sorted=False)[1]
        src_keypoints_idx = src_topk_idx.repeat(1, 3, 1)
        tgt_keypoints_idx = tgt_topk_idx.repeat(1, 3, 1)
        src_embedding_idx = src_topk_idx.repeat(1, num_dims, 1)
        tgt_embedding_idx = tgt_topk_idx.repeat(1, num_dims, 1)

        src_keypoints = torch.gather(src, dim=2, index=src_keypoints_idx)
        tgt_keypoints = torch.gather(tgt, dim=2, index=tgt_keypoints_idx)
        
        src_embedding = torch.gather(src_embedding, dim=2, index=src_embedding_idx)
        tgt_embedding = torch.gather(tgt_embedding, dim=2, index=tgt_embedding_idx)
        return src_keypoints, tgt_keypoints, src_embedding, tgt_embedding


# +
### Get all inliers

def get_inliers_keypoint_full(src, tgt, src_embedding, tgt_embedding,inliers):
    # TOPK inliers
    K=-1
    l_s0=[]
    l_e0=[]
    l_s1=[]
    l_e1=[]
    print(src.shape)
    for it in range(len(inliers)):
        ""
#         print(inliers[it].shape)
#         src_idx=inliers[it][:,0][:K].to(src.device)
#         tgt_idx=inliers[it][:,1][:K].to(src.device)
        src_idx=inliers[0][:,0][:K].to(src.device)
        tgt_idx=inliers[0][:,1][:K].to(src.device)        
        
#         print('iner',src_idx.shape,len(inliers))
        
        src_keypoints_k = torch.index_select(src[it], dim=1, index=src_idx)
        tgt_keypoints_k = torch.index_select(tgt[it], dim=1, index=tgt_idx)  
        
        src_embedding_k = torch.index_select(src_embedding[it], dim=1, index=src_idx)
        tgt_embedding_k = torch.index_select(tgt_embedding[it], dim=1, index=tgt_idx)        
        
        l_s0.append(src_keypoints_k.squeeze())
        l_s1.append(tgt_keypoints_k.squeeze())
        l_e0.append(src_embedding_k.squeeze())
        l_e1.append(tgt_embedding_k.squeeze())
#         print('it',it,tgt_embedding_k.squeeze().shape)
#         print('it',it,tgt_embedding_k.shape)
        
    src_t=torch.stack(l_s0)
    tgt_t=torch.stack(l_s1)
    src_E_t=torch.stack(l_e0)
    tgt_E_t=torch.stack(l_e1)
    print('SRC',src_t.shape)
    
    return src_t,tgt_t,src_E_t,tgt_E_t
    
#         l.append(torch.from_numpy(inliers))   


# +
def get_inliers_keypoint(src, tgt, src_embedding, tgt_embedding,inliers):
    # TOPK inliers
    K=30
    l_s0=[]
    l_e0=[]
    l_s1=[]
    l_e1=[]
    print(src.shape)
    for it in range(len(inliers)):
        ""
#         print(inliers[it].shape)
        src_idx=inliers[it][:,0][:K].to(src.device)
        tgt_idx=inliers[it][:,1][:K].to(src.device)
        
#         print('iner',src_idx.shape,len(inliers))
        
        src_keypoints_k = torch.index_select(src[it], dim=1, index=src_idx)
        tgt_keypoints_k = torch.index_select(tgt[it], dim=1, index=tgt_idx)  
        
        src_embedding_k = torch.index_select(src_embedding[it], dim=1, index=src_idx)
        tgt_embedding_k = torch.index_select(tgt_embedding[it], dim=1, index=tgt_idx)        
        
        l_s0.append(src_keypoints_k.squeeze())
        l_s1.append(tgt_keypoints_k.squeeze())
        l_e0.append(src_embedding_k.squeeze())
        l_e1.append(tgt_embedding_k.squeeze())
#         print('it',it,tgt_embedding_k.squeeze().shape)
#         print('it',it,tgt_embedding_k.shape)
        
    src_t=torch.stack(l_s0)
    tgt_t=torch.stack(l_s1)
    src_E_t=torch.stack(l_e0)
    tgt_E_t=torch.stack(l_e1)
    print('SRC',src_t.shape)
    
    return src_t,tgt_t,src_E_t,tgt_E_t
    
#         l.append(torch.from_numpy(inliers))   

# +
# ### open3d
# # http://www.open3d.org/docs/0.9.0/tutorial/Advanced/global_registration.html
# # from open3d.geometry import voxel_down_sample,estimate_normals

# def preprocess_point_cloud(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
#     pcd_down = pcd.voxel_down_sample(voxel_size)

#     radius_normal = voxel_size * 2
#     print(":: Estimate normal with search radius %.3f." % radius_normal)
#     pcd_down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

#     radius_feature = voxel_size * 5
#     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
#     pcd_fpfh = o3d.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh

# +
# ### RANSAC with features
# def prepare_dataset(src_xyz,tgt_xyz,voxel_size):
#     print(":: Load two point clouds and disturb initial pose.")
# #     source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
# #     target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
#     trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
#                              [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# #     source.transform(trans_init)
# #     draw_registration_result(source, target, np.identity(4))
#     print(src_xyz.shape)
#     src_pcd = o3d.geometry.PointCloud()
#     src_pcd.points = o3d.utility.Vector3dVector(src_xyz)
    
#     tgt_pcd = o3d.geometry.PointCloud()
#     tgt_pcd.points = o3d.utility.Vector3dVector(tgt_xyz)    
    
#     print('src_xyz',src_xyz.shape)

#     source_down, source_fpfh = preprocess_point_cloud(src_pcd, voxel_size)
#     target_down, target_fpfh = preprocess_point_cloud(tgt_pcd, voxel_size)
#     return src_pcd, tgt_pcd, source_down, target_down, source_fpfh, target_fpfh
# -



# +
# def execute_global_registration(source_down, target_down, source_fpfh,
#                                 target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 1.5
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("   Since the downsampling voxel size is %.3f," % voxel_size)
#     print("   we use a liberal distance threshold %.3f." % distance_threshold)
#     result = o3d.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
#         o3d.registration.TransformationEstimationPointToPoint(False), 4, [
#             o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#             o3d.registration.CorrespondenceCheckerBasedOnDistance(
#                 distance_threshold)
#         ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
#     return result

# +
### open3d
# http://www.open3d.org/docs/0.9.0/tutorial/Advanced/global_registration.html
# from open3d.geometry import voxel_down_sample,estimate_normals
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("   Since the downsampling voxel size is %.3f," % voxel_size)
#     print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def prepare_dataset(src_xyz,tgt_xyz,voxel_size):
#     print(":: Load two point clouds and disturb initial pose.")
#     source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
#     target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#     source.transform(trans_init)
#     draw_registration_result(source, target, np.identity(4))
    print(src_xyz.shape)
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_xyz)
    
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_xyz)    
    
    print('src_xyz',src_xyz.shape)

    source_down, source_fpfh = preprocess_point_cloud(src_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(tgt_pcd, voxel_size)
    return src_pcd, tgt_pcd, source_down, target_down, source_fpfh, target_fpfh

def preprocess_point_cloud(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
#     print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
#     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


# -

import open3d as o3d


# +
def get_ransac_inliers_with_features(src_keypts,tgt_keypts):
    ""
    source_np=src_keypts[:1].permute(0,1,2).cpu().detach().numpy()[0]
    target_np=tgt_keypts[:1].permute(0,1,2).cpu().detach().numpy()[0]
    
    voxel_size = 0.05  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(source_np,target_np,voxel_size)
    
    
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
#     print(result_ransac)
    
    trans=torch.from_numpy(result_ransac.transformation)
#     print(trans)
    pred_trans = torch.eye(4)[None].to(src_keypts.device)
    pred_trans[:, :4, :4] = torch.from_numpy(result_ransac.transformation)
    
#     print(reg_result)
    
    pred_R=pred_trans[:,:3,:3]
    
    inliers = np.array(result_ransac.correspondence_set)
    return pred_R, inliers

# R,inliers=get_ransac_inliers_with_features(src_keypts,tgt_keypts)

# +
# src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding=get_inliers_keypoint(src, tgt, src_embedding, tgt_embedding,inliers)
# -

import json
import sys
import argparse
import logging
import torch
import numpy as np
import importlib
import open3d as o3d
from tqdm import tqdm
from easydict import EasyDict as edict


def run_ransac_and_inliers(src_keypts,tgt_keypts):
    src_pcd = make_point_cloud(src_keypts[0].detach().cpu().numpy())
    tgt_pcd = make_point_cloud(tgt_keypts[0].detach().cpu().numpy())    
    corr = np.array([np.arange(src_keypts.shape[1]), np.arange(src_keypts.shape[1])])
#                 pred_inliers = np.where(pred_labels.detach().cpu().numpy() > 0)[1]
#                 corr = o3d.utility.Vector2iVector(corr[:, pred_inliers].T)
    corr = o3d.utility.Vector2iVector(corr.T)
    reg_result = o3d.registration.registration_ransac_based_on_correspondence(
        src_pcd, tgt_pcd, corr,
        max_correspondence_distance=config.inlier_threshold,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.registration.RANSACConvergenceCriteria(max_iteration=5000, max_validation=5000)
    )
    inliers = np.array(reg_result.correspondence_set)
#                 pred_labels = torch.zeros_like(gt_labels)
#                 pred_labels[0, inliers[:, 0]] = 1
    pred_trans = torch.eye(4)[None].to(src_keypts.device)
    pred_trans[:, :4, :4] = torch.from_numpy(reg_result.transformation)        
    
    pred_R=pred_trans[:,:3,:3]
    return pred_R, inliers


# +
def ransac_align(src_keypts,tgt_keypts):
    src_pcd = make_point_cloud(src_keypts[0].detach().cpu().numpy())
    tgt_pcd = make_point_cloud(tgt_keypts[0].detach().cpu().numpy())    
    corr = np.array([np.arange(src_keypts.shape[1]), np.arange(src_keypts.shape[1])])
#                 pred_inliers = np.where(pred_labels.detach().cpu().numpy() > 0)[1]
#                 corr = o3d.utility.Vector2iVector(corr[:, pred_inliers].T)
    corr = o3d.utility.Vector2iVector(corr.T)
    reg_result = o3d.registration.registration_ransac_based_on_correspondence(
        src_pcd, tgt_pcd, corr,
        max_correspondence_distance=0.1,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.registration.RANSACConvergenceCriteria(max_iteration=5000, max_validation=5000)
    )
    inliers = np.array(reg_result.correspondence_set)
#                 pred_labels = torch.zeros_like(gt_labels)
#                 pred_labels[0, inliers[:, 0]] = 1
    pred_trans = torch.eye(4)[None].to(src_keypts.device)
    pred_trans[:, :4, :4] = torch.from_numpy(reg_result.transformation)        
    
#     pred_R=pred_trans[:,:3,:3]
#     return pred_R, inliers
    return pred_trans,inliers


# +
def get_ransac_inliers_(src0, tgt0, src_embedding0, tgt_embedding0):
#     print(src.shape,src_embedding.shape)
#     src=src.permute(0,2,1)
#     tgt=tgt.permute(0,2,1)
    ### Batch
    l=[]
    for it in range(src0.shape[0]):
        src=src0[it].unsqueeze(0)
        tgt=tgt0[it].unsqueeze(0)
        
        src_embedding=src_embedding0[it].unsqueeze(0)
#         tgt_embedding=tgt0_embedding[it].unsqueeze(0       ) 
        
        #####
        # Get ransac keypoint inliers on the batch
        # T1
#         pred_R, inliers=run_ransac_and_inliers(src.permute(0,2,1),tgt.permute(0,2,1))
        # T2
        # Get ransac keypoint on the first item
        pred_R, inliers=get_ransac_inliers_with_features(src0[0].unsqueeze(0).permute(0,2,1),\
                                               tgt0[0].unsqueeze(0).permute(0,2,1))
        print('# inliers',inliers.shape)
#         print('src',src.shape,src_embedding.shape)
    #     print(inliers)
        src_keypoints_idx=torch.from_numpy(inliers[:,0]).type(torch.IntTensor).to(src_embedding.device)
        tgt_keypoints_idx=torch.from_numpy(inliers[:,1]).type(torch.IntTensor).to(src_embedding.device)
    #     src_keypoints_idx=inliers[:,0]
    #     tgt_keypoints_idx=inliers[:,1]

    #     src_keypoints = torch.gather(src, dim=2, index=src_keypoints_idx)
    #     tgt_keypoints = torch.gather(tgt, dim=2, index=tgt_keypoints_idx)
#         src_keypoints = torch.index_select(src, dim=2, index=src_keypoints_idx)
#         tgt_keypoints = torch.index_select(tgt, dim=2, index=tgt_keypoints_idx)    

    #     src_embedding = torch.gather(src_embedding, dim=2, index=src_embedding_idx)
    #     tgt_embedding = torch.gather(tgt_embedding, dim=2, index=tgt_embedding_idx)
#         src_embedding = torch.index_select(src_embedding, dim=2, index=src_keypoints_idx)
#         tgt_embedding = torch.index_select(tgt_embedding, dim=2, index=tgt_keypoints_idx)
    
    #     return src_keypoints, tgt_keypoints, src_embedding, tgt_embedding
#         print(src_keypoints_idx, tgt_keypoints_idx)
#         l.append([src_keypoints_idx,tgt_keypoints_idx])
#         print(inliers.__class__)
#         print(inliers.shape)
#         l.append(torch.from_numpy(inliers[:50]))
        l.append(torch.from_numpy(inliers))
        
#     t_keys=torch.stack(l)
#     print(t_keys.shape)
#     print(l)
#     print

#     return src_keypoints_idx, tgt_keypoints_idx
    return l


# +
# import sys
# sys.path.append('/workspace/program_2')
# from PointDSC.evaluation.test_3DMatch import run_ransac_and_inliers

def get_ransac_inliers(src0, tgt0, src_embedding0, tgt_embedding0):
#     print(src.shape,src_embedding.shape)
#     src=src.permute(0,2,1)
#     tgt=tgt.permute(0,2,1)
    ### Batch
    l=[]
    for it in range(src0.shape[0]):
        src=src0[it].unsqueeze(0)
        tgt=tgt0[it].unsqueeze(0)
        
        src_embedding=src_embedding0[it].unsqueeze(0)
#         tgt_embedding=tgt0_embedding[it].unsqueeze(0       ) 
        
        #####
        # Get ransac keypoint inliers on the batch
        # T1
#         pred_R, inliers=run_ransac_and_inliers(src.permute(0,2,1),tgt.permute(0,2,1))
        # T2
        # Get ransac keypoint on the first item
        pred_R, inliers=run_ransac_and_inliers(src0[0].unsqueeze(0).permute(0,2,1),\
                                               tgt0[0].unsqueeze(0).permute(0,2,1))
        
        get_ransac_inliers2
        print('# inliers',inliers.shape)
#         print('src',src.shape,src_embedding.shape)
    #     print(inliers)
        src_keypoints_idx=torch.from_numpy(inliers[:,0]).type(torch.IntTensor).to(src_embedding.device)
        tgt_keypoints_idx=torch.from_numpy(inliers[:,1]).type(torch.IntTensor).to(src_embedding.device)
    #     src_keypoints_idx=inliers[:,0]
    #     tgt_keypoints_idx=inliers[:,1]

    #     src_keypoints = torch.gather(src, dim=2, index=src_keypoints_idx)
    #     tgt_keypoints = torch.gather(tgt, dim=2, index=tgt_keypoints_idx)
#         src_keypoints = torch.index_select(src, dim=2, index=src_keypoints_idx)
#         tgt_keypoints = torch.index_select(tgt, dim=2, index=tgt_keypoints_idx)    

    #     src_embedding = torch.gather(src_embedding, dim=2, index=src_embedding_idx)
    #     tgt_embedding = torch.gather(tgt_embedding, dim=2, index=tgt_embedding_idx)
#         src_embedding = torch.index_select(src_embedding, dim=2, index=src_keypoints_idx)
#         tgt_embedding = torch.index_select(tgt_embedding, dim=2, index=tgt_keypoints_idx)
    
    #     return src_keypoints, tgt_keypoints, src_embedding, tgt_embedding
#         print(src_keypoints_idx, tgt_keypoints_idx)
#         l.append([src_keypoints_idx,tgt_keypoints_idx])
#         print(inliers.__class__)
#         print(inliers.shape)
#         l.append(torch.from_numpy(inliers[:50]))
        l.append(torch.from_numpy(inliers))
        
#     t_keys=torch.stack(l)
#     print(t_keys.shape)
#     print(l)
#     print

#     return src_keypoints_idx, tgt_keypoints_idx
    return l


# +
### Default to pointnet
# add module for inlier test
# add module for randomizing seed

class PRNet_mod_full(nn.Module):
    def __init__(self, emb_nn='pointnet', attention='transformer', head='svd', emb_dims=512, num_keypoints=512, num_subsampled_points=768, num_iters=3, cycle_consistency_loss=0.1, feature_alignment_loss=0.1, discount_factor = 0.9, input_shape='bnc'):
        super(PRNet_mod_full, self).__init__()
        self.emb_dims = emb_dims
        self.num_keypoints = num_keypoints
        self.num_subsampled_points = num_subsampled_points
        self.num_iters = num_iters
        self.discount_factor = discount_factor
        self.feature_alignment_loss = feature_alignment_loss
        self.cycle_consistency_loss = cycle_consistency_loss
        self.input_shape = input_shape
        
        # backbone
        if emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')
        
        # co-attention
        if attention == 'identity':
            self.attention = Identity()
        elif attention == 'transformer':
            self.attention = Transformer(emb_dims=self.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
        else:
            raise Exception("Not implemented")
        
        # heat kernel
        self.temp_net = TemperatureNet(emb_dims=self.emb_dims, temp_factor=100)

        # pose-recovery
        if head == 'mlp':
            self.head = MLPHead(emb_dims=self.emb_dims)
        elif head == 'svd':
            self.head = SVDHead(emb_dims=self.emb_dims, cat_sampler='softmax')
        else:
            raise Exception('Not implemented')

        # topk 
        if self.num_keypoints != self.num_subsampled_points:
            self.keypointnet = KeyPointNet(num_keypoints=self.num_keypoints)
        else:
            self.keypointnet = Identity()

    def predict_embedding(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        # self-supervision
        src_embedding_p, tgt_embedding_p = self.attention(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        # select points
#         src, tgt, src_embedding, tgt_embedding = self.keypointnet(src, tgt, src_embedding, tgt_embedding)
        # select inliers
#         src_inlier, tgt_inlier=get_ransac_inliers(src, tgt, src_embedding, tgt_embedding)
        inliers=get_ransac_inliers(src, tgt, src_embedding, tgt_embedding)
        
        src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding=get_inliers_keypoint_full(src, tgt, src_embedding, tgt_embedding,inliers)
        
        print('inlier',src_inlier.shape)

        # inlier correspondence
        temperature, feature_disparity = self.temp_net(src_inlier_embedding, tgt_inlier_embedding)
#         temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)

        return src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding, temperature, feature_disparity
    
    # Single Pass Alignment Module for PRNet
    def spam(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.predict_embedding(*input)
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature)
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity

    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size*num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores

    def forward(self, *input):
        calculate_loss = False
        if len(input) == 2:
            src, tgt = input[0], input[1]
        elif len(input) == 3:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2][:, :3, :3], input[2][:, :3, 3].view(-1, 3)
            calculate_loss = True
        elif len(input) == 4:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2], input[3]
            calculate_loss = True

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)

        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_feature_alignment_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0

        for i in range(self.num_iters):
            # selection + correspondence + 
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, feature_disparity = self.spam(src, tgt)
            
            # Updated R_ab
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) + translation_ab_pred_i

            # Updated R_ba
            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) + translation_ba_pred_i

            if calculate_loss:
                # isometry
                loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                       + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor**i
            
                feature_alignment_loss = feature_disparity.mean() * self.feature_alignment_loss * self.discount_factor**i
                cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                           rotation_ba_pred_i, translation_ba_pred_i) \
                                         * self.cycle_consistency_loss * self.discount_factor**i

                scale_consensus_loss = 0
                total_feature_alignment_loss += feature_alignment_loss
                total_cycle_consistency_loss += cycle_consistency_loss
                total_loss = total_loss + loss + feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss
            
            if self.input_shape == 'bnc':
                # New correction with RANSAC
                
                
                src = transform.transform_point_cloud(src.permute(0, 2, 1), rotation_ab_pred_i, translation_ab_pred_i).permute(0, 2, 1)
            else:
                src = transform.transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)
            
        result = {'est_R': rotation_ab_pred,
                  'est_t': translation_ab_pred,
                  'est_T': transform.convert2transformation(rotation_ab_pred, translation_ab_pred),
                  'transformed_source': src}

        if calculate_loss:
            result['loss'] = total_loss
        return result

# +
import sys
sys.path.append('/workspace/multistage_v2')
from algo import calc_svd_with_H, calc_svd_from_matches, Harris_matching, match_chamfer, batch_pairwise_dist

def get_harris_points(src,tgt,src_embedding,tgt_embedding):
    ""
    # Key points guid ethe embedding

    # Calculate harris matching features
    T_est, data1,data2, topk_H_src_idx, topk_H_tgt_idx, H_min_src, H_min_tgt=Harris_matching(src,tgt,th1=0.8,th2=0.8)

    print('topk H',topk_H_src_idx[0].shape,topk_H_tgt_idx[0].shape) 
    t_src_Hmin_topk_i=H_min_src.gather(dim=0,index=topk_H_src_idx[0]).unsqueeze(0).unsqueeze(2)
    t_tgt_Hmin_topk_i=H_min_tgt.gather(dim=0,index=topk_H_tgt_idx[0]).unsqueeze(0).unsqueeze(2)

#         # correspondence and matching
    PP_i=batch_pairwise_dist(t_src_Hmin_topk_i,t_tgt_Hmin_topk_i)
    
    # specify number to find from the topk
    NUM_M=15
    MM_i=match_chamfer(PP_i,n_samples=NUM_M).squeeze(2)

    print(MM_i.shape,PP_i.shape)

    mm_tgt_idx_i=MM_i[0,:,1]
    mm_src_idx_i=MM_i[0,:,0]
    print(MM_i)
    
    print(topk_H_src_idx[0].shape)
    print(topk_H_tgt_idx[0].shape)
    
    # GEt the original indices
    it0_i=topk_H_src_idx[0].gather(dim=0,index=mm_src_idx_i)
    it1_i=topk_H_tgt_idx[0].gather(dim=0,index=mm_tgt_idx_i)
    
   

    t_src_mm_topk_i=src.index_select(dim=2,index=it0_i)
    t_tgt_mm_topk_i=tgt.index_select(dim=2,index=it1_i)

    t_src_embed_mm_topk_i=src_embedding.index_select(dim=2,index=it0_i)
    t_tgt_embed_mm_topk_i=tgt_embedding.index_select(dim=2,index=it1_i)    
    
    return t_src_mm_topk_i, t_tgt_mm_topk_i, t_src_embed_mm_topk_i, t_tgt_embed_mm_topk_i


# get_harris_points(src)
# src_embed=model.emb_nn(src_keypts.permute(0,2,1).to(device))
# tgt_embed=model.emb_nn(tgt_keypts.permute(0,2,1).to(device))

# +
### Default to pointnet
# add module for inlier test
# add module for randomizing seed

class PRNet_mod2(nn.Module):
    def __init__(self, emb_nn='pointnet', attention='transformer', head='svd', emb_dims=512, num_keypoints=512, num_subsampled_points=768, num_iters=3, cycle_consistency_loss=0.1, feature_alignment_loss=0.1, discount_factor = 0.9, input_shape='bnc'):
        super(PRNet_mod, self).__init__()
        self.emb_dims = emb_dims
        self.num_keypoints = num_keypoints
        self.num_subsampled_points = num_subsampled_points
        self.num_iters = num_iters
        self.discount_factor = discount_factor
        self.feature_alignment_loss = feature_alignment_loss
        self.cycle_consistency_loss = cycle_consistency_loss
        self.input_shape = input_shape
        
        # backbone
        if emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')
        
        # co-attention
        if attention == 'identity':
            self.attention = Identity()
        elif attention == 'transformer':
            self.attention = Transformer(emb_dims=self.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
        else:
            raise Exception("Not implemented")
        
        # heat kernel
        self.temp_net = TemperatureNet(emb_dims=self.emb_dims, temp_factor=100)

        # pose-recovery
        if head == 'mlp':
            self.head = MLPHead(emb_dims=self.emb_dims)
        elif head == 'svd':
            self.head = SVDHead(emb_dims=self.emb_dims, cat_sampler='softmax')
        else:
            raise Exception('Not implemented')

        # topk 
        if self.num_keypoints != self.num_subsampled_points:
            self.keypointnet = KeyPointNet(num_keypoints=self.num_keypoints)
        else:
            self.keypointnet = Identity()

    def predict_embedding(self, *input):
        srco = input[0]
        tgto = input[1]
        src_embeddingo = self.emb_nn(srco)
        tgt_embeddingo = self.emb_nn(tgto)

        # self-supervision
        src_embedding_p, tgt_embedding_p = self.attention(src_embeddingo, tgt_embeddingo)

        src_embeddingo = src_embeddingo + src_embedding_p
        tgt_embeddingo = tgt_embeddingo + tgt_embedding_p
        
        src,tgt,src_embedding,tgt_embedding=get_harris_points(srco,tgto,src_embeddingo,tgt_embeddingo)

        # select points
#         src, tgt, src_embedding, tgt_embedding = self.keypointnet(src, tgt, src_embedding, tgt_embedding)
        # select inliers
#         src_inlier, tgt_inlier=get_ransac_inliers(src, tgt, src_embedding, tgt_embedding)
#         inliers=get_ransac_inliers(src, tgto, src_embedding, tgt_embedding)
#         get_ransac_inliers_with_features
        src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding=get_inliers_keypoint(src, tgt, src_embedding, tgt_embedding,inliers)
        
        src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding=get_inliers_keypoint(src, tgt, src_embedding, tgt_embedding,inliers)
        
        print('inlier',src_inlier.shape)

        # inlier correspondence
        temperature, feature_disparity = self.temp_net(src_inlier_embedding, tgt_inlier_embedding)
#         temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)

        return src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding, temperature, feature_disparity
    
    # Single Pass Alignment Module for PRNet
    def spam(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.predict_embedding(*input)
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature)
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity

    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size*num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores

    def forward(self, *input):
        calculate_loss = False
        if len(input) == 2:
            src, tgt = input[0], input[1]
        elif len(input) == 3:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2][:, :3, :3], input[2][:, :3, 3].view(-1, 3)
            calculate_loss = True
        elif len(input) == 4:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2], input[3]
            calculate_loss = True

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)

        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_feature_alignment_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0

        for i in range(self.num_iters):
            # selection + correspondence + 
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, feature_disparity = self.spam(src, tgt)
            
            # Updated R_ab
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) + translation_ab_pred_i

            # Updated R_ba
            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) + translation_ba_pred_i

            if calculate_loss:
                # isometry
                loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                       + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor**i
            
                feature_alignment_loss = feature_disparity.mean() * self.feature_alignment_loss * self.discount_factor**i
                cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                           rotation_ba_pred_i, translation_ba_pred_i) \
                                         * self.cycle_consistency_loss * self.discount_factor**i

                scale_consensus_loss = 0
                total_feature_alignment_loss += feature_alignment_loss
                total_cycle_consistency_loss += cycle_consistency_loss
                total_loss = total_loss + loss + feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss
            
            if self.input_shape == 'bnc':
                src = transform.transform_point_cloud(src.permute(0, 2, 1), rotation_ab_pred_i, translation_ab_pred_i).permute(0, 2, 1)
            else:
                src = transform.transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)
            
        result = {'est_R': rotation_ab_pred,
                  'est_t': translation_ab_pred,
                  'est_T': transform.convert2transformation(rotation_ab_pred, translation_ab_pred),
                  'transformed_source': src}

        if calculate_loss:
            result['loss'] = total_loss
        return result
# -



# +
### Default to pointnet
# add module for inlier test
# add module for randomizing seed

class PRNet_mod_ransac(nn.Module):
    def __init__(self, emb_nn='pointnet', attention='transformer', head='svd', emb_dims=512, num_keypoints=512, num_subsampled_points=768, num_iters=3, cycle_consistency_loss=0.1, feature_alignment_loss=0.1, discount_factor = 0.9, input_shape='bnc'):
        super(PRNet_mod_ransac, self).__init__()
        self.emb_dims = emb_dims
        self.num_keypoints = num_keypoints
        self.num_subsampled_points = num_subsampled_points
        self.num_iters = num_iters
        self.discount_factor = discount_factor
        self.feature_alignment_loss = feature_alignment_loss
        self.cycle_consistency_loss = cycle_consistency_loss
        self.input_shape = input_shape
        
        # backbone
        if emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')
        
        # co-attention
        if attention == 'identity':
            self.attention = Identity()
        elif attention == 'transformer':
            self.attention = Transformer(emb_dims=self.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
        else:
            raise Exception("Not implemented")
        
        # heat kernel
        self.temp_net = TemperatureNet(emb_dims=self.emb_dims, temp_factor=100)

        # pose-recovery
        if head == 'mlp':
            self.head = MLPHead(emb_dims=self.emb_dims)
        elif head == 'svd':
#             self.head = SVDHead(emb_dims=self.emb_dims, cat_sampler='softmax')
            self.head = SVDHead(emb_dims=self.emb_dims, cat_sampler='gumbel_softmax')
        else:
            raise Exception('Not implemented')

        # topk 
        if self.num_keypoints != self.num_subsampled_points:
            self.keypointnet = KeyPointNet(num_keypoints=self.num_keypoints)
        else:
            self.keypointnet = Identity()

    def predict_embedding(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        # self-supervision
        src_embedding_p, tgt_embedding_p = self.attention(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        # select points
#         src, tgt, src_embedding, tgt_embedding = self.keypointnet(src, tgt, src_embedding, tgt_embedding)
        # select inliers
#         src_inlier, tgt_inlier=get_ransac_inliers(src, tgt, src_embedding, tgt_embedding)
        inliers=get_ransac_inliers_(src, tgt, src_embedding, tgt_embedding)
        
        src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding=get_inliers_keypoint_full(src, tgt, src_embedding, tgt_embedding,inliers)
        
        print('inlier',src_inlier.shape)

        # inlier correspondence
        temperature, feature_disparity = self.temp_net(src_inlier_embedding, tgt_inlier_embedding)
#         temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)

        return src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding, temperature, feature_disparity
    
    # Single Pass Alignment Module for PRNet
    def spam(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.predict_embedding(*input)
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature)
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity

    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size*num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores

    def forward(self, *input):
        calculate_loss = False
        if len(input) == 2:
            src, tgt = input[0], input[1]
        elif len(input) == 3:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2][:, :3, :3], input[2][:, :3, 3].view(-1, 3)
            calculate_loss = True
        elif len(input) == 4:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2], input[3]
            calculate_loss = True

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)

        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_feature_alignment_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0

        for i in range(self.num_iters):
            # selection + correspondence + 
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, feature_disparity = self.spam(src, tgt)
            
            # Updated R_ab
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) + translation_ab_pred_i

            # Updated R_ba
            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) + translation_ba_pred_i

            if calculate_loss:
                # isometry
                loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                       + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor**i
            
                feature_alignment_loss = feature_disparity.mean() * self.feature_alignment_loss * self.discount_factor**i
                cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                           rotation_ba_pred_i, translation_ba_pred_i) \
                                         * self.cycle_consistency_loss * self.discount_factor**i

                scale_consensus_loss = 0
                total_feature_alignment_loss += feature_alignment_loss
                total_cycle_consistency_loss += cycle_consistency_loss
                total_loss = total_loss + loss + feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss
            
            if self.input_shape == 'bnc':
                src = transform.transform_point_cloud(src.permute(0, 2, 1), rotation_ab_pred_i, translation_ab_pred_i).permute(0, 2, 1)
            else:
                src = transform.transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)
            
        result = {'est_R': rotation_ab_pred,
                  'est_t': translation_ab_pred,
                  'est_T': transform.convert2transformation(rotation_ab_pred, translation_ab_pred),
                  'transformed_source': src}

        if calculate_loss:
            result['loss'] = total_loss
        return result


# +
### Default to pointnet
# add module for inlier test
# add module for randomizing seed

class PRNet_mod(nn.Module):
    def __init__(self, emb_nn='pointnet', attention='transformer', head='svd', emb_dims=512, num_keypoints=512, num_subsampled_points=768, num_iters=3, cycle_consistency_loss=0.1, feature_alignment_loss=0.1, discount_factor = 0.9, input_shape='bnc'):
        super(PRNet_mod, self).__init__()
        self.emb_dims = emb_dims
        self.num_keypoints = num_keypoints
        self.num_subsampled_points = num_subsampled_points
        self.num_iters = num_iters
        self.discount_factor = discount_factor
        self.feature_alignment_loss = feature_alignment_loss
        self.cycle_consistency_loss = cycle_consistency_loss
        self.input_shape = input_shape
        
        # backbone
        if emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')
        
        # co-attention
        if attention == 'identity':
            self.attention = Identity()
        elif attention == 'transformer':
            self.attention = Transformer(emb_dims=self.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
        else:
            raise Exception("Not implemented")
        
        # heat kernel
        self.temp_net = TemperatureNet(emb_dims=self.emb_dims, temp_factor=100)

        # pose-recovery
        if head == 'mlp':
            self.head = MLPHead(emb_dims=self.emb_dims)
        elif head == 'svd':
            self.head = SVDHead(emb_dims=self.emb_dims, cat_sampler='softmax')
        else:
            raise Exception('Not implemented')

        # topk 
        if self.num_keypoints != self.num_subsampled_points:
            self.keypointnet = KeyPointNet(num_keypoints=self.num_keypoints)
        else:
            self.keypointnet = Identity()

    def predict_embedding(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        # self-supervision
        src_embedding_p, tgt_embedding_p = self.attention(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        # select points
#         src, tgt, src_embedding, tgt_embedding = self.keypointnet(src, tgt, src_embedding, tgt_embedding)
        # select inliers
#         src_inlier, tgt_inlier=get_ransac_inliers(src, tgt, src_embedding, tgt_embedding)
        inliers=get_ransac_inliers(src, tgt, src_embedding, tgt_embedding)
        
        src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding=get_inliers_keypoint(src, tgt, src_embedding, tgt_embedding,inliers)
        
        print('inlier',src_inlier.shape)

        # inlier correspondence
        temperature, feature_disparity = self.temp_net(src_inlier_embedding, tgt_inlier_embedding)
#         temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)

        return src_inlier, tgt_inlier, src_inlier_embedding, tgt_inlier_embedding, temperature, feature_disparity
    
    # Single Pass Alignment Module for PRNet
    def spam(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.predict_embedding(*input)
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature)
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity

    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size*num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores

    def forward(self, *input):
        calculate_loss = False
        if len(input) == 2:
            src, tgt = input[0], input[1]
        elif len(input) == 3:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2][:, :3, :3], input[2][:, :3, 3].view(-1, 3)
            calculate_loss = True
        elif len(input) == 4:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2], input[3]
            calculate_loss = True

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)

        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_feature_alignment_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0

        for i in range(self.num_iters):
            # selection + correspondence + 
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, feature_disparity = self.spam(src, tgt)
            
            # Updated R_ab
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) + translation_ab_pred_i

            # Updated R_ba
            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) + translation_ba_pred_i

            if calculate_loss:
                # isometry
                loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                       + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor**i
            
                feature_alignment_loss = feature_disparity.mean() * self.feature_alignment_loss * self.discount_factor**i
                cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                           rotation_ba_pred_i, translation_ba_pred_i) \
                                         * self.cycle_consistency_loss * self.discount_factor**i

                scale_consensus_loss = 0
                total_feature_alignment_loss += feature_alignment_loss
                total_cycle_consistency_loss += cycle_consistency_loss
                total_loss = total_loss + loss + feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss
            
            if self.input_shape == 'bnc':
                src = transform.transform_point_cloud(src.permute(0, 2, 1), rotation_ab_pred_i, translation_ab_pred_i).permute(0, 2, 1)
            else:
                src = transform.transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)
            
        result = {'est_R': rotation_ab_pred,
                  'est_t': translation_ab_pred,
                  'est_T': transform.convert2transformation(rotation_ab_pred, translation_ab_pred),
                  'transformed_source': src}

        if calculate_loss:
            result['loss'] = total_loss
        return result


# +
import open3d as o3d
import torch

def make_point_cloud(pts):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd 

def make_feature(data, dim, npts):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    feature = o3d.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.astype('d').transpose()
    return feature

def estimate_normal(pcd, radius=0.06, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


# +
class PRNet_keys(nn.Module):
    def __init__(self, emb_nn='dgcnn', attention='transformer', head='svd', emb_dims=512, num_keypoints=512, \
                 num_subsampled_points=768, num_iters=3, cycle_consistency_loss=0.1, feature_alignment_loss=0.1, \
                 discount_factor = 0.9, input_shape='bnc', \
                 align_with_ransac=False, keypoint_harris=False, inliers_ransac=False):
        super(PRNet_keys, self).__init__()
        self.emb_dims = emb_dims
        self.num_keypoints = num_keypoints
        self.num_subsampled_points = num_subsampled_points
        self.num_iters = num_iters
        self.discount_factor = discount_factor
        self.feature_alignment_loss = feature_alignment_loss
        self.cycle_consistency_loss = cycle_consistency_loss
        self.input_shape = input_shape
        
        # New options
        self.align_with_ransac=align_with_ransac
        self.keypoint_harris=keypoint_harris
        self.inliers_ransac=inliers_ransac
        
        if emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')

        if attention == 'identity':
            self.attention = Identity()
        elif attention == 'transformer':
            self.attention = Transformer(emb_dims=self.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
        else:
            raise Exception("Not implemented")

        self.temp_net = TemperatureNet(emb_dims=self.emb_dims, temp_factor=100)

        if head == 'mlp':
            self.head = MLPHead(emb_dims=self.emb_dims)
        elif head == 'svd':
            if self.inliers_ransac:
#                 self.head_ransacInliers = SVDHead_mod(emb_dims=self.emb_dims, cat_sampler='gumbel_softmax')
                self.head = SVDHead_mod(emb_dims=self.emb_dims, cat_sampler='gumbel_softmax')
            else:
#                 self.head_ransacInliers = SVDHead(emb_dims=self.emb_dims, cat_sampler='gumbel_softmax')
                self.head = SVDHead(emb_dims=self.emb_dims, cat_sampler='gumbel_softmax')
        else:
            raise Exception('Not implemented')

        if self.num_keypoints != self.num_subsampled_points:
            if self.keypoint_harris:
                self.keypointnet = KeyPointNet_mod(num_keypoints=self.num_keypoints)
            else:
                self.keypointnet = KeyPointNet(num_keypoints=self.num_keypoints)
        else:
            self.keypointnet = Identity()
            
            
        ### memory
        self.mem_embedding={}

    def predict_embedding(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.attention(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        # Keypoints selection
        src, tgt, src_embedding, tgt_embedding = self.keypointnet(src, tgt, src_embedding, tgt_embedding)

        temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)

        return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity
    
    # Single Pass Alignment Module for PRNet
    def spam(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.predict_embedding(*input)
        
        self.mem_embedding['embedding']=(src,tgt,src_embedding, tgt_embedding, temperature, feature_disparity)
        
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature)
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity

    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size*num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores

    def forward(self, *input):
        calculate_loss = False
        if len(input) == 2:
            src, tgt = input[0], input[1]
        elif len(input) == 3:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2][:, :3, :3], input[2][:, :3, 3].view(-1, 3)
            calculate_loss = True
        elif len(input) == 4:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2], input[3]
            calculate_loss = True

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)

        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_feature_alignment_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0
        
        total_mse_loss_H = 0
        for i in range(self.num_iters):
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, feature_disparity = self.spam(src, tgt)

            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) + translation_ab_pred_i

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) + translation_ba_pred_i
            
            # EVAL SPAM
            

            if calculate_loss:
                loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                       + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor**i
            
                feature_alignment_loss = feature_disparity.mean() * self.feature_alignment_loss * self.discount_factor**i
                cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                           rotation_ba_pred_i, translation_ba_pred_i) \
                                         * self.cycle_consistency_loss * self.discount_factor**i

                scale_consensus_loss = 0
                
                if self.inliers_ransac:
                    src0,tgt0,_,_,_,_=self.mem_embedding['embedding']
                    
                    print('inlier')
                    
                    def MSELoss(R_pred,R_ab):
                        batch=((R_pred-R_ab)**2)
                        B,_,_=batch.shape
                        batch_mse_loss=batch.view(B, -1).mean(1, keepdim=True)
                        print('b',batch_mse_loss.shape)

                        loss=((R_pred-R_ab)**2).mean()

                        return loss            
                    if (1):
#                         (src,tgt,src_embedding, tgt_embedding, temperature, feature_disparity)=model.mem_embedding['embedding']
                        scores=self.head.mem['scores']
                        scores_b=self.head.mem['scores_b']
#                         print('scores',scores.shape,scores_b.shape)
#                         print(src.shape,scores.shape,scores_b.shape)

                        src_corr = torch.matmul(tgt0, scores.transpose(2, 1).contiguous())
                        src_centered = src0 - src0.mean(dim=2, keepdim=True)
        
                        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
                        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

                        src_corr_b = torch.matmul(tgt0, scores_b.transpose(2, 1).contiguous())
                        src_corr_b_centered = src_corr_b - src_corr_b.mean(dim=2, keepdim=True)
                        H_b = torch.matmul(src_centered, src_corr_b_centered.transpose(2, 1).contiguous())

#                         print('H',H.shape,H_b.shape)
                        mse_loss_H=MSELoss(H,H_b)* self.discount_factor**i 
                        print('H',i,mse_loss_H)
                        
                        total_mse_loss_H += mse_loss_H
#                     src,tgt=src.permute(0,2,1),tgt.permute(0,2,1)
                
                
                total_feature_alignment_loss += feature_alignment_loss
                total_cycle_consistency_loss += cycle_consistency_loss
                total_loss = total_loss + loss + \
                    feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss + \
                    total_mse_loss_H
                
#                 print('GEOLOSS',F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity))
#                 print('FEALOSS',feature_disparity.mean() * self.feature_alignment_loss)
#                 print('CYCLELOSS'                ,cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
#                                                            rotation_ba_pred_i, translation_ba_pred_i) \
#                                          * self.cycle_consistency_loss * self.discount_factor**i)
            
            if self.input_shape == 'bnc':
                print('SRC',src.shape)
                # NEW
                if self.align_with_ransac:
                    print('RUN RANSAC')
                    pred_T,_=ransac_align(src.permute(0,2,1),tgt.permute(0,2,1))
                    
                    rotation_ab_pred_iR=pred_T[:,:3,:3]
                    translation_ab_pred_iR=pred_T[:,:3,3]
                    
                    src = transform.transform_point_cloud(src.permute(0, 2, 1), rotation_ab_pred_iR, translation_ab_pred_iR).permute(0, 2, 1)
                else:
                    src = transform.transform_point_cloud(src.permute(0, 2, 1), rotation_ab_pred_i, translation_ab_pred_i).permute(0, 2, 1)
            else:
                src = transform.transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

#         if self.input_shape == 'bnc':
#             src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)
            
        result = {'est_R': rotation_ab_pred,
                  'est_t': translation_ab_pred,
                  'est_T': transform.convert2transformation(rotation_ab_pred, translation_ab_pred),
                  'transformed_source': src}

        if calculate_loss:
            result['loss'] = total_loss
        return result


# +
class PRNet(nn.Module):
    def __init__(self, emb_nn='dgcnn', attention='transformer', head='svd', emb_dims=512, num_keypoints=512, \
                 num_subsampled_points=768, num_iters=3, cycle_consistency_loss=0.1, feature_alignment_loss=0.1, \
                 discount_factor = 0.9, input_shape='bnc', align_with_ransac=False):
        super(PRNet, self).__init__()
        self.emb_dims = emb_dims
        self.num_keypoints = num_keypoints
        self.num_subsampled_points = num_subsampled_points
        self.num_iters = num_iters
        self.discount_factor = discount_factor
        self.feature_alignment_loss = feature_alignment_loss
        self.cycle_consistency_loss = cycle_consistency_loss
        self.input_shape = input_shape
        
        # New options
        self.align_with_ransac=align_with_ransac
        
        if emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')

        if attention == 'identity':
            self.attention = Identity()
        elif attention == 'transformer':
            self.attention = Transformer(emb_dims=self.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
        else:
            raise Exception("Not implemented")

        self.temp_net = TemperatureNet(emb_dims=self.emb_dims, temp_factor=100)

        if head == 'mlp':
            self.head = MLPHead(emb_dims=self.emb_dims)
        elif head == 'svd':
            self.head = SVDHead(emb_dims=self.emb_dims, cat_sampler='softmax')
        else:
            raise Exception('Not implemented')

        if self.num_keypoints != self.num_subsampled_points:
            self.keypointnet = KeyPointNet(num_keypoints=self.num_keypoints)
        else:
            self.keypointnet = Identity()
    
    # Produce Embedding
    def predict_embedding(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.attention(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        # Select keyponts
        src, tgt, src_embedding, tgt_embedding = self.keypointnet(src, tgt, src_embedding, tgt_embedding)

        # Apply Heat kernel
        temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)

        return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity
    
    # Single Pass Alignment Module for PRNet
    def spam(self, *input):
        # Get keypoints and Heat
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.predict_embedding(*input)
        
        # SVD head calculation
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature)
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity

    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size*num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores

    def forward(self, *input):
        calculate_loss = False
        if len(input) == 2:
            src, tgt = input[0], input[1]
        elif len(input) == 3:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2][:, :3, :3], input[2][:, :3, 3].view(-1, 3)
            calculate_loss = True
        elif len(input) == 4:
            src, tgt, rotation_ab, translation_ab = input[0], input[1], input[2], input[3]
            calculate_loss = True

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)

        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_feature_alignment_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0

        for i in range(self.num_iters):
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, feature_disparity = self.spam(src, tgt)

            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) + translation_ab_pred_i

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) + translation_ba_pred_i

            if calculate_loss:
                loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                       + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor**i
            
                feature_alignment_loss = feature_disparity.mean() * self.feature_alignment_loss * self.discount_factor**i
                cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                           rotation_ba_pred_i, translation_ba_pred_i) \
                                         * self.cycle_consistency_loss * self.discount_factor**i

                scale_consensus_loss = 0
                total_feature_alignment_loss += feature_alignment_loss
                total_cycle_consistency_loss += cycle_consistency_loss
                total_loss = total_loss + loss + feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss
                
#                 print('GEOLOSS',F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity))
#                 print('FEALOSS',feature_disparity.mean() * self.feature_alignment_loss)
#                 print('CYCLELOSS'                ,cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
#                                                            rotation_ba_pred_i, translation_ba_pred_i) \
#                                          * self.cycle_consistency_loss * self.discount_factor**i)
            
            if self.input_shape == 'bnc':
                # NEW
                if self.align_with_ransac:
                    print('RUN RANSAC')
                    pred_T,_=ransac_align(src.permute(0,2,1),tgt.permute(0,2,1))
                    
                    rotation_ab_pred_iR=pred_T[:,:3,:3]
                    translation_ab_pred_iR=pred_T[:,:3,3]
                    
                    src = transform.transform_point_cloud(src.permute(0, 2, 1), rotation_ab_pred_iR, translation_ab_pred_iR).permute(0, 2, 1)
                else:
                    src = transform.transform_point_cloud(src.permute(0, 2, 1), rotation_ab_pred_i, translation_ab_pred_i).permute(0, 2, 1)
            else:
                src = transform.transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

        if self.input_shape == 'bnc':
            src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)
            
        result = {'est_R': rotation_ab_pred,
                  'est_t': translation_ab_pred,
                  'est_T': transform.convert2transformation(rotation_ab_pred, translation_ab_pred),
                  'transformed_source': src}

        if calculate_loss:
            result['loss'] = total_loss
        return result


# -

if __name__ == '__main__':
    model = PRNet()
    src = torch.tensor(10, 1024, 3)
    tgt = torch.tensor(10, 768, 3)
    rotation_ab, translation_ab = torch.tensor(10, 3, 3), torch.tensor(10, 3)
    src, tgt = src.to(device), tgt.to(device)
    rotation_ab, translation_ab = rotation_ab.to(device), translation_ab.to(device)
    rotation_ab_pred, translation_ab_pred, loss = model(src, tgt, rotation_ab, translation_ab)
