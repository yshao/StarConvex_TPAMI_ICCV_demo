#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

# +

#### Adds implementataion of sinkhorn
# from . rpmnet import match_features
from .. utils import square_distance, angle_difference


# -

def compute_affinity(self, beta, feat_distance, alpha=0.5):
    """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
    if isinstance(alpha, float):
        hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
    else:
        hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
    return hybrid_affinity


# +
# alpha=0.5

# feat_distance = match_features(template, source)
# affinity = compute_affinity(beta, feat_distance, alpha=alpha)

# # Compute weighted coordinates
# log_perm_matrix = sinkhorn(affinity, n_iters=3, slack=True)
# perm_matrix = torch.exp(log_perm_matrix)

# # Calc weighted matrix
# weighted_template = perm_matrix @ xyz_template / (torch.sum(\perm_matrix, dim=2, keepdim=True) + _EPS)
# -
def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
	"""Compute rigid transforms between two point sets

	Args:
		a (torch.Tensor): (B, M, 3) points
		b (torch.Tensor): (B, N, 3) points
		weights (torch.Tensor): (B, M)

	Returns:
		Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
	"""

	weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
	centroid_a = torch.sum(a * weights_normalized, dim=1)
	centroid_b = torch.sum(b * weights_normalized, dim=1)
	a_centered = a - centroid_a[:, None, :]
	b_centered = b - centroid_b[:, None, :]
	cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

	# Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
	# and choose based on determinant to avoid flips
	u, s, v = torch.svd(cov, some=False, compute_uv=True)
	rot_mat_pos = v @ u.transpose(-1, -2)
	v_neg = v.clone()
	v_neg[:, :, 2] *= -1
	rot_mat_neg = v_neg @ u.transpose(-1, -2)
	rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
	assert torch.all(torch.det(rot_mat) > 0)

	# Compute translation (uncenter centroid)
	translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

	transform = torch.cat((rot_mat, translation), dim=2)
	return transform


def match_features(feat_src, feat_ref, metric='l2'):
	""" Compute pairwise distance between features

	Args:
		feat_src: (B, J, C)
		feat_ref: (B, K, C)
		metric: either 'angle' or 'l2' (squared euclidean)

	Returns:
		Matching matrix (B, J, K). i'th row describes how well the i'th point
		 in the src agrees with every point in the ref.
	"""
	assert feat_src.shape[-1] == feat_ref.shape[-1]

	if metric == 'l2':
		dist_matrix = square_distance(feat_src, feat_ref)
	elif metric == 'angle':
		feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
		feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

		dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
	else:
		raise NotImplementedError

	return dist_matrix

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

# +
import torch
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
# from tensorboardX import SummaryWriter

def get_OPT_plan(x,y):
    # Create some large point clouds in 3D
    # x = torch.randn(100, 90, requires_grad=True).cuda()
    # y = torch.randn(100, 90).cuda()
    print("OPTPlan",x.shape,y.shape)
    
    NP=x.shape[1]
    
    x_weight =torch.softmax(torch.randn(NP, requires_grad=True).to(x.device),dim=-1)*NP
    y_weight = torch.softmax(torch.randn(NP, requires_grad=True).to(x.device),dim=-1)*NP
    # Define a Sinkhorn (~Wasserstein) loss between sampled measures
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    L = loss(x_weight, x,y_weight, y)  # By default, use constant weights = 1/number of samples
#     g_x_weight, = torch.autograd.grad(L, [x_weight])  # GeomLoss fully supports autograd!
# # print(g_x_weight)

    N, M, D = x.shape[0], y.shape[0], x.shape[1]  # Number of points, dimension
    p = 2
    blur = .05
    OT_solver = SamplesLoss(loss = "sinkhorn", p = p, blur = blur, 
                            debias = False, potentials = True)
    F, G = OT_solver(x_weight, x, y_weight, y)  # Dual potentials

    a_i, x_i = x_weight.view(N,1), x.view(N,1,D)
    b_j, y_j = y_weight.view(1,M), y.view(1,M,D)
    F_i, G_j = F.view(N,1), G.view(1,M)

    C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)  # (N,M) cost matrix
    eps = blur**p  # temperature epsilon
    P_ij = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)  # (N,M) transport plan
    
    
    
    return P_ij
# -

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
WIDTH=256
class TemperatureNet(nn.Module):
    def __init__(self, emb_dims, temp_factor):
        super(TemperatureNet, self).__init__()
        self.n_emb_dims = emb_dims
        self.temp_factor = temp_factor
        self.nn = nn.Sequential(nn.Linear(self.n_emb_dims, WIDTH),
                                nn.BatchNorm1d(WIDTH),
                                nn.ReLU(),
                                nn.Linear(WIDTH, WIDTH),
                                nn.BatchNorm1d(WIDTH),
                                nn.ReLU(),
                                nn.Linear(WIDTH, WIDTH),
                                nn.BatchNorm1d(WIDTH),
                                nn.ReLU(),                              
                                # add one more
#                                 nn.Linear(WIDTH, WIDTH),
#                                 nn.BatchNorm1d(WIDTH),
#                                 nn.ReLU(),
                                
                                nn.Linear(WIDTH, 1),
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
          
#             print('temperature',temperature.max(),temperature.min(),torch.isnan(temperature).any())
        
            temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
            
#             print('scores',scores.max(),scores.min(),torch.isnan(scores).any())
            scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
         
#             print('scores_max',scores.max(),scores.min(),torch.isnan(scores).any())
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
                  
#         print(torch.isnan(src).any())
#         print(torch.isnan(tgt).any())
# #         print(torch.isnan(src).any())
#         print(src.max(),src.min())
#         print('tgt',tgt.max(),tgt.min(),torch.isnan(tgt).any())
        
#         print('scores',scores_b.max(),scores_b.min(),scores_b.isnan().any())
#         print('src_corr_centered',src_corr_centered.max(),src_corr_centered.min(),torch.isnan(src_corr_centered).any())
        
        src_corr = torch.matmul(tgt, scores_b.transpose(2, 1).contiguous())
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

#         print('H',src_corr_centered.max(),src_corr_centered.min(),torch.isnan(src_corr_centered).any())
#         print('H',src_centered.max(),src_centered.min(),torch.isnan(src_centered).any())
#         print('H',H.max(),H.min(),torch.isnan(H).any())
    
        R = []

        for i in range(src.size(0)):
#             u, s, v = torch.svd(H[i])
            L=H[i]
            try:
                u, s, v = torch.svd(L)
            except Exception as e:                     # torch.svd may have convergence issues for GPU and CPU.
                print(src.shape)
                print(e)
                print(L)
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
def calcSVDFromH(H):
        R = []
        batch_size=H.size(0)
        print('batch_size')
        for i in range(H.size(0)):
#             u, s, v = torch.svd(H[i])
            L=H[i]
            try:
                u, s, v = torch.svd(L)
            except Exception as e:                     # torch.svd may have convergence issues for GPU and CPU.
                print(H.shape)
                print(e)
                print(L)
#                 print(src_corr)
                u, s, v = torch.svd(L + 1e-4*L.mean()*torch.rand(L.shape[0], L.shape[1]).to(L.device))
                
            
            r = torch.matmul(v, u.transpose(1, 0)).contiguous()
            r_det = torch.det(r).item()
            diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                              [0, 1.0, 0],
                                              [0, 0, r_det]]).astype('float32')).to(v.device)
            r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
            R.append(r)

        R = torch.stack(R, dim=0).to(H.device)

#         t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        print("R SVD",R)
        return R, R

# +
from jakteristics import compute_features
feature_names=['eigenvalue_sum', 'omnivariance', 'eigenentropy', 'anisotropy', 'planarity', 'linearity', 'PCA1', 'PCA2', 'surface_variation', 'sphericity', 'verticality', 'nx', 'ny', 'nz']
# # feature_names=['nx', 'ny', 'nz','eigenvalue_sum']
# features_pcd_src = compute_features(t_pcd_src0[0].permute(1,0).cpu().numpy().astype(float), search_radius=0.15,max_k_neighbors=20,feature_names=feature_names)
# # t_pcd_src0
# # t_pcd_src0[0].shape, features.shape
# features_pcd_tgt_part = compute_features(t_pcd_tgt_part0[0].permute(1,0).cpu().numpy().astype(float), search_radius=0.15,max_k_neighbors=20,feature_names=feature_names)



def get_geometric_features(pcd,K=20,R=1.5):
#     print('pcd',pcd.shape)
    features=compute_features(pcd.permute(1,0).detach().cpu().numpy().astype(float), search_radius=R,max_k_neighbors=K,feature_names=feature_names)
    eps=1e-6
    mat_det=features[:,1]
    mat_trace=features[:,0]

#     lam=(mat_det+eps)/(mat_trace+eps)**(1/3)
#     sharpness=(mat_det+eps) - (mat_trace+eps)**(3)
    mat_det=mat_det**(3)
    #mat_trace=mat_trace**(1/3)
    mat_trace=mat_trace
    lam=(mat_det+eps)/(mat_trace+eps)
    sharpness=(mat_det+eps) - (mat_trace+eps)
#     lam, sharpness, H_S_src, H_min_src
    return lam, sharpness

def tensor_batch_geo_features(t_pcd_src0,K=20,R=1.5):
    H_min=[]
    H_sharpness=[]
    for it in range(t_pcd_src0.shape[0]):
        pcd=t_pcd_src0[it]
#         print('pcd',pcd.shape)
        lam,sharpness=get_geometric_features(pcd,K=K,R=R)
        H_min.append(torch.from_numpy(lam))
        H_sharpness.append(torch.from_numpy(sharpness))
    H_min=torch.stack(H_min,dim=0)
    H_sharpness=torch.stack(H_sharpness,dim=0)

    return H_min, H_sharpness

# t_features_src_0,t_features_src_1=tensor_batch_geo_features(t_pcd_src0)


# +

def geometric_matching(t_pcd_src,t_pcd_tgt_part,K=20,C=50,th1=0.8,th2=0.0001, R=1.5, renormalize=False):
#     C=50
    verbose=False
    
    if (renormalize):
        print('normalize input')
        
        
    # SRC    
#     H_min_src,H_S_src=tensor_batch_geo_features(t_pcd_src,K=K)
#     H_min_src,H_S_src=tensor_batch_geo_features(t_pcd_src,K=K)
    H_S_src,H_min_src=tensor_batch_geo_features(t_pcd_src,K=K,R=R)
    # H_S=H_S
    
    # TGT
#     H_min_tgt_part,H_S_tgt_part=tensor_batch_geo_features(t_pcd_tgt_part,K=K)
    H_S_tgt_part,H_min_tgt_part=tensor_batch_geo_features(t_pcd_tgt_part,K=K,R=R)
    
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

    return R_ransac_part, t_src_th_vec_topk2, t_tgt_part_vec_topk2, sel_idx_src, sel_idx_tgt_part, (H_min_src, H_min_tgt_part, H_S_src, H_S_tgt_part)


# -

def normalize(input):    
    input += 1e-5  #For Numerical Stability
    stdv = input.std(dim=1,keepdim=True)
    input = (input - input.mean(dim=1,keepdim=True)) #/ np.std(input)        #0 mean 1 std
#     input = input / stdv #np.max(abs(input))
    if torch.isnan(torch.sum(input)):
        print("[Nan Values in Normalize is ::]", torch.isnan(np.sum(input)))
    return input        
def check_input(source):
    src_keypts=normalize(source).to(source.device)
    print(source.min(),source.max(),source.mean(),source.isnan().any())
    print(src_keypts.min(),src_keypts.max(),src_keypts.mean(),src_keypts.isnan().any())
#             tgt_keypts=normalize(tgt_keypts).to(source.device)



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
            
            for Rs in [1.5,1,0.75,1.75]:
                try:
                    n_subsamples=self.num_keypoints
                    T_estN, data1N,data2N, topk_H_src_idxN, topk_H_tgt_idxN,feats \
                    =geometric_matching(src,tgt,th1=0.3,th2=0.3,C=n_subsamples,R=Rs)

        #             print(topk_H_src_idxN.shape)
        #             print(topk_H_tgt_idxN.shape)

                    src_topk_idx=topk_H_src_idxN.to(device).unsqueeze(1)
                    tgt_topk_idx=topk_H_tgt_idxN.to(device).unsqueeze(1)

                    print('harris done',src_topk_idx.shape,tgt_topk_idx.shape)

                    ### Check harris stats
                    H_min_srcN, H_min_tgtN, H_S_srcN, H_S_tgtN = feats 

                    check_input(H_min_srcN)
                    check_input(H_min_tgtN)
                    check_input(H_S_srcN)
                    check_input(H_S_tgtN)
                    print('break out with',Rs)
                    break
                    
                except:
                    continue
        
        
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
        return src_keypoints, tgt_keypoints, src_embedding, tgt_embedding, feats


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
# import open3d as o3d
# def to_o3d_pcd(xyz):
#     """
#     Convert tensor/array to open3d PointCloud
#     xyz:       [N, 3]
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
#     return pcd

# def to_o3d_feats(embedding):
#     """
#     Convert tensor/array to open3d features
#     embedding:  [N, 3]
#     """
#     feats = o3d.registration.Feature()
#     feats.data = to_array(embedding).T
#     return feats
# -

from . benchmark_utils import *


def ransac_align_with_feats(src_pcd0,tgt_pcd0,src_feats0,tgt_feats0):
    print('ransac_align_with_feats')
    print(src_pcd0.shape,tgt_pcd0.shape)
    print(src_feats0.shape,tgt_feats0.shape)
    
    src_pcd = to_o3d_pcd(src_pcd0.permute(1,0))
    tgt_pcd = to_o3d_pcd(tgt_pcd0.permute(1,0))
    src_feats = to_o3d_feats(src_feats0[:,:1])
    tgt_feats = to_o3d_feats(tgt_feats0[:,:1])
    distance_threshold = 0.05
    ransac_n = 3
    reg_result = o3d.registration.registration_ransac_based_on_feature_matching(
        src_pcd, tgt_pcd, src_feats, tgt_feats,distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), ransac_n,
        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    print(reg_result)
    inliers = np.array(reg_result.correspondence_set)
#                 pred_labels = torch.zeros_like(gt_labels)
#                 pred_labels[0, inliers[:, 0]] = 1
    pred_trans = torch.eye(4)[None].to(src_pcd0.device)
    pred_trans[:, :4, :4] = torch.from_numpy(reg_result.transformation)
    return pred_trans,inliers


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
# -



def uniform_2_sphere(num: int = 3, Lambda: float=1.0):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
#     if num is not None:
#     phi = np.random.uniform(0.0, 2 * np.pi, num)
    phi = torch.distributions.uniform.Uniform(0,np.pi).sample([num]).squeeze()
#     cos_theta = np.random.uniform(-Lambda, Lambda, num)
    cos_theta=torch.distributions.uniform.Uniform(-1,1).sample([num]).squeeze()
#     else:
#         phi = np.random.uniform(0.0, 2 * np.pi)
#         cos_theta = np.random.uniform(-Lambda, Lambda)

    theta = torch.arccos(cos_theta)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack((x, y, z), dim=-1)


def get_ransac_inliers_iter(template,source,n_iter=10):
    ROT=template[0]
    R_flip=torch.eye(3).to(ROT.device)
    print('mu initial ROT',ROT.shape)
    print(R_flip)
    l_inliers=[]

    for it in range(n_iter):
        print('itera',it)
        TInput=torch.stack([ROT,ROT])
    #     print(TInput.shape)

        n_subsamples=1000
        T_estN, data1N,data2N, topk_H_src_idxN, topk_H_tgt_idxN, feats \
        =geometric_matching(template.permute(0,2,1),source.permute(0,2,1),th1=0.3,th2=0.3,C=n_subsamples)

        H_S_src=feats[0][:1].permute(1,0).to(ROT.device)
        H_S_tgt=feats[1][:1].permute(1,0).to(ROT.device)

        H_min_src=feats[2][:1].permute(1,0)
        H_min_tgt=feats[3][:1].permute(1,0)

        print(H_S_src.shape,H_S_tgt.shape )

    #     gg

        a=H_min_src.isnan().any()
        b=H_min_tgt.isnan().any()
        non_nan=a|b
#         print(non_nan)

    #     print(src_feats.shape)


        pred_trans,inliers=ransac_align_with_feats(ROT,source[0],H_min_src,H_min_tgt)
    #     print(pred_trans.shape, R_ab.shape)
#         print('===> RMatrix',pred_trans[0,:3,:3])
        R_flip=pred_trans[0,:3,:3].matmul(R_flip)

    #     print('flip',ROT.shape)
        ROT=pred_trans[0,:3,:3].matmul(ROT.permute(1,0)).permute(1,0)
    #     print(ROT.shape)
        print('---> mse',it,((R_flip-R_ab[0])**2).mean())

        l_inliers.append(torch.from_numpy(inliers))

    t_inliers=torch.cat(l_inliers)
    print('______')
    print('RFlip',R_flip,R_ab[0])
    return t_inliers


# +
class PRNet_keys_cvx(nn.Module):
    def __init__(self, emb_nn='dgcnn', attention='transformer', head='svd', emb_dims=512, num_keypoints=512, \
                 num_subsampled_points=768, num_iters=3, cycle_consistency_loss=0.1, feature_alignment_loss=0.1, \
                 discount_factor = 0.9, input_shape='bnc', \
                 align_with_ransac=False, keypoint_harris=False, inliers_ransac=False, \
                 temp_factor=1e12, cat_sampler='gumbel_softmax', \
                 Lambda=0.2, Mu=0.3, \
                ):
        super(PRNet_keys_cvx, self).__init__()
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
        
        self.temp_factor=temp_factor
        
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

        self.temp_net = TemperatureNet(emb_dims=self.emb_dims, temp_factor=self.temp_factor)
        print('TempFactor',self.temp_factor)

        if head == 'mlp':
            self.head = MLPHead(emb_dims=self.emb_dims)
        elif head == 'svd':
            if self.inliers_ransac:
#                 self.head_ransacInliers = SVDHead_mod(emb_dims=self.emb_dims, cat_sampler='gumbel_softmax')
                self.head = SVDHead_mod(emb_dims=self.emb_dims, cat_sampler=cat_sampler)
            else:
#                 self.head_ransacInliers = SVDHead(emb_dims=self.emb_dims, cat_sampler='gumbel_softmax')
                self.head = SVDHead(emb_dims=self.emb_dims, cat_sampler=cat_sampler)
        else:
            raise Exception('Not implemented')

        if self.num_keypoints != self.num_subsampled_points:
            if self.keypoint_harris:
                self.keypointnet = KeyPointNet_mod(num_keypoints=self.num_keypoints)
            else:
                self.keypointnet = KeyPointNet(num_keypoints=self.num_keypoints)
        else:
            self.keypointnet = Identity()
            
        ### Convex training ###
        # convexStart loss function
        class convexStarLoss(nn.Module):
            def __init__(self, Lambda, mu, n_samples, balancer, scaler=10):
                super(convexStarLoss, self).__init__()
                self.Lambda = nn.Parameter(torch.tensor([1.0 * Lambda]), requires_grad=True)
                self.mu = nn.Parameter(torch.tensor([1.0 * mu]), requires_grad=False)
                # self.mu = nn.Parameter(torch.tensor([1.0 * mu]), requires_grad=True)
                self.balancer = balancer
                self.n_samples = n_samples
                self.scaler = scaler
                
                
                # customize MSE
                def MSELoss(R_pred,R_ab):
                    batch=((R_pred-R_ab)**2)
                    B,_,_=batch.shape
#                     print('batch shape',batch.shape)
                    batch_mse_loss=batch.view(B, -1).mean(1, keepdim=True)
#                     print('b',batch_mse_loss.shape)

                    loss=((R_pred-R_ab)**2).mean()

                    return loss
    
                def CorrespondenceLoss(xyz_src, xyz_tgt,feats_src, feats_tgt,corres):
                    ""
                    return loss
                
                self.h_fn = MSELoss


            def forward(self, predictions, labels):
                h_fn=self.h_fn
                
                loss1 = h_fn(predictions, labels) # h_label
                loss2 = 0.

                print('shape',predictions.shape[2])
                mu = self.mu # mu >=0
                Lambda = 1 / (1 + torch.exp(-self.Lambda)) # 0<Lambda<1
                for _ in range(self.n_samples):
                    #original
                    noise = (torch.rand(predictions.size(), device=predictions.get_device())  - 0.5 ) / self.scaler
                    # H - 3D points
#                     noise = uniform_2_sphere(predictions.shape[2]).to(predictions.device)/self.scaler
                    # score_b
#                     noise = (uniform_2_sphere(predictions.shape[2]).to(predictions.device)-0.5)/self.scaler
#                     noise = noise.permute(1,0).repeat(2, 1, 1)
#                     lambda_noise = uniform_2_sphere(3,Lambda).to(predictions.device)/self.scaler
#                     print('noise',noise.shape,noise.device,labels.device)
#                     print(labels.shape,noise.shape)
                    noisy_labels = (labels + noise).softmax(dim=1)
#                     print('noisy',noisy_labels.shape)
#                     print('noisy',Lambda,labels,noise,noisy_labels.shape,noisy_labels)
                    lambda_noisy_labels = (labels + Lambda*noise).softmax(dim=1)
#                     print('lambda noisy',lambda_noisy_labels.shape,noisy_labels.shape)
                    h_noise_label = h_fn(predictions, noisy_labels)
                    h_lambda_noise_label = h_fn(predictions, lambda_noisy_labels)

                    # print(loss1)
                    # print(h_lambda_noise_label)


                    # equ1
                    term1 = torch.clamp(loss1 - h_lambda_noise_label, min=0)
                    # equ2
                    term2 = torch.clamp(loss1 - h_noise_label + mu * torch.sum((noisy_labels - labels)**2) / 2, min=0)[0]
                    # equ3
                    term3 = torch.clamp(mu * Lambda * (1 - Lambda) * torch.sum((noisy_labels - labels)**2) / 2 + h_lambda_noise_label - (1 - Lambda) * loss1 - Lambda * h_noise_label, min=0)[0]
                    # term3 = torch.clamp(mu * (1 - Lambda) * torch.sum((noisy_labels - labels)**2) / 2 +  h_lambda_noise_label - h_noise_label, min=0)[0]
                    # print(term2)
                    loss2 += term1 + term2 + term3

                return loss1, loss2, loss1 + self.balancer * loss2        
#         print('Lambda',Lambda)
#         print('Mu',Mu)
        self.convex_sinkhornloss=convexStarLoss(Lambda=Lambda,mu=Mu,n_samples=10,balancer=0.5)        
        
        ### memory
        self.mem_embedding={}
    
    def convexMSELoss(self,p,q):
        
        mseLoss=self.convex_sinkhornloss(p,q)
        return mseLoss
        

    def predict_embedding(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.attention(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        # Keypoints selection
        src, tgt, src_embedding, tgt_embedding, feats = self.keypointnet(src, tgt, src_embedding, tgt_embedding)
        # Save features
        self.mem_embedding['feats']=feats
        
        H_min_srcN, H_min_tgtN, H_S_srcN, H_S_tgtN = feats        
        
        print('H_S',H_S_srcN.shape,H_S_tgtN.shape)
#         pred_trans,inliers=ransac_align_with_feats(input[0].permute(0,2,1)[0],input[1].permute(0,2,1)[0],\
#                                                    H_S_srcN[0],H_S_tgtN[0])
        print('---','Get RANSAC point sets')
        template=src
        source=tgt
        H_S_src=feats[0].to(template.device).permute(1,0)[:,:1]
        H_S_tgt=feats[1].to(template.device).permute(1,0)[:,:1]
        H_min_src=feats[2].to(template.device).permute(1,0)[:,:1]
        H_min_tgt=feats[3].to(template.device).permute(1,0)[:,:1]    
        H_feats_src=torch.column_stack([H_S_src,H_min_src])
        H_feats_tgt=torch.column_stack([H_S_tgt,H_min_tgt])  
        
#         pred_trans,inliers=ransac_align_with_feats(template[0].detach(),source[0].detach(), \
#                                                    H_feats_src,H_feats_tgt)
# #         pred_trans        
#         # Measure of Fit
        
#         a=H_min_src.isnan().any()
#         b=H_min_tgt.isnan().any()
#         print(a|b)
        
        
#         self.mem_embedding['ransacInliers']={}
#         self.mem_embedding['ransacInliers']['pred_trans']=pred_trans
#         self.mem_embedding['ransacInliers']['inliers']=inliers
#         self.mem_embedding['ransacInliers']['any_nan']=a|b
#         print('ransac outputs:')
#         print(pred_trans.shape,inliers.shape)
        
# #         print('get RANSAC inliers')
# #         t=get_ransac_inliers_iter(template,source,n_iter=10)
        
#         del template
#         del source        
        
        
        
#         print(pred_trans.shape,inliers.shape)

#         print(src.isnan().nonzero().shape,tgt.isnan().nonzero().shape)
#         print(src_embedding.isnan().nonzero().shape,tgt_embedding.isnan().nonzero().shape)
        
#         print(src_embedding,tgt_embedding)
        temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)
        
#         print(feature_disparity)

        return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity
    
    # Single Pass Alignment Module for PRNet
    def spam(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.predict_embedding(*input)
        
        print('SVDtemp',temperature)
#         temperature = temperature.clamp(min=1/self.temp_factor)
        
        self.mem_embedding['embedding']=(src,tgt,src_embedding, tgt_embedding, temperature, feature_disparity)
        
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature)
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity
    
    def get_OPT(self,*input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.mem_embedding['embedding']
        
#         print(src_embedding.shape,tgt_embedding.shape)
        l_plans=[]
        for it in range(src_embedding.shape[0]):
            P_ij=get_OPT_plan(src_embedding[it],src_embedding[it])
#             print('P_ij.shape',P_ij.shape)
            l_plans.append(P_ij)
        t_P_ij=torch.stack(l_plans)
        print('OPT stack tensor',t_P_ij.shape)
        return t_P_ij

    #### TEST & TEST
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
                    if (1):
                        device=src.device
                        (src,tgt,src_embedding, tgt_embedding, temperature, feature_disparity)=self.mem_embedding['embedding']
                        scores=self.head.mem['scores']
                        
                        # RANSAC_scores
                        scores_b=self.head.mem['scores_b']
#                         print('scores',scores.shape,scores_b.shape)
#                         print(src.shape,scores.shape,scores_b.shape)
                        print("SINKHORN")
                        print('scores',scores.nonzero().shape)
                        print('scores',scores[scores >= 1].shape)
                        print('scores RANSAC',scores_b.nonzero().shape)
                        print('scores',scores_b[scores_b >= 1].shape)
#                         scores_b2=self.mem_embedding['ransacInliers']['inliers']
#                         print(scores_b2,scores_b2.shape)
# #                         scores_b2=scores_b2
#                         ###### 512 @@@@@@@
#                         num_points=512
#                         na_ones=np.zeros((num_points,num_points))
                    
#                         for r in range(scores_b2.shape[0]):
#                             na_ones[scores_b2[r,0],scores_b2[r,1]]=1
                        
#                         scores_b2=torch.from_numpy(na_ones)
# #                         scores_b2=torch.stack(b2_inliers)


#                         print('scores INLIERS',scores_b2,scores_b2.shape)
#                         print('scores INLIERS',scores_b2[scores_b2 >= 1].shape)
# #                         print('sum INLIERS',scores_b2.sum())
# #                         print('means INLIERS',scores_b2[scores_b2 >= 1].mean())
                        
#                         scores_=torch.zeros(512,512).to(src.device)
#                         scores_[scores_b2]=1
#                         print(scores.shape,scores.sum(),scores.mean())
#                         scores_b2=torch.stack([scores_,scores_])
                        
                        # weights
#                         print(scores_b.reshape(1024,3).shape)
                        print('weights means', scores[scores >=1].mean())
                        print('weights means RANSAC', scores_b[scores_b >=1].mean())
                        print('sum',scores.sum())
                        print('sum RANSAC',scores_b.sum())

                        scores_opt=self.get_OPT(src_embedding,tgt_embedding)
                        
                        # 1 to 1
#                         scores_b=scores_b[scores_b >= 1].reshape(2,-1,3)
#                         scores_b=scores_opt[scores_opt >= 1].view(2,-1,3)
#                         scores_opt[scores_opt < 1]=0
                        scores_b=scores_opt
                        
                        print('scores batch OPT',scores_b.nonzero().shape)
                        print('scores',scores_b[scores_b >= 1].shape)
#                         print(scores_b.shape)
                        
                        print('weights means OPT', scores_b[scores_b >=1].mean())
                        print('sum OPT',scores_b.sum())
        
                        src_centered = src0 - src0.mean(dim=2, keepdim=True)
        
                        src_corr = torch.matmul(tgt0, scores.transpose(2, 1).contiguous())
                        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)                        
                        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

                        src_corr_b = torch.matmul(tgt0, scores_b.transpose(2, 1).contiguous())                    
                        src_corr_b_centered = src_corr_b - src_corr_b.mean(dim=2, keepdim=True)
                        H_b = torch.matmul(src_centered, src_corr_b_centered.transpose(2, 1).contiguous())
                        
#                         src_corr_b2 = torch.matmul(tgt0, scores_b2.transpose(2, 1).contiguous())                    
#                         src_corr_b2_centered = src_corr_b2 - src_corr_b2.mean(dim=2, keepdim=True)
#                         H_b2 = torch.matmul(src_centered, src_corr_b2_centered.transpose(2, 1).contiguous())                        

                        loss1,loss2,mse_loss_H=self.convexMSELoss(H,H_b)
                        print('H',i,mse_loss_H)                        
                        
              # If H_b is too huge
                        if (mse_loss_H.mean().item() > 1e6):
                            print('<---------- RANSAC - H_b ------------->',H_b)
                            scores_b=self.head.mem['scores_b']
                            src_corr_b = torch.matmul(tgt0, scores_b.transpose(2, 1).contiguous())                    
                            src_corr_b_centered = src_corr_b - src_corr_b.mean(dim=2, keepdim=True)
                            H_b = torch.matmul(src_centered, src_corr_b_centered.transpose(2, 1).contiguous())                           

#                         print('H',H.shape,H_b.shape)

                        loss1,loss2,mse_loss_H=self.convexMSELoss(H,H_b)

                        # precalc R_b
                        R_b=calcSVDFromH(H_b)
                        print('calcSVD',R_b)

#                         print('H',H.shape,H_b.shape)
#                         mse_loss_H=MSELoss(H,H_b)* self.discount_factor**i 
#                         print('H',i,mse_loss_H)
#                         loss1,loss2,mse_loss_H=self.convexMSELoss(src_corr_centered,src_corr_b_centered)
                        print('H,H_b',H,H_b)
                        print('loss',loss1,loss2,mse_loss_H)
                        
#                         print('H',i,mse_loss_H)        
                        mse_loss_H=mse_loss_H* self.discount_factor**i 
                        print('H',i,mse_loss_H,'discount',self.discount_factor**i)        
                        
#                         total_mse_loss_H += mse_loss_H
#                     src,tgt=src.permute(0,2,1),tgt.permute(0,2,1)
                
                total_mse_loss_H += mse_loss_H
                total_feature_alignment_loss += feature_alignment_loss
                total_cycle_consistency_loss += cycle_consistency_loss
                total_loss = total_loss + loss + \
                    feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss + \
                    total_mse_loss_H
                if (total_loss.isnan().any()):
                    print('PRINT NANs')
                    print(total_mse_loss_H)
                    print(total_feature_alignment_loss)
                    print(total_cycle_consistency_loss)
                    
                    print(total_loss)
#                 total_loss = total_loss + total_mse_loss_H
                
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
class PRNet_keys(nn.Module):
    def __init__(self, emb_nn='dgcnn', attention='transformer', head='svd', emb_dims=512, num_keypoints=512, \
                 num_subsampled_points=768, num_iters=3, cycle_consistency_loss=0.1, feature_alignment_loss=0.1, \
                 discount_factor = 0.9, input_shape='bnc', \
                 align_with_ransac=False, keypoint_harris=False, inliers_ransac=False, \
                 temp_factor=1e12, cat_sampler='gumbel_softmax'
                ):
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
        
        self.temp_factor=temp_factor
        
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

        self.temp_net = TemperatureNet(emb_dims=self.emb_dims, temp_factor=self.temp_factor)

        if head == 'mlp':
            self.head = MLPHead(emb_dims=self.emb_dims)
        elif head == 'svd':
            if self.inliers_ransac:
#                 self.head_ransacInliers = SVDHead_mod(emb_dims=self.emb_dims, cat_sampler='gumbel_softmax')
                self.head = SVDHead_mod(emb_dims=self.emb_dims, cat_sampler=cat_sampler)
            else:
#                 self.head_ransacInliers = SVDHead(emb_dims=self.emb_dims, cat_sampler='gumbel_softmax')
                self.head = SVDHead(emb_dims=self.emb_dims, cat_sampler=cat_sampler)
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

#         print(src.isnan().nonzero().shape,tgt.isnan().nonzero().shape)
#         print(src_embedding.isnan().nonzero().shape,tgt_embedding.isnan().nonzero().shape)
        
#         print(src_embedding,tgt_embedding)
        temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)
        
#         print(feature_disparity)

        return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity
    
    # Single Pass Alignment Module for PRNet
    def spam(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.predict_embedding(*input)
        
        print('SVDtemp',temperature.max(),temperature)
        temperature = temperature.clamp(min=1/self.temp_factor)
        
        self.mem_embedding['embedding']=(src,tgt,src_embedding, tgt_embedding, temperature, feature_disparity)
        
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature)
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity
    
    def get_OPT(self,*input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.mem_embedding['embedding']
        
#         print(src_embedding.shape,tgt_embedding.shape)
        l_plans=[]
        for it in range(src_embedding.shape[0]):
            P_ij=get_OPT_plan(src_embedding[it],src_embedding[it])
#             print('P_ij.shape',P_ij.shape)
            l_plans.append(P_ij)
        t_P_ij=torch.stack(l_plans)
        print('tensor OPT',t_P_ij.shape)
        return t_P_ij

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
                        (src,tgt,src_embedding, tgt_embedding, temperature, feature_disparity)=self.mem_embedding['embedding']
                        scores=self.head.mem['scores']
                        
                        # RANSAC_scores
                        scores_b=self.head.mem['scores_b']
#                         print('scores',scores.shape,scores_b.shape)
#                         print(src.shape,scores.shape,scores_b.shape)
                        print("SINKHORN")
                        print('scores',scores.nonzero().shape)
                        print('scores b',scores_b.nonzero().shape)
                        print('SCORES',scores[scores!=0])
                        print(scores_b[scores_b!=0])

                        scores_b=self.get_OPT(src_embedding,tgt_embedding)
                        print('embding shape',src_embedding.shape)
                        print('scores OPT',scores_b.nonzero().shape)
                        print(scores_b[scores_b!=0])
        
                        src_centered = src0 - src0.mean(dim=2, keepdim=True)
        
                        src_corr = torch.matmul(tgt0, scores.transpose(2, 1).contiguous())
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
