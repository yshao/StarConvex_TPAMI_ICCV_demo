import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn import DGCNN
from .pointnet import PointNet
from .. ops import transform_functions as transform
from .. utils import Transformer, SVDHead, Identity


# +
def sample_match(PP,n_samples=15):
#         PP=self.batch_pairwise_dist(gts.permute(0,2,1),preds.permute(0,2,1))
        N_SAMPLES=n_samples
        MM=[]
        
#         MM=torch.zeros(PP.shape[0],2)
        for it in range(PP.shape[0]):
            P = PP[it]
            loss = 0
            it=-1
    #         CM=torch.Tensor(1024,2)
#             N_SAMPLES=int(float(1024*(self.per_samples)))
            ct_sample=0
#             M = torch.zeros(0,2)
            M = []
            while (ct_sample < N_SAMPLES):
                ct_sample=ct_sample+1
#                 min_check=torch.min(P)
                min_check=torch.max(P)
                
         
                # find the index of min value in 2D tensor
#                 idx = (P==torch.min(P)).nonzero()
                idx = (P==torch.max(P)).nonzero()
                row = idx[0][0]
                col = idx[0][1]
                loss += P[row, col]
                # eliminate rows and cols where the min value located
    #             P = torch.cat((P[:row,:], P[row+1:,:]), axis = 0)
    #             P = torch.cat((P[:,:col], P[:,col+1:]), axis = 1)
    #             print(row.item(),col.item())
                P[row,:]=0
            #     P = torch.cat((P[:,:col], P[:,col+1:]), axis = 1)
                P[:,col]=0
                # Pair of correspondence
                M1=torch.cat((row.view(1),col.view(1))).unsqueeze(0)
    #             T1=torch.tensor((row.view(1),col.view(1)))
    #             CM[it]=T1
                it=it+1
    #             print(it)     
#                 M=torch.cat((M,M1))
                M.append(M1)
            M = torch.stack(M, dim=0)
            MM.append(M)
#             MM=torch.cat((MM,M.unsqueeze(0)))
        MM = torch.stack(MM, dim=0).squeeze(2)
#         print('MMshape',MM.shape)
#         print('MM',MM)
        return MM


# +
class ChamferLossSampler(nn.Module):

    def __init__(self,per_samples=0.1):
        super(ChamferLossSampler, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.per_samples=per_samples
        
        print('Sampler',self.per_samples)

    def forward(self, preds, gts):
        N_SAMPLES=int(float(1024*(self.per_samples)))
        N_SAMPLES=30
#         MM=torch.zeros(0,N_SAMPLES,2)
#         P = self.batch_pairwise_dist(gts, preds)
        PP=self.batch_pairwise_dist(gts.permute(0,2,1),preds.permute(0,2,1))
        
        MM=[]

#         MM=torch.zeros(PP.shape[0],2)
        for it in range(PP.shape[0]):
            P = PP[it]
            loss = 0
            it=-1
    #         CM=torch.Tensor(1024,2)
#             N_SAMPLES=int(float(1024*(self.per_samples)))
            ct_sample=0
#             M = torch.zeros(0,2)
            M = []
            while (ct_sample < N_SAMPLES):
                ct_sample=ct_sample+1
                min_check=torch.min(P)
         
                # find the index of min value in 2D tensor
                idx = (P==torch.min(P)).nonzero()
                row = idx[0][0]
                col = idx[0][1]
                loss += P[row, col]
                # eliminate rows and cols where the min value located
    #             P = torch.cat((P[:row,:], P[row+1:,:]), axis = 0)
    #             P = torch.cat((P[:,:col], P[:,col+1:]), axis = 1)
    #             print(row.item(),col.item())
                P[row,:]=10000
            #     P = torch.cat((P[:,:col], P[:,col+1:]), axis = 1)
                P[:,col]=10000
                # Pair of correspondence
                M1=torch.cat((row.view(1),col.view(1))).unsqueeze(0)
    #             T1=torch.tensor((row.view(1),col.view(1)))
    #             CM[it]=T1
                it=it+1
    #             print(it)            

#                 M=torch.cat((M,M1))
                M.append(M1)
            M = torch.stack(M, dim=0)
            MM.append(M)
#             MM=torch.cat((MM,M.unsqueeze(0)))
        MM = torch.stack(MM, dim=0)
        return loss, MM

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))

        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


# +
class ChamferLossMaxSampler(nn.Module):

    def __init__(self,per_samples=0.1):
        super(ChamferLossMaxSampler, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.per_samples=per_samples
        
        print('Sampler',self.per_samples)

    def forward(self, preds, gts):
        N_SAMPLES=int(float(1024*(self.per_samples)))
#         MM=torch.zeros(0,N_SAMPLES,2)
#         P = self.batch_pairwise_dist(gts, preds)
        PP=self.batch_pairwise_dist(gts.permute(0,2,1),preds.permute(0,2,1))
        
        MM=[]
        
#         MM=torch.zeros(PP.shape[0],2)
        for it in range(PP.shape[0]):
            P = PP[it]
            loss = 0
            it=-1
    #         CM=torch.Tensor(1024,2)
#             N_SAMPLES=int(float(1024*(self.per_samples)))
            ct_sample=0
#             M = torch.zeros(0,2)
            M = []
            while (ct_sample < N_SAMPLES):
                ct_sample=ct_sample+1
#                 min_check=torch.min(P)
                min_check=torch.max(P)
                
         
                # find the index of min value in 2D tensor
#                 idx = (P==torch.min(P)).nonzero()
                idx = (P==torch.max(P)).nonzero()
                row = idx[0][0]
                col = idx[0][1]
                loss += P[row, col]
                # eliminate rows and cols where the min value located
    #             P = torch.cat((P[:row,:], P[row+1:,:]), axis = 0)
    #             P = torch.cat((P[:,:col], P[:,col+1:]), axis = 1)
    #             print(row.item(),col.item())
                P[row,:]=10000
            #     P = torch.cat((P[:,:col], P[:,col+1:]), axis = 1)
                P[:,col]=10000
                # Pair of correspondence
                M1=torch.cat((row.view(1),col.view(1))).unsqueeze(0)
    #             T1=torch.tensor((row.view(1),col.view(1)))
    #             CM[it]=T1
                it=it+1
    #             print(it)     
#                 M=torch.cat((M,M1))
                M.append(M1)
            M = torch.stack(M, dim=0)
            MM.append(M)
#             MM=torch.cat((MM,M.unsqueeze(0)))
        MM = torch.stack(MM, dim=0)
        return loss, MM

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))

        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P    

# +
import torch
import torch.nn as nn
import math

class SVDHead_matching2(nn.Module):
    def __init__(self, emb_dims, input_shape="bnc"):
        super(SVDHead_matching2, self).__init__()
        self.emb_dims = emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.input_shape = input_shape
        
    def get_correspondence(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)
        
        device=src.device
        
        if self.input_shape == "bnc":
            src = src.permute(0, 2, 1)
            tgt = tgt.permute(0, 2, 1)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)
        
        samples=sample_match(scores.detach().clone().to(device))

        ### one to one matching
        topk_src=[]
        topk_tgt=[]
        
        topk_src_idx=[]
        topk_tgt_idx=[]
#         topk_src_embedding=[]
#         topk_tgt_embedding=[]
#         topk_scores=[]
        for it in range(samples.shape[0]):
            mm_src_idx_i=samples[it,:,0]
            mm_tgt_idx_i=samples[it,:,1]

            
            it0_i=mm_src_idx_i
            it1_i=mm_tgt_idx_i
            # Select indices from the 1-to-1 matching matrices to get them from the original src, tgt <- N x 3d points
#             it0_i=topk_H_src_idx[0].gather(dim=0,index=mm_src_idx_i)
#             it1_i=topk_H_tgt_idx[0].gather(dim=0,index=mm_tgt_idx_i)


            # Select topk 3d points based on indices
            t_src_mm_topk_i=src.index_select(dim=2,index=it0_i)[it]
            t_tgt_mm_topk_i=tgt.index_select(dim=2,index=it1_i)[it]

            # Select topk network embedding based on indices
#             t_src_embed_mm_topk_i=src_embedding.index_select(dim=2,index=it0_i)
#             t_tgt_embed_mm_topk_i=tgt_embedding.index_select(dim=2,index=it1_i)        
            
#             t_tgt_scores=scores.index_select(dim=2,index=it1_i)        
            print(t_tgt_mm_topk_i.shape)
            print(it0_i.shape)
            
            topk_src_idx.append(it0_i)
            topk_tgt_idx.append(it1_i)
    
            topk_src.append(t_src_mm_topk_i)
            topk_tgt.append(t_tgt_mm_topk_i)
#             topk_src_embedding.append(t_src_embed_mm_topk_i)
#             topk_tgt_embedding.append(t_tgt_embed_mm_topk_i)
            
#             topk_scores.append(t_tgt_scores)
            
        topk_src=torch.stack(topk_src,dim=0).squeeze(1)
        topk_tgt=torch.stack(topk_tgt,dim=0).squeeze(1)
        
        topk_src_idx=torch.stack(topk_src_idx,dim=0).squeeze(1)
        topk_tgt_idx=torch.stack(topk_tgt_idx,dim=0).squeeze(1)
        
        return topk_src, topk_tgt, topk_src_idx, topk_tgt_idx
        

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)
        
        device=src.device
        
        if self.input_shape == "bnc":
            src = src.permute(0, 2, 1)
            tgt = tgt.permute(0, 2, 1)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)
        
#         print('scores',scores.shape,scores)
#         print('inputs',src_embedding.shape,src.shape)
#         print(scores.mean())
#         samples=self.sampler(scores)
        samples=sample_match(scores.detach().clone().to(device))
#         print('Mshape',samples.shape)
#         print(samples)
        
        ### one to one matching
        topk_src=[]
        topk_tgt=[]
#         topk_src_embedding=[]
#         topk_tgt_embedding=[]
#         topk_scores=[]
        for it in range(samples.shape[0]):
            mm_src_idx_i=samples[it,:,0]
            mm_tgt_idx_i=samples[it,:,1]

            
            it0_i=mm_src_idx_i
            it1_i=mm_tgt_idx_i
            # Select indices from the 1-to-1 matching matrices to get them from the original src, tgt <- N x 3d points
#             it0_i=topk_H_src_idx[0].gather(dim=0,index=mm_src_idx_i)
#             it1_i=topk_H_tgt_idx[0].gather(dim=0,index=mm_tgt_idx_i)


            # Select topk 3d points based on indices
            t_src_mm_topk_i=src.index_select(dim=2,index=it0_i)[it]
            t_tgt_mm_topk_i=tgt.index_select(dim=2,index=it1_i)[it]

            # Select topk network embedding based on indices
#             t_src_embed_mm_topk_i=src_embedding.index_select(dim=2,index=it0_i)
#             t_tgt_embed_mm_topk_i=tgt_embedding.index_select(dim=2,index=it1_i)        
            
#             t_tgt_scores=scores.index_select(dim=2,index=it1_i)        
            
            topk_src.append(t_src_mm_topk_i)
            topk_tgt.append(t_tgt_mm_topk_i)
#             topk_src_embedding.append(t_src_embed_mm_topk_i)
#             topk_tgt_embedding.append(t_tgt_embed_mm_topk_i)
            
#             topk_scores.append(t_tgt_scores)
            
        topk_src=torch.stack(topk_src,dim=0).squeeze(1)
        topk_tgt=torch.stack(topk_tgt,dim=0).squeeze(1)
#         topk_src_embedding=torch.stack(topk_src_embedding,dim=0).squeeze(1)
#         topk_tgt_embedding=torch.stack(topk_tgt_embedding,dim=0).squeeze(1)
        
#         topk_scores=torch.stack(topk_scores,dim=0)

#         topk_d_k = topk_src_embedding.size(1)
#         topk_scores = torch.matmul(topk_src_embedding.transpose(2, 1).contiguous(), topk_tgt_embedding) / math.sqrt(topk_d_k)
#         topk_scores = torch.softmax(topk_scores, dim=2)
        
#         print('ouputs',topk_src.shape,topk_src_embedding.shape,topk_scores.shape)

#         return scores, samples, 
        U, S, V = [], [], []
        R = []
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
        for i in range(src.size(0)):        
#         src_corr = torch.matmul(topk_tgt, topk_scores.transpose(2, 1).contiguous())
            tgt_centered = topk_tgt[i] - topk_tgt[i].mean(dim=1, keepdim=True)

#         src_centered = src - src.mean(dim=2, keepdim=True)
            src_centered = topk_src[i] - topk_src[i].mean(dim=1, keepdim=True)

#         src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
#         src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
    
#         print('correspondence',src_corr_centered.shape,src_centered.shape)

#         H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
            H = torch.matmul(src_centered, tgt_centered.transpose(1, 0).contiguous())

#             u, s, v = torch.svd(H[i])
            u, s, v = torch.svd(H)        
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H)
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
#         t = torch.matmul(-R, topk_src.mean(dim=2, keepdim=True)) + topk_tgt.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)


# +
### Hand-coded, to be re-code with geometric descriptor library

class HarrisFeatureSelectionModule(nn.Module):
    def __init__(self, args):
        super(HarrisSelectionModule, self).__init__()
    # 1-to-1 Matching

    def forward(self,src,tgt,src_embedding,tgt_embedding):
        # Calculate harris matching features
        # and Select the topk indices based on the geometric features with threshold set for both src, tgt <- N x 3d points
        T_est, data1,data2, topk_H_src_idx, topk_H_tgt_idx, H_min_src, H_min_tgt=Harris_matching(src,tgt,th1=0.8,th2=0.8)
                
        t_src_Hmin_topk_i=H_min_src.gather(dim=0,index=topk_H_src_idx[0]).unsqueeze(0).unsqueeze(2)
        t_tgt_Hmin_topk_i=H_min_tgt.gather(dim=0,index=topk_H_tgt_idx[0]).unsqueeze(0).unsqueeze(2)

        # Correspondence matrix
        PP_i=batch_pairwise_dist(t_src_Hmin_topk_i,t_tgt_Hmin_topk_i)
        
        # 1-to-1 matching
        MM_i=match_chamfer(PP_i,n_samples=15).squeeze(2)

        mm_tgt_idx_i=MM_i[0,:,0]
        mm_src_idx_i=MM_i[0,:,1]

        # Select indices from the 1-to-1 matching matrices to get them from the original src, tgt <- N x 3d points
        it0_i=topk_H_src_idx[0].gather(dim=0,index=mm_src_idx_i)
        it1_i=topk_H_tgt_idx[0].gather(dim=0,index=mm_tgt_idx_i)


        # Select topk 3d points based on indices
        t_src_mm_topk_i=src.index_select(dim=2,index=it0_i)
        t_tgt_mm_topk_i=tgt.index_select(dim=2,index=it1_i)

        # Select topk network embedding based on indices
        t_src_embed_mm_topk_i=src_embedding.index_select(dim=2,index=it0_i)
        t_tgt_embed_mm_topk_i=tgt_embedding.index_select(dim=2,index=it1_i)
        
        return t_src_mm_topk_i, t_tgt_mm_topk_i, t_src_embed_mm_topk_i, t_tgt_embed_mm_topk_i, it0_i, it1_i

# +
import sys
sys.path.append('/workspace/multistage_v2')

from algo import calc_svd_with_H, calc_svd_from_matches, Harris_matching, match_chamfer, batch_pairwise_dist
# Key points guid ethe embedding

class FeatureSelectionModule(nn.Module):
    def __init__(self, args):
        super(FeatureSelectionModule, self).__init__()
    # 1-to-1 Matching

    def forward(self,src,tgt,src_embedding,tgt_embedding):
        # Calculate harris matching features
        T_est, data1,data2, topk_H_src_idx, topk_H_tgt_idx, H_min_src, H_min_tgt=Harris_matching(src,tgt,th1=0.8,th2=0.8)
        
        print('topk H',topk_H_src_idx[0].shape)        
        t_src_Hmin_topk_i=H_min_src.gather(dim=0,index=topk_H_src_idx[0]).unsqueeze(0).unsqueeze(2)
        t_tgt_Hmin_topk_i=H_min_tgt.gather(dim=0,index=topk_H_tgt_idx[0]).unsqueeze(0).unsqueeze(2)

#         # correspondence and matching
        PP_i=batch_pairwise_dist(t_src_Hmin_topk_i,t_tgt_Hmin_topk_i)
        MM_i=match_chamfer(PP_i,n_samples=15).squeeze(2)


        mm_tgt_idx_i=MM_i[0,:,0]
        mm_src_idx_i=MM_i[0,:,1]

        it0_i=topk_H_src_idx[0].gather(dim=0,index=mm_src_idx_i)
        it1_i=topk_H_tgt_idx[0].gather(dim=0,index=mm_tgt_idx_i)



        t_src_mm_topk_i=src.index_select(dim=2,index=it0_i)
        t_tgt_mm_topk_i=tgt.index_select(dim=2,index=it1_i)

        t_src_embed_mm_topk_i=src_embedding.index_select(dim=2,index=it0_i)
        t_tgt_embed_mm_topk_i=tgt_embedding.index_select(dim=2,index=it1_i)
        
        return t_src_mm_topk_i, t_tgt_mm_topk_i, t_src_embed_mm_topk_i, t_tgt_embed_mm_topk_i, it0_i, it1_i


# -

class DCP_matching_one2one(nn.Module):
	def __init__(self, feature_model=DGCNN(), cycle=False, pointer_='transformer', head='svd'):
		super(DCP_matching_one2one, self).__init__()
		self.cycle = cycle
		self.emb_nn = feature_model

		if pointer_ == 'identity':
			self.pointer = Identity()
		elif pointer_ == 'transformer':
			self.pointer = Transformer(self.emb_nn.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
		else:
			raise Exception("Not implemented")

		if head == 'mlp':
			self.head = MLPHead(self.emb_nn.emb_dims)
		elif head == 'svd':
			self.head = SVDHead(self.emb_nn.emb_dims)
		else:
			raise Exception('Not implemented')
		self.head = SVDHead_matching2(self.emb_nn.emb_dims)

	def get_correspondence(self, template, source):
        # Create Embedding with DGCNN
		source_features = self.emb_nn(source)
		template_features = self.emb_nn(template)
        # Self-supervised pointers
		source_features_p, template_features_p = self.pointer(source_features, template_features)
        
		source_features = source_features + source_features_p
		template_features = template_features + template_features_p

		topk_src, topk_tgt, topk_src_idx, topk_tgt_idx = self.head.get_correspondence(source_features, template_features, source, template)        
        
		return  topk_src, topk_tgt, topk_src_idx, topk_tgt_idx
        
	def forward(self, template, source):
        # Create Embedding with DGCNN
		source_features = self.emb_nn(source)
		template_features = self.emb_nn(template)
        # Self-supervised pointers
		source_features_p, template_features_p = self.pointer(source_features, template_features)

		source_features = source_features + source_features_p
		template_features = template_features + template_features_p
# 		print(torch.nonzero(torch.isnan(template_features.view(-1))))
        # SVD
		rotation_ab, translation_ab = self.head(source_features, template_features, source, template)
		if self.cycle:
			rotation_ba, translation_ba = self.head(template_features, source_features, template, source)
		else:
			rotation_ba = rotation_ab.transpose(2, 1).contiguous()
			translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
# 		return self.head(template_features, source_features, template, source)
		transformed_source = transform.transform_point_cloud(source, rotation_ab, translation_ab)

		result = {'est_R': rotation_ab,
				  'est_t': translation_ab,
				  'est_R_': rotation_ba,
				  'est_t_': translation_ba,
				  'est_T': transform.convert2transformation(rotation_ab, translation_ab),
				  'r': template_features - source_features,
				  'transformed_source': transformed_source}
		return result


# +
class DCP_mod(nn.Module):
	def __init__(self, feature_model=DGCNN(), cycle=False, pointer_='transformer', head='svd'):
		super(DCP_mod, self).__init__()
		self.cycle = cycle
		self.emb_nn = feature_model

# 		if pointer_ == 'identity':
# 			self.pointer = Identity()
# 		elif pointer_ == 'transformer':
# 			self.pointer = Transformer(self.emb_nn.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
# 		else:
# 			raise Exception("Not implemented")
            
		self.pointer = Transformer(self.emb_nn.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)            

# 		if head == 'mlp':
# 			self.head = MLPHead(self.emb_nn.emb_dims)
# 		elif head == 'svd':
# 			self.head = SVDHead(self.emb_nn.emb_dims)
# 		else:
# 			raise Exception('Not implemented')

		self.head=SVDHead(self.emb_nn.emb_dims)
    
		self.keypointnet=FeatureSelectionModule()    

	def forward(self, template, source):
		source_features = self.emb_nn(source)
		template_features = self.emb_nn(template)

		source_features_p, template_features_p = self.pointer(source_features, template_features)

		source_features = source_features + source_features_p
		template_features = template_features + template_features_p
        
        # Once feature are present
        # NEW: make stub
# 		source_features=template_features.clone()
		topk_src, topk_tgt, topk_src_embedding, topk_tgt_embedding, topk_src_idx, topk_tgt_idx=compute_topk_harris(self.keypointnet,template,source, template_features, source_features)
        
		topk_source=source.select_index(dim=1,idnex=topk_src_idx)
		topk_source_features=source_features.select_index(dim=1,idnex=topk_src_idx)        
		topk_template=template.select_index(dim=1,idnex=topk_tgt_idx)
		topk_template_features=template_features.select_index(dim=1,index=topk_tgt_idx)        
        
        # NEW rotations
		rotation_ab, translation_ab = self.head(topk_source_features, topk_template_features, topk_source, topk_template)
		if self.cycle:
			rotation_ba, translation_ba = self.head(topk_template_features, topk_source_features, topk_template, topk_source)
		else:
			rotation_ba = rotation_ab.transpose(2, 1).contiguous()
			translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
        
# 		rotation_ab, translation_ab = self.head(source_features, template_features, source, template)
# 		if self.cycle:
# 			rotation_ba, translation_ba = self.head(template_features, source_features, template, source)
# 		else:
# 			rotation_ba = rotation_ab.transpose(2, 1).contiguous()
# 			translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

		transformed_source = transform.transform_point_cloud(source, rotation_ab, translation_ab)

		result = {'est_R': rotation_ab,
				  'est_t': translation_ab,
				  'est_R_': rotation_ba,
				  'est_t_': translation_ba,
				  'est_T': transform.convert2transformation(rotation_ab, translation_ab),
				  'r': template_features - source_features,
				  'transformed_source': transformed_source}
		return result


# -

class DCP(nn.Module):
	def __init__(self, feature_model=DGCNN(), cycle=False, pointer_='transformer', head='svd'):
		super(DCP, self).__init__()
		self.cycle = cycle
		self.emb_nn = feature_model

		if pointer_ == 'identity':
			self.pointer = Identity()
		elif pointer_ == 'transformer':
			self.pointer = Transformer(self.emb_nn.emb_dims, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)
		else:
			raise Exception("Not implemented")

		if head == 'mlp':
			self.head = MLPHead(self.emb_nn.emb_dims)
		elif head == 'svd':
			self.head = SVDHead(self.emb_nn.emb_dims)
		else:
			raise Exception('Not implemented')

	def forward(self, template, source):
        # Create Embedding with DGCNN
		source_features = self.emb_nn(source)
		template_features = self.emb_nn(template)
        # Self-supervised pointers
		source_features_p, template_features_p = self.pointer(source_features, template_features)

		source_features = source_features + source_features_p
		template_features = template_features + template_features_p
        
		print('SVD')
        # SVD
		rotation_ab, translation_ab = self.head(source_features, template_features, source, template)
		if self.cycle:
			rotation_ba, translation_ba = self.head(template_features, source_features, template, source)
		else:
			rotation_ba = rotation_ab.transpose(2, 1).contiguous()
			translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

		transformed_source = transform.transform_point_cloud(source, rotation_ab, translation_ab)

		result = {'est_R': rotation_ab,
				  'est_t': translation_ab,
				  'est_R_': rotation_ba,
				  'est_t_': translation_ba,
				  'est_T': transform.convert2transformation(rotation_ab, translation_ab),
				  'r': template_features - source_features,
				  'transformed_source': transformed_source}
		return result


class MLPHead(nn.Module):
    def __init__(self, emb_dims):
        super(MLPHead, self).__init__()
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	pn = PointNet()

	# Not Tested Yet.
	net = DCP(pn)
	result = net(template, source)
	import ipdb; ipdb.set_trace()
