import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNet
from .pooling import Pooling
from .. ops.transform_functions import PCRNetTransform as transform

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

        # it0_i=topk_H_src_idx[0].gather(index=mm_src_idx_i)
        # it1_i=topk_H_tgt_idx[0].gather(index=mm_tgt_idx_i)

        # topk_H_tgt_idx[0].shape,it0,it1,it0_i,it1_i
        it0_i=topk_H_src_idx[0].gather(dim=0,index=mm_src_idx_i)
        it1_i=topk_H_tgt_idx[0].gather(dim=0,index=mm_tgt_idx_i)



        t_src_mm_topk_i=src.index_select(dim=2,index=it0_i)
        t_tgt_mm_topk_i=tgt.index_select(dim=2,index=it1_i)

        t_src_embed_mm_topk_i=src_embedding.index_select(dim=2,index=it0_i)
        t_tgt_embed_mm_topk_i=tgt_embedding.index_select(dim=2,index=it1_i)
        
        return t_src_mm_topk_i, t_tgt_mm_topk_i, t_src_embed_mm_topk_i, t_tgt_embed_mm_topk_i, it0_i, it1_i


# +
class iPCRNet_mod(nn.Module):
	def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling(pooling)

		self.linear = [nn.Linear(self.feature_model.emb_dims * 2, 1024), nn.ReLU(),
				   	   nn.Linear(1024, 1024), nn.ReLU(),
				   	   nn.Linear(1024, 512), nn.ReLU(),
				   	   nn.Linear(512, 512), nn.ReLU(),
				   	   nn.Linear(512, 256), nn.ReLU()]

		if droput>0.0:
			self.linear.append(nn.Dropout(droput))
		self.linear.append(nn.Linear(256,7))

		self.linear = nn.Sequential(*self.linear)
        
		self.keypointnet=FeatureSelectionModule()         

	# Single Pass Alignment Module (SPAM)
	def spam(self, template_features, source, est_R, est_t):
		batch_size = source.size(0)

		self.source_features = self.pooling(self.feature_model(source))
		y = torch.cat([template_features, self.source_features], dim=1)
		pose_7d = self.linear(y)
		pose_7d = transform.create_pose_7d(pose_7d)

		# Find current rotation and translation.
		identity = torch.eye(3).to(source).view(1,3,3).expand(batch_size, 3, 3).contiguous()
		est_R_temp = transform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
		est_t_temp = transform.get_translation(pose_7d).view(-1, 1, 3)

		# update translation matrix.
		est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
		# update rotation matrix.
		est_R = torch.bmm(est_R_temp, est_R)
		
		source = transform.quaternion_transform(source, pose_7d)      # Ps' = est_R*Ps + est_t
		return est_R, est_t, source

	def forward(self, template, source, max_iteration=8):
		est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()         # (Bx3x3)
		est_t = torch.zeros(1,3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()     # (Bx1x3)
		template_features = self.pooling(self.feature_model(template))

        # Once feature are present
        # NEW: make stub
# 		source_features=template_features.clone()
		topk_src, topk_tgt, topk_src_embedding, topk_tgt_embedding, topk_src_idx, topk_tgt_idx=compute_topk_harris(self.keypointnet,template,source, template_features, source_features)
        
		topk_source=source.select_index(dim=1,idnex=topk_src_idx)
		topk_source_features=source_features.select_index(dim=1,idnex=topk_src_idx)        
		topk_template=template.select_index(dim=1,idnex=topk_tgt_idx)
		topk_template_features=template_features.select_index(dim=1,index=topk_tgt_idx)         
		# NEW
		if max_iteration == 1:
			est_R, est_t, source = self.spam(topk_template_features, topk_source, est_R, est_t)
		else:
			for i in range(max_iteration):
				est_R, est_t, source = self.spam(topk_template_features, topk_source, est_R, est_t)

# 		if max_iteration == 1:
# 			est_R, est_t, source = self.spam(template_features, source, est_R, est_t)
# 		else:
# 			for i in range(max_iteration):
# 				est_R, est_t, source = self.spam(template_features, source, est_R, est_t)

		result = {'est_R': est_R,				# source -> template
				  'est_t': est_t,				# source -> template
				  'est_T': transform.convert2transformation(est_R, est_t),			# source -> template
				  'r': template_features - self.source_features,
				  'transformed_source': source}
		return result


# -

class iPCRNet(nn.Module):
	def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling(pooling)

		self.linear = [nn.Linear(self.feature_model.emb_dims * 2, 1024), nn.ReLU(),
				   	   nn.Linear(1024, 1024), nn.ReLU(),
				   	   nn.Linear(1024, 512), nn.ReLU(),
				   	   nn.Linear(512, 512), nn.ReLU(),
				   	   nn.Linear(512, 256), nn.ReLU()]

		if droput>0.0:
			self.linear.append(nn.Dropout(droput))
		self.linear.append(nn.Linear(256,7))

		self.linear = nn.Sequential(*self.linear)

	# Single Pass Alignment Module (SPAM)
	def spam(self, template_features, source, est_R, est_t):
		batch_size = source.size(0)

		self.source_features = self.pooling(self.feature_model(source))
		y = torch.cat([template_features, self.source_features], dim=1)
		pose_7d = self.linear(y)
		pose_7d = transform.create_pose_7d(pose_7d)

		# Find current rotation and translation.
		identity = torch.eye(3).to(source).view(1,3,3).expand(batch_size, 3, 3).contiguous()
		est_R_temp = transform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
		est_t_temp = transform.get_translation(pose_7d).view(-1, 1, 3)

		# update translation matrix.
		est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
		# update rotation matrix.
		est_R = torch.bmm(est_R_temp, est_R)
		
		source = transform.quaternion_transform(source, pose_7d)      # Ps' = est_R*Ps + est_t
		return est_R, est_t, source

	def forward(self, template, source, max_iteration=8):
		est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()         # (Bx3x3)
		est_t = torch.zeros(1,3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()     # (Bx1x3)
		template_features = self.pooling(self.feature_model(template))

		if max_iteration == 1:
			est_R, est_t, source = self.spam(template_features, source, est_R, est_t)
		else:
			for i in range(max_iteration):
				est_R, est_t, source = self.spam(template_features, source, est_R, est_t)

		result = {'est_R': est_R,				# source -> template
				  'est_t': est_t,				# source -> template
				  'est_T': transform.convert2transformation(est_R, est_t),			# source -> template
				  'r': template_features - self.source_features,
				  'transformed_source': source}
		return result


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	pn = PointNet()
	
	net = iPCRNet(pn)
	result = net(template, source)
	import ipdb; ipdb.set_trace()
