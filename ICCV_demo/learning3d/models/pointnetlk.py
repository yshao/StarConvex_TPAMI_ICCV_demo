import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNet
from .pooling import Pooling
from .. ops import data_utils
from .. ops import se3, so3, invmat

# +
import sys
sys.path.append('/workspace/multistage_v2')

from algo import calc_svd_with_H, calc_svd_from_matches, Harris_matching, match_chamfer, batch_pairwise_dist
# Key points guid ethe embedding

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


# -

# Becauase modules are assembled different
def compute_topk_harris(module,src,tgt,src_embedding,tgt_embedding):
        ### Batch version
        topk_src=[]
        topk_tgt=[]
        topk_src_embedding=[]
        topk_tgt_embedding=[]
        topk_src_idx=[]
        topk_tgt_idx=[] 
        
        topk_temperature=[]
        topk_feature_disparity=[]
        
        for bit in range(src.shape[0]):
            src_bit=src[bit].unsqueeze(0)
            tgt_bit=tgt[bit].unsqueeze(0)
            tgt_embedding_bit=tgt_embedding[bit].unsqueeze(0)
            src_embedding_bit=src_embedding[bit].unsqueeze(0)
            
            print(bit,src_bit.shape,src_embedding_bit.shape)
            src_bit, tgt_bit, src_embedding_bit, tgt_embedding_bit, src_idx_bit, tgt_idx_bit = module(src_bit, tgt_bit, src_embedding_bit, tgt_embedding_bit)
    
            topk_src.append(src_bit)
            topk_tgt.append(tgt_bit)
            topk_src_embedding.append(src_embedding_bit)
            topk_tgt_embedding.append(tgt_embedding_bit)
            
            topk_src_idx.append(src_idx_bit)
            topk_tgt_idx.append(tgt_idx_bit)            
#             topk_temperature.append(temperature_bit)
#             topk_feature_disparity.append(feature_disparity_bit)
    
        topk_src=torch.stack(topk_src,dim=0).squeeze(1)
        topk_tgt=torch.stack(topk_tgt,dim=0).squeeze(1)
        topk_src_embedding=torch.stack(topk_src_embedding,dim=0).squeeze(1)
        topk_tgt_embedding=torch.stack(topk_tgt_embedding,dim=0).squeeze(1)

        topk_src_idx=torch.stack(topk_src_idx,dim=0).squeeze(1)
        topk_tgt_idx=torch.stack(topk_tgt_idx,dim=0).squeeze(1)
        
        print('TOPK src',topk_src.shape)
        return topk_src, topk_tgt, topk_src_embedding, topk_tgt_embedding, topk_src_idx, topk_tgt_idx


# +
class PointNetLK_mod(nn.Module):
	def __init__(self, feature_model=PointNet(), delta=1.0e-2, learn_delta=False, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True, pooling='max'):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling(pooling)
		self.inverse = invmat.InvMatrix.apply
		self.exp = se3.Exp # [B, 6] -> [B, 4, 4]
		self.transform = se3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

		w1, w2, w3, v1, v2, v3 = delta, delta, delta, delta, delta, delta
		twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
		self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

		# results
		self.last_err = None
		self.g_series = None # for debug purpose
		self.prev_r = None
		self.g = None # estimation result
		self.itr = 0
		self.xtol = xtol
		self.p0_zero_mean = p0_zero_mean
		self.p1_zero_mean = p1_zero_mean
        
		self.keypointnet=FeatureSelectionModule()       

	def forward(self, template, source, maxiter=10):
		template, source, template_mean, source_mean = data_utils.mean_shift(template, source, 
																			 self.p0_zero_mean, self.p1_zero_mean)

		result = self.iclk(template, source, maxiter)
		result = data_utils.postprocess_data(result, template, source, template_mean, source_mean, 
											 self.p0_zero_mean, self.p1_zero_mean)
		return result

	def iclk(self, template, source, maxiter):
		batch_size = template.size(0)

		est_T0 = torch.eye(4).to(template).view(1, 4, 4).expand(template.size(0), 4, 4).contiguous()
		est_T = est_T0
		self.est_T_series = torch.zeros(maxiter+1, *est_T0.size(), dtype=est_T0.dtype)
		self.est_T_series[0] = est_T0.clone()

		training = self.handle_batchNorm(template, source)        

		# re-calc. with current modules
		template_features = self.pooling(self.feature_model(template)) # [B, N, 3] -> [B, K]

        # Once feature are present
        # NEW: make stub
		source_features=template_features.clone()
		topk_src, topk_tgt, topk_src_embedding, topk_tgt_embedding, topk_src_idx, topk_tgt_idx=compute_topk_harris(self.keypointnet,template,source, template_features, source_features)
        
		topk_source=source.select_index(dim=1,idnex=topk_src_idx)
		topk_template=template.select_index(dim=1,idnex=topk_tgt_idx)
		topk_template_features=template_features.select_index(dim=1,idnex=topk_tgt_idx)
        
        # NEW: What's changed
		# approx. J by finite difference
		dt = self.dt.to(topk_template).expand(batch_size, 6)
		J = self.approx_Jic(topk_template, topk_template_features, dt)

		self.last_err = None
		pinv = self.compute_inverse_jacobian(J, topk_template_features, topk_source)        
        
# 		# approx. J by finite difference
# 		dt = self.dt.to(template).expand(batch_size, 6)
# 		J = self.approx_Jic(template, template_features, dt)

# 		self.last_err = None
# 		pinv = self.compute_inverse_jacobian(J, template_features, source)                        
        
		if pinv == {}:
			result = {'est_R': est_T[:,0:3,0:3],
					  'est_t': est_T[:,0:3,3],
					  'est_T': est_T,
					  'r': None,
					  'transformed_source': self.transform(est_T.unsqueeze(1), source),
					  'itr': 1,
					  'est_T_series': self.est_T_series}
			return result

		itr = 0
		r = None
		for itr in range(maxiter):
			self.prev_r = r
			transformed_source = self.transform(est_T.unsqueeze(1), source) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
			source_features = self.pooling(self.feature_model(transformed_source)) # [B, N, 3] -> [B, K]
			r = source_features - template_features

			pose = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

			check = pose.norm(p=2, dim=1, keepdim=True).max()
			if float(check) < self.xtol:
				if itr == 0:
					self.last_err = 0 # no update.
				break

			est_T = self.update(est_T, pose)
			self.est_T_series[itr+1] = est_T.clone()

		rep = len(range(itr, maxiter))
		self.est_T_series[(itr+1):] = est_T.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

		self.feature_model.train(training)
		self.est_T = est_T

		result = {'est_R': est_T[:,0:3,0:3],
				  'est_t': est_T[:,0:3,3],
				  'est_T': est_T,
				  'r': r,
				  'transformed_source': self.transform(est_T.unsqueeze(1), source),
				  'itr': itr+1,
				  'est_T_series': self.est_T_series}
		
		return result

	def update(self, g, dx):
		# [B, 4, 4] x [B, 6] -> [B, 4, 4]
		dg = self.exp(dx)
		return dg.matmul(g)

	def approx_Jic(self, template, template_features, dt):
		# p0: [B, N, 3], Variable
		# f0: [B, K], corresponding feature vector
		# dt: [B, 6], Variable
		# Jk = (feature_model(p(-delta[k], p0)) - f0) / delta[k]

		batch_size = template.size(0)
		num_points = template.size(1)

		# compute transforms
		transf = torch.zeros(batch_size, 6, 4, 4).to(template)
		for b in range(template.size(0)):
			d = torch.diag(dt[b, :]) # [6, 6]
			D = self.exp(-d) # [6, 4, 4]
			transf[b, :, :, :] = D[:, :, :]
		transf = transf.unsqueeze(2).contiguous()  #   [B, 6, 1, 4, 4]
		p = self.transform(transf, template.unsqueeze(1)) # x [B, 1, N, 3] -> [B, 6, N, 3]

		#f0 = self.feature_model(p0).unsqueeze(-1) # [B, K, 1]
		template_features = template_features.unsqueeze(-1) # [B, K, 1]
		f = self.pooling(self.feature_model(p.view(-1, num_points, 3))).view(batch_size, 6, -1).transpose(1, 2) # [B, K, 6]

		df = template_features - f # [B, K, 6]
		J = df / dt.unsqueeze(1)

		return J

	def compute_inverse_jacobian(self, J, template_features, source):
		# compute pinv(J) to solve J*x = -r
		try:
			Jt = J.transpose(1, 2) # [B, 6, K]
			H = Jt.bmm(J) # [B, 6, 6]
			B = self.inverse(H)
			pinv = B.bmm(Jt) # [B, 6, K]
			return pinv
		except RuntimeError as err:
			# singular...?
			self.last_err = err
			g = torch.eye(4).to(source).view(1, 4, 4).expand(source.size(0), 4, 4).contiguous()
			#print(err)
			# Perhaps we can use MP-inverse, but,...
			# probably, self.dt is way too small...
			source_features = self.pooling(self.feature_model(source)) # [B, N, 3] -> [B, K]
			r = source_features - template_features
			self.feature_model.train(self.feature_model.training)
			return {}

	def handle_batchNorm(self, template, source):
		training = self.feature_model.training
		if training:
			# first, update BatchNorm modules
			template_features, source_features = self.pooling(self.feature_model(template)), self.pooling(self.feature_model(source))
		self.feature_model.eval()	# and fix them.
		return training


# +
class PointNetLK(nn.Module):
	def __init__(self, feature_model=PointNet(), delta=1.0e-2, learn_delta=False, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True, pooling='max'):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling(pooling)
		self.inverse = invmat.InvMatrix.apply
		self.exp = se3.Exp # [B, 6] -> [B, 4, 4]
		self.transform = se3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

		w1, w2, w3, v1, v2, v3 = delta, delta, delta, delta, delta, delta
		twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
		self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

		# results
		self.last_err = None
		self.g_series = None # for debug purpose
		self.prev_r = None
		self.g = None # estimation result
		self.itr = 0
		self.xtol = xtol
		self.p0_zero_mean = p0_zero_mean
		self.p1_zero_mean = p1_zero_mean

	def forward(self, template, source, maxiter=10):
		template, source, template_mean, source_mean = data_utils.mean_shift(template, source, 
																			 self.p0_zero_mean, self.p1_zero_mean)

		result = self.iclk(template, source, maxiter)
		result = data_utils.postprocess_data(result, template, source, template_mean, source_mean, 
											 self.p0_zero_mean, self.p1_zero_mean)
		return result

	def iclk(self, template, source, maxiter):
		batch_size = template.size(0)

		est_T0 = torch.eye(4).to(template).view(1, 4, 4).expand(template.size(0), 4, 4).contiguous()
		est_T = est_T0
		self.est_T_series = torch.zeros(maxiter+1, *est_T0.size(), dtype=est_T0.dtype)
		self.est_T_series[0] = est_T0.clone()

		training = self.handle_batchNorm(template, source)

		# re-calc. with current modules
		template_features = self.pooling(self.feature_model(template)) # [B, N, 3] -> [B, K]

		# approx. J by finite difference
		dt = self.dt.to(template).expand(batch_size, 6)
		J = self.approx_Jic(template, template_features, dt)

		self.last_err = None
		pinv = self.compute_inverse_jacobian(J, template_features, source)
		if pinv == {}:
			result = {'est_R': est_T[:,0:3,0:3],
					  'est_t': est_T[:,0:3,3],
					  'est_T': est_T,
					  'r': None,
					  'transformed_source': self.transform(est_T.unsqueeze(1), source),
					  'itr': 1,
					  'est_T_series': self.est_T_series}
			return result

		itr = 0
		r = None
		for itr in range(maxiter):
			self.prev_r = r
			transformed_source = self.transform(est_T.unsqueeze(1), source) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
			source_features = self.pooling(self.feature_model(transformed_source)) # [B, N, 3] -> [B, K]
			r = source_features - template_features

			pose = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

			check = pose.norm(p=2, dim=1, keepdim=True).max()
			if float(check) < self.xtol:
				if itr == 0:
					self.last_err = 0 # no update.
				break

			est_T = self.update(est_T, pose)
			self.est_T_series[itr+1] = est_T.clone()

		rep = len(range(itr, maxiter))
		self.est_T_series[(itr+1):] = est_T.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

		self.feature_model.train(training)
		self.est_T = est_T

		result = {'est_R': est_T[:,0:3,0:3],
				  'est_t': est_T[:,0:3,3],
				  'est_T': est_T,
				  'r': r,
				  'transformed_source': self.transform(est_T.unsqueeze(1), source),
				  'itr': itr+1,
				  'est_T_series': self.est_T_series}
		
		return result

	def update(self, g, dx):
		# [B, 4, 4] x [B, 6] -> [B, 4, 4]
		dg = self.exp(dx)
		return dg.matmul(g)

	def approx_Jic(self, template, template_features, dt):
		# p0: [B, N, 3], Variable
		# f0: [B, K], corresponding feature vector
		# dt: [B, 6], Variable
		# Jk = (feature_model(p(-delta[k], p0)) - f0) / delta[k]

		batch_size = template.size(0)
		num_points = template.size(1)

		# compute transforms
		transf = torch.zeros(batch_size, 6, 4, 4).to(template)
		for b in range(template.size(0)):
			d = torch.diag(dt[b, :]) # [6, 6]
			D = self.exp(-d) # [6, 4, 4]
			transf[b, :, :, :] = D[:, :, :]
		transf = transf.unsqueeze(2).contiguous()  #   [B, 6, 1, 4, 4]
		p = self.transform(transf, template.unsqueeze(1)) # x [B, 1, N, 3] -> [B, 6, N, 3]

		#f0 = self.feature_model(p0).unsqueeze(-1) # [B, K, 1]
		template_features = template_features.unsqueeze(-1) # [B, K, 1]
		f = self.pooling(self.feature_model(p.view(-1, num_points, 3))).view(batch_size, 6, -1).transpose(1, 2) # [B, K, 6]

		df = template_features - f # [B, K, 6]
		J = df / dt.unsqueeze(1)

		return J

	def compute_inverse_jacobian(self, J, template_features, source):
		# compute pinv(J) to solve J*x = -r
		try:
			Jt = J.transpose(1, 2) # [B, 6, K]
			H = Jt.bmm(J) # [B, 6, 6]
			B = self.inverse(H)
			pinv = B.bmm(Jt) # [B, 6, K]
			return pinv
		except RuntimeError as err:
			# # singular...?
			self.last_err = err
			g = torch.eye(4).to(source).view(1, 4, 4).expand(source.size(0), 4, 4).contiguous()
			#print(err)
			# Perhaps we can use MP-inverse, but,...
			# probably, self.dt is way too small...
			source_features = self.pooling(self.feature_model(source)) # [B, N, 3] -> [B, K]
			r = source_features - template_features
			self.feature_model.train(self.feature_model.training)
			return {}

	def handle_batchNorm(self, template, source):
		training = self.feature_model.training
		if training:
			# first, update BatchNorm modules
			template_features, source_features = self.pooling(self.feature_model(template)), self.pooling(self.feature_model(source))
		self.feature_model.eval()	# and fix them.
		return training
# -

if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	pn = PointNet()

	net = PointNetLK(pn)
	result = net(template, source)
	import ipdb; ipdb.set_trace()
