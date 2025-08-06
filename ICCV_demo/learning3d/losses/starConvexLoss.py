import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# loss function
def h_fn(predictions, labels):
    loss = nn.CrossEntropyLoss()
    return loss(predictions, labels)



# convexStart loss function
class convexStarLoss(nn.Module):
    def __init__(self, Lambda, mu, n_samples, balancer, scaler=10,h_fn=h_fn):
        super(convexStarLoss, self).__init__()
        self.Lambda = nn.Parameter(torch.tensor([1.0 * Lambda]), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor([1.0 * mu]), requires_grad=False)
        # self.mu = nn.Parameter(torch.tensor([1.0 * mu]), requires_grad=True)
        self.balancer = balancer
        self.n_samples = n_samples
        self.scaler = scaler

    
    def forward(self, predictions, labels):
        loss1 = h_fn(predictions, labels) # h_label
        loss2 = 0.

        mu = self.mu # mu >=0
        Lambda = 1 / (1 + torch.exp(-self.Lambda)) # 0<Lambda<1
        for _ in range(self.n_samples):
            noise = (torch.rand(predictions.size(), device=predictions.get_device())  - 0.5 ) / self.scaler
            noisy_labels = (labels + noise).softmax(dim=1)
            lambda_noisy_labels = (labels + Lambda * noise).softmax(dim=1)
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
