import torch
from torchaudio.transforms import Resample
from geomloss import SamplesLoss

class EEGSimpleConv(torch.nn.Module):
    def __init__(self, fm, n_convs, resampling, kernel_size,n_chan,n_classes,sfreq,n_subjects=None):
        super(EEGSimpleConv, self).__init__()
        #self.pool = torch.nn.AvgPool1d(init_pool)
        '''
        Parameters:
        fm (int): Number of out_channels for conv layer
        n_chan (int): Number of in_channels for conv layers
        n_convs (int): Number of convolutional blocks
        kernel_size (int or tuple): Size of the convolving kernel
        sfreq (int): Original frequency of the signal
        resampling (int): Desired frequency of the signal
        n_subjects (int): Number of subjects in the dataset, if None no subject regularization is applied

        '''
        self.rs = Resample(orig_freq=sfreq,new_freq=resampling)
        self.conv = torch.nn.Conv1d(n_chan, fm, kernel_size = kernel_size, padding = kernel_size // 2, bias = False)
        self.bn = torch.nn.BatchNorm1d(fm)
        self.blocks = []
        newfm = fm
        oldfm = fm
        for i in range(n_convs):
            if i > 0:
                newfm = int(1.414 * newfm)
            self.blocks.append(torch.nn.Sequential(
                (torch.nn.Conv1d(oldfm, newfm, kernel_size = kernel_size, padding = kernel_size // 2, bias = False)),
                (torch.nn.BatchNorm1d(newfm)),
                (torch.nn.MaxPool1d(2) if i > 0 - 1 else torch.nn.MaxPool1d(1)),
                (torch.nn.ReLU()),
                (torch.nn.Conv1d(newfm, newfm, kernel_size = kernel_size, padding = kernel_size // 2, bias = False)),
                (torch.nn.BatchNorm1d(newfm)),
                (torch.nn.ReLU())
            ))
            oldfm = newfm
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.fc = torch.nn.Linear(oldfm, n_classes)
        self.fc2 = torch.nn.Linear(oldfm, n_subjects) if n_subjects else None #Subject regularization
        self.domain_discriminator = torch.nn.Linear(oldfm, 1) if n_subjects else None



    def forward(self, x):
        y = torch.relu(self.bn(self.conv(self.rs(x.contiguous()))))
        for seq in self.blocks:
            y = seq(y)
        y = y.mean(dim = 2)
        return (self.fc(y),self.fc2(y), y) if self.fc2 else (self.fc(y),y)

import torch
import torch.nn as nn
import torch.optim as optim


# Critic (Domain Discriminator): Using Lipschitz 1 networks
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, z):
        return self.model(z)

# Gradient Penalty
def gradient_penalty(critic, z_s, z_t, device):
    alpha = torch.rand(z_s.size(0), 1, device=device)
    alpha = alpha.expand_as(z_s)
    
    interpolates = alpha * z_s + (1 - alpha) * z_t
    interpolates = interpolates.requires_grad_(True)
    
    critic_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates, device=device),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    return ((gradients_norm - 1) ** 2).mean()

# Training Loop
def train_wasserstein_discriminator(feature_extractor, critic, data_loader_source, data_loader_target, device):
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
    feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=1e-4)
    
    lambda_gp = 10  # Gradient penalty coefficient
    for epoch in range(num_epochs):
        for (x_s, _), (x_t, _) in zip(data_loader_source, data_loader_target):
            x_s, x_t = x_s.to(device), x_t.to(device)
            
            # Extract features
            z_s = feature_extractor(x_s)
            z_t = feature_extractor(x_t)
            
            # Train Critic
            critic_optimizer.zero_grad()
            d_s = critic(z_s)
            d_t = critic(z_t)
            
            loss_critic = d_s.mean() - d_t.mean()
            gp = gradient_penalty(critic, z_s, z_t, device)
            loss_critic_total = -loss_critic + lambda_gp * gp
            loss_critic_total.backward()
            critic_optimizer.step()
            
            # Train Feature Extractor
            feature_optimizer.zero_grad()
            z_s = feature_extractor(x_s)
            z_t = feature_extractor(x_t)
            loss_feature = -critic(z_s).mean() + critic(z_t).mean()
            loss_feature.backward()
            feature_optimizer.step()
        
        print(f"Epoch {epoch}, Critic Loss: {loss_critic.item()}, Feature Loss: {loss_feature.item()}")
