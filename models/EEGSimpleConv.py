import torch
from torchaudio.transforms import Resample

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
        

    def compute_wasserstein_distance(self, n_subjects):

        return torch.norm(x - y, p = 2, dim = 1)


    def forward(self, x):
        y = torch.relu(self.bn(self.conv(self.rs(x.contiguous()))))
        for seq in self.blocks:
            y = seq(y)
        y = y.mean(dim = 2)
        return (self.fc(y),self.fc2(y)) if self.fc2 else self.fc(y)