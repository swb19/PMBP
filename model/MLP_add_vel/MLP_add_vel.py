import torch.nn as nn
import torch

class NNPred(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_size, dropout=0.05):
        super(NNPred, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.relu = nn.ReLU()

        self.fcp1 = nn.Linear(6*2, 128)
        self.fcp2 = nn.Linear(6*2, 128)
        self.fcp = nn.Linear(256, 128)

        self.fcv1 = nn.Linear(5, 48)
        self.fcv2 = nn.Linear(5, 48)
        self.fcv = nn.Linear(96, 48)

        self.fca1 = nn.Linear(4, 24)
        self.fca2 = nn.Linear(4, 24)
        self.fca = nn.Linear(24*2, 24)

        self.fc1 = nn.Linear(128 + 48 + 24, 64)
        self.fc2 = nn.Linear(64, output_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, pos1, pos2):
        # preprocess
        vel1 = pos1[:,1:,:] - pos1[:,:-1,:]
        acc1 = vel1[:,1:,:] - vel1[:,:-1,:]
        vel1 = (vel1**2).sum(-1)**0.5
        acc1 = (acc1**2).sum(-1)**0.5
        vel2 = pos2[:,1:,:] - pos2[:,:-1,:]
        acc2 = vel2[:,1:,:] - vel2[:,:-1,:]
        vel2 = (vel2**2).sum(-1)**0.5
        acc2 = (acc2**2).sum(-1)**0.5

        # position encoding
        N, T, V = pos1.size()
        pos1 = pos1.view(N,T*V)
        N, T, V = pos2.size()
        pos2 = pos2.view(N, T * V)

        outp1 = self.relu(self.fcp1(pos1))
        outp2 = self.relu(self.fcp2(pos2))
        outp = torch.cat([outp1, outp2], 1)
        outp = self.dropout(outp)
        outp = self.relu(self.fcp(outp))

        # velocity encoding
        outv1 = self.relu(self.fcv1(vel1))
        outv2 = self.relu(self.fcv2(vel2))
        outv = torch.cat([outv1, outv2], 1)
        outv = self.dropout(outv)
        outv = self.relu(self.fcv(outv))

        # acceleration encoding
        outa1 = self.relu(self.fca1(acc1))
        outa2 = self.relu(self.fca2(acc2))
        outa = torch.cat([outa1, outa2], 1)
        outa = self.dropout(outa)
        outa = self.relu(self.fca(outa))

        # combine and predict
        out = torch.cat([outp, outv, outa], 1)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        output = out.squeeze()
        return output
