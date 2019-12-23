import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch.nn import Sequential as Seq
from gcn_lib import BasicConv, MLP


class ResGCN(torch.nn.Module):
    def __init__(self, input, output):
        super(ResGCN, self).__init__()
        self.conv = GCNConv(input, output)

    def forward(self, x, edge_index):
        res = self.conv(x, edge_index)
        x = x + res
        return x


class DeepResGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepResGCN, self).__init__()

        num_features = opt.num_feature
        channels = opt.n_filters

        act = opt.act
        norm = opt.norm
        bias = opt.bias

        c_growth = channels

        self.n_blocks = opt.n_blocks

        self.head = GCNConv(num_features, channels)
        self.backbone = Seq(*[ResGCN(channels, channels) for i in range(self.n_blocks - 1)])

        self.fusion_block = MLP([channels + c_growth * (self.n_blocks - 1), 1024], act, norm, bias)

        self.prediction = Seq(*[MLP([1024, 512], act, norm, bias),
                                MLP([512, 256], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                MLP([256, opt.n_classes], None, None, bias)])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data):
        batch_size = data.num_graphs

        feats = [self.head(data.x, data.edge_index)]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1], data.edge_index))
        feats = torch.cat(feats, dim=1)
        fusion = self.fusion_block(feats)
        fusion = fusion.reshape(batch_size, 17, 1024)
        fusion = torch.mean(fusion, dim=1)
        pred = self.prediction(fusion)
        return pred
