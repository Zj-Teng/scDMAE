import torch
import torch.nn.functional as func
import dgl
import lightning.pytorch as pl

from dgl.nn.pytorch.conv import SAGEConv
from torch import nn
from src.utils.Functional import self_conv, compute_loss_para
from torch import optim
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


class BaseBlock(pl.LightningModule):
    def __init__(self, n_feat, layer_depth, dropout, activation):
        super(BaseBlock, self).__init__()

        if not isinstance(n_feat, int):
            raise Exception('n_feat must be an int')
        if not isinstance(layer_depth, int):
            raise Exception('layer_depth must be an int')
        if not isinstance(dropout, float):
            raise Exception('dropout must be a float')
        if not isinstance(activation, str):
            raise Exception('activation must be a str')

        self.n_feat = n_feat
        if activation == 'elu':
            activation = nn.ELU()
        elif activation == 'relu':
            activation = nn.ReLU()

        self.module = nn.Sequential()
        for _ in range(layer_depth):
            self.module.append(
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features=n_feat, out_features=n_feat),
                    activation
                )
            )

    def forward(self, x):
        out = x
        for m in self.module:
            out = m(out)

        return out


class BaseSAGE(pl.LightningModule):
    def __init__(self, n_feat, layer_depth):
        super(BaseSAGE, self).__init__()

        if not isinstance(n_feat, int):
            raise Exception('n_feat must be an int')
        if not isinstance(layer_depth, int):
            raise Exception('layer_depth must be an int')

        self.n_feat = n_feat

        self.module = nn.Sequential()
        for _ in range(layer_depth):
            self.module.append(
                SAGEConv(in_feats=n_feat, out_feats=n_feat, activation=torch.relu, aggregator_type='mean')
            )

    def forward(self, block, edge_weight=None):
        out = block.ndata['feature']
        if edge_weight is None:
            for m in self.module:
                out = m(block, out)
        else:
            for m in self.module:
                out = m(block, out, edge_weight=edge_weight)

        return out


class TransLayer(pl.LightningModule):
    def __init__(self, n_feat):
        super(TransLayer, self).__init__()

        if not isinstance(n_feat, int):
            raise Exception('n_gene must be an int')

        self.n_feat = n_feat

        self.mean = BaseSAGE(n_feat=n_feat, layer_depth=1)
        self.var = BaseSAGE(n_feat=n_feat, layer_depth=1)
        self.norm = nn.BatchNorm1d(n_feat)

    def forward(self, block):
        mean = self.mean(block)
        var = self.var(block)

        gaussian_noise = torch.randn(mean.size(0), self.n_feat, device=self.device)
        z = mean + gaussian_noise * torch.exp(var)
        z = self.norm(z)

        return mean, var, z


class CVGAE(pl.LightningModule):
    def __init__(self, n_gene, n_feat):
        super(CVGAE, self).__init__()

        if not isinstance(n_gene, int):
            raise Exception('n_gene must be an int')
        if not isinstance(n_feat, int):
            raise Exception('n_feat must be an int')

        self.in_features = n_gene
        self.out_features = n_feat

        self.encoder = BaseSAGE(n_feat=n_feat, layer_depth=3)
        self.transformer = TransLayer(n_feat=n_feat)
        self.decoder = BaseBlock(
            n_feat=n_gene, layer_depth=1, dropout=0.2, activation='relu'
        )

    def forward(self, block, edge_weight):
        h = self.encoder(block, edge_weight)
        block.ndata['feature'] = h

        mean, var, z = self.transformer(block)
        inferred_net = self_conv(z, z)
        inferred_net = self.decoder(inferred_net)
        inferred_net = 2. * nn.functional.normalize(inferred_net)

        return inferred_net, mean, var, z


class CLGVAE(pl.LightningModule):
    def __init__(self, n_gene, n_feat):
        super(CLGVAE, self).__init__()

        self.n_gene = n_gene
        self.n_feat = n_feat

        self.teacher = CVGAE(n_gene, n_feat)
        self.t_readout = BaseBlock(n_feat, 1, 0., 'elu')
        self.student = CVGAE(n_gene, n_feat)
        self.s_readout = BaseBlock(n_feat, 1, 0., 'elu')

        self.au_roc = BinaryAUROC()
        self.au_prc = BinaryAveragePrecision()

    def forward(self, block):
        raw_features = block.ndata['feature']
        g_net, mean, var, gh = self.teacher(block, None)
        position_pairwise = torch.nonzero(g_net > 1., as_tuple=True)
        # mask = torch.where(g_net > 1., True, False)
        # edge_weight = torch.masked_select(g_net, mask)
        # edge_weight = torch.sigmoid(edge_weight.unsqueeze(0).reshape(-1, 1))
        new_block = dgl.graph(position_pairwise, idtype=torch.int64, num_nodes=self.n_gene)
        new_block.ndata['feature'] = raw_features
        self.print('new_block num_edges:{}'.format(new_block.num_edges()))
        rg_net, _, _, reh = self.student(new_block, None)

        t_readout = torch.mean(gh, dim=0)
        s_readout = torch.mean(reh, dim=0)
        t_readout = self.t_readout(t_readout)
        s_readout = self.s_readout(s_readout)

        return g_net, rg_net, t_readout, s_readout, mean, var
        # return g_net, mean, var

    def training_step(self, batch, batch_idx):
        src_nodes, dst_nodes, batch = batch
        truth_net = batch.adjacency_matrix().to_dense().to(torch.int64)
        weight_tensor, norm = compute_loss_para(truth_net)

        g_net, rg_net, t_readout, s_readout, mean, var = self(batch)
        logits = g_net + rg_net
        # Can not make sense that truth_net must be cast to the torch.float64
        bce_loss = norm * func.binary_cross_entropy_with_logits(
            logits.view(-1), truth_net.view(-1).to(torch.float64), weight=weight_tensor
        )
        kl_loss = 0.5 / logits.size(0) * (1 + 2 * var - mean ** 2 - torch.exp(var) ** 2).sum(1).mean()

        loss = bce_loss - kl_loss
        self.print('loss:{}'.format(loss))
        self.log('loss', loss)

        training_step_au_roc = self.au_roc(logits, truth_net)
        training_step_au_prc = self.au_prc(logits, truth_net)
        self.print('auroc:{},auprc;{}'.format(training_step_au_roc, training_step_au_prc))
        self.au_roc.reset()
        self.au_prc.reset()

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)

        return optimizer

    def backward(self, loss, **kwargs):
        loss.backward()

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def on_train_end(self):
        pass
