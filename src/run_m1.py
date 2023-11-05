# %%
import pandas as pd
import torch
import os
import numpy as np

from src.module import Smoother, ExpEncoder, TpyEncoder, ExpDecoder, NoiseSeparator
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from utils.data import ScDataset
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_float32_matmul_precision('high')


def normalize_data(X):
    X = X.T
    X_norm = X.copy()
    X_norm = X_norm.astype('float32')
    libs = X.sum(axis=1)
    med = np.median(libs)
    norm_factor = np.diag(med / libs)
    data_norm = np.dot(norm_factor, X_norm)
    data_norm = pd.DataFrame(data_norm, index=None, columns=None)
    return data_norm.T


def logtranform_data(X):
    Y = X.copy()
    Y = np.log10(Y + 1.)
    return Y


def main():
    processed_data_file = '../Data/10k+/AMB.csv'
    module_path = '../output/10k+/AMB_module.pt'
    exp_save_path = '../output/10k+/AMB_scDMAE_DM.csv'
    noise_save_path = '../output/10k+/AMB_noise.csv'
    fusefeature_save_path = '../output/10k+/AMB_fusefeature.csv'

    print('Preprocess data')
    process_data = pd.read_csv(processed_data_file, header=0, index_col=0)
    n_gene = process_data.shape[0]
    n_cell = process_data.shape[1]
    #process_data1 = process_data.T
    process_data = logtranform_data(normalize_data(process_data))

    feature_exp = torch.from_numpy(np.array(process_data.values))
    feature_exp = feature_exp.to(torch.float)

    n_tpy = 1024
    feature_tpy = torch.rand((n_gene, n_tpy))
    feature_tpy = feature_tpy.to(torch.float)

    # Build an iterable loader and an optimizer
    batch_size = 256
    max_epochs = 500

    sc_dataset = ScDataset(feature_exp, feature_tpy, batch_size)
    sc_dataloader = DataLoader(
        sc_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    exp_encoder = ExpEncoder(in_features=n_cell, out_features=n_tpy, hid_features=[4096, 2048])
    tpy_encoder = TpyEncoder(in_features=n_tpy, out_features=n_tpy, layer_depth=3)
    exp_decoder = ExpDecoder(in_features=n_tpy + n_tpy, out_features=n_cell, hid_features=[4096, 2048])
    noise_separator = NoiseSeparator(n_feat=n_cell, batch_size=batch_size)
    smoother = Smoother(exp_encoder, tpy_encoder, exp_decoder, noise_separator, n_cell, n_tpy)
    logger = TensorBoardLogger('../tf-logs', name='smoother')
    trainer = Trainer(
        accelerator='gpu', max_epochs=max_epochs, devices='auto', logger=False, fast_dev_run=False,
        callbacks=[EarlyStopping(monitor='loss', mode="min", patience=5)]
    )

    trainer.fit(smoother, sc_dataloader)
    torch.save(smoother.state_dict(), module_path)
    smoother.eval()
    # smoother.load_state_dict(torch.load(module_path))

    predict_dataloader = DataLoader(
        sc_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    results = trainer.predict(smoother, predict_dataloader)

    exp_y = torch.cat([results[i][0] for i in range(len(results))], dim=0)
    exp_y = pd.DataFrame(exp_y.detach().cpu().numpy(), columns=None, index=None)
    exp_y.to_csv(exp_save_path)
    noise = torch.cat([results[i][1] for i in range(len(results))], dim=0)
    noise = pd.DataFrame(noise.detach().cpu().numpy(), columns=None, index=None)
    noise.to_csv(noise_save_path)
    #f_feature = torch.cat([results[i][2] for i in range(len(results))], dim=0)
    # = pd.DataFrame(f_feature.detach().cpu().numpy(), columns=None, index=None)
    #f_feature.to_csv(fusefeature_save_path)

if __name__ == '__main__':
    main()

    # task_class = ['Clustering', 'DifferentialExpression', 'Dimension reduction', 'TrajectoryInference']
    # clustering_class = ['Celseq2_5cl_p1', 'Celseq2_5cl_p2', 'Celseq2_5cl_p3', 'Yan']
    # DE_class = ['ECvsDEC', 'H1vsDEC', 'H9vsDEC', 'HFFvsDEC', 'NPCvsDEC', 'TBvsDEC']
    # DR_class = ['10x_5cl', 'Loh', 'Pollen', 'Zeisel']
    # TR_class = ['Deng', 'Petropoulos']

    # run_task = []
    # for e in clustering_class:
    #     run_task.append(
    #         [
    #             '../Data/{}/HVG2000/{}/{}_HVG.csv'.format(task_class[0], e, e),
    #             '../output/{}/HVG2000/{}/module.pt'.format(task_class[0], e),
    #             '../output/{}/HVG2000/{}/gene_exp.csv'.format(task_class[0], e),
    #             '../output/{}/HVG2000/{}/noise.csv'.format(task_class[0], e)
    #         ]
    #     )
    # for e in DE_class:
    #     run_task.append(
    #         [
    #             '../Data/{}/{}/HVG2000/{}_HVG.csv'.format(task_class[1], e, e),
    #             '../output/{}/{}/HVG2000/module.pt'.format(task_class[1], e),
    #             '../output/{}/{}/HVG2000/gene_exp.csv'.format(task_class[1], e),
    #             '../output/{}/{}/HVG2000/noise.csv'.format(task_class[1], e)
    #         ]
    #     )
    # for e in DR_class:
    #     run_task.append(
    #         [
    #             '../Data/{}/HVG2000/{}/{}_HVG.csv'.format(task_class[2], e, e),
    #             '../output/{}/{}/HVG2000/module.pt'.format(task_class[2], e),
    #             '../output/{}/{}/HVG2000/gene_exp.csv'.format(task_class[2], e),
    #             '../output/{}/{}/HVG2000/noise.csv'.format(task_class[2], e)
    #         ]
    #     )
    # for e in TR_class:
    #     run_task.append(
    #         [
    #             '../Data/{}/HVG2000/{}/{}_HVG.csv'.format(task_class[3], e, e),
    #             '../output/{}/{}/HVG2000/module.pt'.format(task_class[3], e),
    #             '../output/{}/{}/HVG2000/gene_exp.csv'.format(task_class[3], e),
    #             '../output/{}/{}/HVG2000/noise.csv'.format(task_class[3], e)
    #         ]
    #     )
