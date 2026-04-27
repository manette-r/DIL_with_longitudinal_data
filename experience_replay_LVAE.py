from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import gpytorch
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from copy import deepcopy
import pickle
import argparse
import ast
import pandas as pd

from kernel_gen import generate_kernel, generate_kernel_approx, generate_kernel_batched
from VAE import ConvVAE, SimpleVAE
from GP_def import ExactGPModel
from elbo_functions import deviance_upper_bound, elbo, KL_closed, minibatch_KLD_upper_bound, minibatch_KLD_upper_bound_iter
from model_test import MSE_test_GPapprox, MSE_test
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader
from predict_HealthMNIST import recon_complete_gen, gen_rotated_mnist_plot, variational_complete_gen
from validation16 import validate
from torch.utils.data import Dataset

from collections import OrderedDict
import itertools
import numpy as np
from torch.utils.data import Sampler

eps = 1e-6
class LoadFromFile (argparse.Action):
    """
    Read parameters from config file
    """
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().splitlines(), namespace)



def str2bool(v):

    """
    Change a string into a boolean.
    
    :param v: string 
    :return: boolean  
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_models(title, optimiser, nnet_model, gp_model, zt_list, m, H, train_x, log_var, Z, penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr, best_epoch=None, best_val_pred_mse=None):
    
    """
    Save all parameters needed to load a model.
    
    :param title: string added at the end of every file, example 'epoch75'
    :param optimiser: training optimiser 
    :param nnet_model: VAE model 
    :param gp_model: GP model 
    :param zt_list: list of inducing points
    :param m: variational mean
    :param H: variational variance
    :param train_x: auxiliary covariate information
    :param log_var: log variance of approximating variational distribution
    :param Z: inducing points
    :param penalty_term_arr: list of penalty terms 
    :param net_train_loss_arr: list of VAE model training loss 
    :param nll_loss_arr: list of loss 
    :param recon_loss_arr: list of reconstruction loss 
    :param gp_loss_arr: list of GP loss 
    :param best_epoch: integer of best epoch if a model is loaded 
    :param best_val_pred_mse: float of best mse if a model is loaded 

    """

    if 'epoch' in title : 
        torch.save(optimiser.state_dict(), os.path.join(save_path, 'optimiser'+str(title)+'.pth'))

    pd.to_pickle([penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr],
                os.path.join(save_path, 'diagnostics'+str(title)+'.pkl'))

    pd.to_pickle([train_x, log_var, Z, None], os.path.join(save_path, 'plot_values'+str(title)+'.pkl'))
    torch.save(nnet_model.state_dict(), os.path.join(save_path, 'final-vae_model'+str(title)+'.pth'))

    try:
        torch.save(gp_model.state_dict(), os.path.join(save_path, 'gp_model'+str(title)+'.pth'))
        torch.save(zt_list, os.path.join(save_path, 'zt_list'+str(title)+'.pth'))
        torch.save(m, os.path.join(save_path, 'm'+str(title)+'.pth'))
        torch.save(H, os.path.join(save_path, 'H'+str(title)+'.pth'))
    except:
        pass

    if best_epoch != None : 
        pd.to_pickle([best_epoch, best_val_pred_mse],
                    os.path.join(save_path, 'best_values'+str(title)+'.pkl'))


class ModelArgs:
    """
    Runtime parameters for the L-VAE model
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Enter configuration arguments for the model')

        self.parser.add_argument('--data_source_path', type=str, default='./data', help='Path to data')
        self.parser.add_argument('--save_path', type=str, default='./results', help='Path to save data')
        self.parser.add_argument('--csv_file_data', type=ast.literal_eval, help='Name of data file', required=False)
        self.parser.add_argument('--csv_file_test_data', type=ast.literal_eval, help='Name of test data file', required=False)
        self.parser.add_argument('--csv_file_label', type=ast.literal_eval, help='Name of label file', required=False)
        self.parser.add_argument('--csv_file_test_label', type=ast.literal_eval, help='Name of test label file', required=False)
        self.parser.add_argument('--csv_file_prediction_data', type=ast.literal_eval, help='Name of prediction data file', required=False)
        self.parser.add_argument('--csv_file_prediction_label', type=ast.literal_eval, help='Name of prediction label file', required=False)
        self.parser.add_argument('--csv_file_validation_data', type=ast.literal_eval, help='Name of validation data file', required=False)
        self.parser.add_argument('--csv_file_validation_label', type=ast.literal_eval, help='Name of validation label file', required=False)
        self.parser.add_argument('--csv_file_generation_data', type=str, help='Name of data file for image generation', required=False)
        self.parser.add_argument('--csv_file_generation_label', type=str, help='Name of label file for image generation', required=False)
        self.parser.add_argument('--mask_file', type=ast.literal_eval, help='Name of mask file', default=None)
        self.parser.add_argument('--test_mask_file', type=ast.literal_eval, help='Name of test mask file', default=None)
        self.parser.add_argument('--prediction_mask_file', type=ast.literal_eval, help='Name of prediction mask file', default=None)
        self.parser.add_argument('--validation_mask_file', type=ast.literal_eval, help='Name of validation mask file', default=None)
        self.parser.add_argument('--generation_mask_file', type=str, help='Name of mask file for image generation', default=None)
        self.parser.add_argument('--dataset_type', required=False, choices=['RotatedMNIST', 'HealthMNIST', 'Physionet'],
                                 help='Type of dataset being used.')
        self.parser.add_argument('--latent_dim', type=int, default=2, help='Number of latent dimensions')
        self.parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimensions for RNN')
        self.parser.add_argument('--id_covariate', type=int, help='Index of ID (unique identifier) covariate')
        self.parser.add_argument('--M', type=int, help='Number of inducing points')
        self.parser.add_argument('--P', type=int, help='Number of unique instances')
        self.parser.add_argument('--T', type=int, help='Number of longitudinal samples per instance')
        self.parser.add_argument('--varying_T', type=str2bool, default=False, help='Varying number of samples per instance')
        self.parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
        self.parser.add_argument('--weight', type=float, default=1,
                                 help='Trade-off parameter balancing data reconstruction and latent space prior' +
                                      ' regularisation')
        self.parser.add_argument('--num_dim', type=int, help='Number of input dimensions', required=False)
        self.parser.add_argument('--num_samples', type=int, default=1, help='Number of Monte Carlo samples')
        self.parser.add_argument('--loss_function', type=str, default='mse', help='LVAE loss function for training (mse/nll)')
        self.parser.add_argument('--type_nnet', required=False, choices=['rnn', 'conv', 'simple'],
                                 help='Type of neural network for the encoder and decoder')
        self.parser.add_argument('--type_rnn', required=False, choices=['lstm', 'gru'],
                                 help='Type of rnn for rnn-encoder')
        self.parser.add_argument('--type_KL', required=False, choices=['closed', 'other', 'GPapprox', 'GPapprox_closed'],
                                 help='Type of loss computation')
        self.parser.add_argument('--constrain_scales', type=str2bool, default=False, required=False,
                                 help='Constrain the marginal variances')
        self.parser.add_argument('--model_params', type=str, default='model_params.pth',
                                 help='Pre-trained VAE parameters')
        self.parser.add_argument('--gp_model_folder', type=str, default='./pretrainedVAE',
                                 help='Pre-trained GP model parameters')
        self.parser.add_argument('--generate_plots', type=str2bool, default=False, help='Generate plots')
        self.parser.add_argument('--iter_num', type=int, default=1, help='Iteration number. Useful for multiple runs.')
        self.parser.add_argument('--test_freq', type=int, default=50, help='Period of computing test MSE.')
        self.parser.add_argument('--cat_kernel', type=ast.literal_eval)
        self.parser.add_argument('--bin_kernel', type=ast.literal_eval)
        self.parser.add_argument('--sqexp_kernel', type=ast.literal_eval)
        self.parser.add_argument('--cat_int_kernel', type=ast.literal_eval)
        self.parser.add_argument('--bin_int_kernel', type=ast.literal_eval)
        self.parser.add_argument('--covariate_missing_val', type=ast.literal_eval)
        self.parser.add_argument('--run_tests', type=str2bool, default=False,
                                 help='Perform tests using the trained model')
        self.parser.add_argument('--run_validation', type=str2bool, default=False,
                                 help='Test the model using a validation set')
        self.parser.add_argument('--generate_images', type=str2bool, default=False,
                                 help='Generate images of unseen individuals')
        self.parser.add_argument('--results_path', type=str, required=False, help='Path to results')
        self.parser.add_argument('--f', type=open, action=LoadFromFile)
        self.parser.add_argument('--mini_batch', type=str2bool, default=False, help='Use mini-batching for training.')
        self.parser.add_argument('--hensman', type=str2bool, default=False, help='Use true mini-batch training.')
        self.parser.add_argument('--variational_inference_training', type=str2bool, default=False, help='Use variational inference training.')
        self.parser.add_argument('--memory_dbg', type=str2bool, default=False, help='Debug memory usage in training')
        self.parser.add_argument('--natural_gradient', type=str2bool, default=True, help='Use natural gradients for parameters m and H')
        self.parser.add_argument('--natural_gradient_lr', type=float, default=0.01, help='Learning rate for variational parameters m and H if natural gradient is used')
        self.parser.add_argument('--subjects_per_batch', type=int, default=20, help='Number of subjects per batch in mini-batching.')
        self.parser.add_argument('--vy_fixed', type=str2bool, default=False, help='Use fixed variance for y in VAE')
        self.parser.add_argument('--vy_init', type=float, default=1.0, help='Initial variance for y in VAE')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='Probability for dropout')
        self.parser.add_argument('--dropout_input', type=float, default=0.2, help='Probability for dropout at input layer')
        self.parser.add_argument('--first_t', type=int, default=0, help='Number of the first step to do in the learning process')
        self.parser.add_argument('--t_steps', type=int, default=0, help='Number of total steps')
        self.parser.add_argument('--num_past_step', type=int, default=-1, help='Number of the preivous step (to load models)')
        self.parser.add_argument('--len_plot_values', type=int, default=5, help='Lenght of values to unpack.')
        self.parser.add_argument('--csv_file_data_memory', type=str, help='Path to save data', required=False)
        self.parser.add_argument('--csv_file_label_memory', type=str, help='Path to save data', required=False)
        self.parser.add_argument('--csv_file_mask_memory', type=str, help='Path to save data', required=False)
        self.parser.add_argument('--n_subjects_memory', type=int, default=0, help='Number of patients to replay')
        self.parser.add_argument('--memory_batch_size', type=int, default=0, help='Memory batch size')


    def parse_options(self):
        opt = vars(self.parser.parse_args())
        return opt
    
    
class HealthMNISTDatasetConv(Dataset):
    """
    Dataset definiton for the Health MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 36 x 36.
    """

    # Si on veut donner des dataframe au lieu de donner des path 
    def __init__(self, data_source, label_source, mask_source, root_dir, transform=None, bool_original=True, df_data = None, df_mask = None, df_label = None, val_dataset_type = None):
        
        self.bool_original = bool_original
        self.val_dataset_type = val_dataset_type

        self.label_source = label_source
        self.data_source = data_source
        self.mask_source = mask_source
        self.data_source = data_source.reset_index(drop=True)
        self.label_source = label_source.reset_index(drop=True)
        self.mask_source = mask_source.reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform

    
    def __len__(self):
        # print('(len(self.data), len(self.labels), len(self.mask_df))')
        # print((len(self.data_source), len(self.label_source), len(self.mask_source)))
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # print('__getitem__ is instance1, ',self.val_dataset_type )
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)] 
        elif isinstance(key, int):
            # for k, v in self.get_item(key).items():
                # print( k, type(v))
            # print('__getitem__ is instance2, ', self.val_dataset_type, self.get_item(key), type(self.get_item(key)), key)
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        digit = self.data_source.iloc[idx, :]
        mask = self.mask_source.iloc[idx, :]

        digit = np.array(digit, dtype='uint8')
        digit = digit.reshape(36, 36)
        digit = digit[..., np.newaxis]

        mask_series = self.mask_source.iloc[idx, :]
        # convert Series → numpy array
        mask_np = mask_series.to_numpy()
        # convert numpy array → tensor
        mask = torch.tensor(mask_np, dtype=torch.float32)

        label = self.label_source.iloc[idx, :]
        # CHANGED
        # time_age,  disease_time,  subject,  gender,  disease,  location
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([6, 4, 0, 5, 3, 7])])))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample
    
    


def hensman_training(nnet_model, type_nnet, epochs, dataset, optimiser, type_KL, num_samples, latent_dim, covar_module0,
                     covar_module1, likelihoods, m, H, zt_list, P, T, varying_T, Q, weight, id_covariate, loss_function, N,
                     natural_gradient=False, natural_gradient_lr=0.01, subjects_per_batch=20, memory_dbg=False,
                     eps=1e-6, results_path=None, validation_dataset=None, generation_dataset=None,
                     prediction_dataset=None, gp_model=None, csv_file_test_data=None, csv_file_test_label=None,
                     test_mask_file=None, data_source_path=None, memory_dataset=None, memory_batch_size=0, P_train=0):

    """
    Perform training with minibatching and Stochastic Variational Inference [Hensman et. al, 2013]. 
    Compared to the initial method, continual learning and checkpoint model were added. See L-VAE supplementary materials [Ramchandran et. al 2021].

    :param nnet_model: encoder/decoder neural network model 
    :param type_nnet: type of encoder/decoder
    :param epochs: numner of epochs
    :param dataset: dataset to use in training
    :param optimiser: optimiser to be used
    :param type_KL: type of KL divergenve computation to use
    :param num_samples: number of samples to use
    :param latent_dim: number of latent dimensions
    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihoods: GPyTorch likelihood model
    :param m: variational mean
    :param H: variational variance
    :param zt_list: list of inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param Q: number of covariates
    :param weight: value for the weight
    :param id_covariate: covariate number of the id
    :param loss_function: selected loss function
    :param natural_gradient: use of natural gradients
    :param natural_gradient_lr: natural gradients learning rate
    :param subject_per_batch; number of subjects per batch (vectorisation)
    :param memory_dbg: enable debugging
    :param eps: jitter
    :param results_path: path to results
    :param validation_dataset: dataset for vaildation set
    :param generation_dataset: dataset to help with sample image generation
    :param prediction_dataset; dataset with subjects for prediction
    :param gp_mode: GPyTorch gp model
    :param c: path to test data
    :param csv_file_test_label: path to test label
    :param test_mask_file: path to test mask
    :param data_source_path: path to data source

    :return trained models and resulting losses

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert type_KL == 'GPapprox_closed'

    if varying_T:
        n_batches = (P_train + subjects_per_batch - 1)//subjects_per_batch
        dataloader = HensmanDataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch), num_workers=4)    
    else:
        batch_size = subjects_per_batch*T
        n_batches = (P*T + batch_size - 1)//(batch_size)
        dataloader = HensmanDataLoader(dataset, batch_sampler=BatchSampler(SubjectSampler(dataset, P_train, T), batch_size, drop_last=False), num_workers=4)

    if memory_dataset is not None and memory_batch_size > 0:
        memory_loader = HensmanDataLoader(
            memory_dataset,
            batch_sampler=VaryingLengthBatchSampler(
                VaryingLengthSubjectSampler(memory_dataset, id_covariate),
                memory_batch_size
            ),
            num_workers=4
        )
        memory_iter = iter(memory_loader)


    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    best_val_pred_mse = np.inf
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0
        iid_kld_sum = 0
        for batch_idx, sample_batched in enumerate(dataloader):


            #memory replay 
            if memory_dataset is not None and memory_batch_size > 0:
                try:
                    memory_batch = next(memory_iter)
                except StopIteration:
                    memory_iter = iter(memory_loader)
                    memory_batch = next(memory_iter)

                # concat new dataset and memory dataset 
                for key in ['idx', 'digit', 'label', 'mask']:
                    sample_batched[key] = torch.cat(
                        (sample_batched[key], memory_batch[key]),
                        dim=0
                    )

            optimiser.zero_grad()
            nnet_model.train()
            covar_module0.train()
            covar_module1.train()
            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]

            covariates = torch.cat((train_x[:, :id_covariate], train_x[:, id_covariate+1:]), dim=1)

            recon_batch, mu, log_var = nnet_model(data)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

            PSD_H = H if natural_gradient else torch.matmul(H, H.transpose(-1, -2))

            if varying_T:
                P_in_current_batch = torch.unique(train_x[:, id_covariate]).shape[0]
                kld_loss, grad_m, grad_H = minibatch_KLD_upper_bound_iter(covar_module0, covar_module1, likelihoods, latent_dim, m, PSD_H, train_x, mu, log_var, zt_list, P, P_in_current_batch, N, natural_gradient, id_covariate, eps)
            else:
                P_in_current_batch = N_batch // T
                kld_loss, grad_m, grad_H = minibatch_KLD_upper_bound(covar_module0, covar_module1, likelihoods, latent_dim, m, PSD_H, train_x, mu, log_var, zt_list, P, P_in_current_batch, T, natural_gradient, eps)

            recon_loss = recon_loss * P/P_in_current_batch
            nll_loss = nll_loss * P/P_in_current_batch

            if loss_function == 'nll':
                net_loss = nll_loss + kld_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * kld_loss

            net_loss.backward()
            optimiser.step()

            if natural_gradient:
                LH = torch.cholesky(H)
                iH = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LH)
                iH_new = iH + natural_gradient_lr*(grad_H + grad_H.transpose(-1,-2))
                LiH_new = torch.cholesky(iH_new)
                H = torch.cholesky_solve(torch.eye(H.shape[-1], dtype=torch.double).to(device), LiH_new).detach()
                m = torch.matmul(H, torch.matmul(iH, m) - natural_gradient_lr*(grad_m - 2*torch.matmul(grad_H, m))).detach()

            net_loss_sum += net_loss.item() / n_batches 
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)
        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr,  net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        if (not epoch % 25) and epoch != epochs:

            #checkpoint 
            save_models('_epoch'+str(epoch), optimiser, nnet_model, gp_model, zt_list, m, H, train_x, log_var, Z, penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, best_epoch, best_val_pred_mse)

            with torch.no_grad():
                nnet_model.eval()
                covar_module0.eval()
                covar_module1.eval()
                if validation_dataset is not None:
                    full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double).to(device)
                    prediction_x = torch.zeros(len(dataset), Q, dtype=torch.double).to(device)
                    for batch_idx, sample_batched in enumerate(dataloader):
                        label_id = sample_batched['idx']
                        prediction_x[label_id] = sample_batched['label'].double().to(device)
                        data = sample_batched['digit'].double().to(device)
                        covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate+1:]), dim=1)

                        mu, log_var = nnet_model.encode(data)
                        full_mu[label_id] = mu
                    val_pred_mse = validate(nnet_model, type_nnet, validation_dataset, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods, zt_list, T, weight, full_mu, prediction_x, id_covariate, loss_function, eps=1e-6)
                    if val_pred_mse < best_val_pred_mse:
                        best_val_pred_mse = val_pred_mse
                        best_epoch = epoch

                        prediction_dataloader = DataLoader(prediction_dataset, batch_sampler=VaryingLengthBatchSampler(
                            VaryingLengthSubjectSampler(prediction_dataset, id_covariate), subjects_per_batch),
                                                           num_workers=4)
                        full_mu = torch.zeros(len(prediction_dataset), latent_dim, dtype=torch.double).to(device)
                        prediction_x = torch.zeros(len(prediction_dataset), Q, dtype=torch.double).to(device)

                        with torch.no_grad():
                            for batch_idx, sample_batched in enumerate(prediction_dataloader):
                                label_id = sample_batched['idx']
                                prediction_x[label_id] = sample_batched['label'].double().to(device)
                                data = sample_batched['digit'].double().to(device)
                                covariates = torch.cat(
                                    (prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate + 1:]),
                                    dim=1)

                                mu, log_var = nnet_model.encode(data)
                                full_mu[label_id] = mu
                            covar_module0.eval()
                            covar_module1.eval()
                            if type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                                MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file,
                                                  data_source_path, type_nnet,
                                                  nnet_model, covar_module0, covar_module1, likelihoods, results_path,
                                                  latent_dim, prediction_x,
                                                  full_mu, zt_list, P, T, id_covariate, varying_T,
                                                  save_file='result_error_best.csv')

                        print('Saving better model')
                        try:
                            torch.save(nnet_model.state_dict(), (results_path+'/nnet_model_best.pth'))
                            torch.save(gp_model.state_dict(), (results_path+ '/gp_model_best.pth'))
                            torch.save(zt_list, (results_path+ '/zt_list_best.pth'))
                            torch.save(m, (results_path+ '/m_best.pth'))
                            torch.save(H, (results_path+ '/H_best.pth'))

                            if results_path and generation_dataset:
                                prediction_dataloader = DataLoader(prediction_dataset,
                                                                   batch_sampler=VaryingLengthBatchSampler(
                                                                       VaryingLengthSubjectSampler(prediction_dataset,
                                                                                                   id_covariate),
                                                                       subjects_per_batch), num_workers=4)
                                full_mu = torch.zeros(len(prediction_dataset), latent_dim, dtype=torch.double).to(
                                    device)
                                prediction_x = torch.zeros(len(prediction_dataset), Q, dtype=torch.double).to(device)
                                for batch_idx, sample_batched in enumerate(prediction_dataloader):
                                    label_id = sample_batched['idx']
                                    prediction_x[label_id] = sample_batched['label'].double().to(device)
                                    data = sample_batched['digit'].double().to(device)
                                    covariates = torch.cat((prediction_x[label_id, :id_covariate],
                                                            prediction_x[label_id, id_covariate + 1:]), dim=1)

                                    mu, log_var = nnet_model.encode(data)
                                    full_mu[label_id] = mu

                                recon_complete_gen(generation_dataset, nnet_model, type_nnet,
                                                   results_path, covar_module0,
                                                   covar_module1, likelihoods, latent_dim,
                                                   './data', prediction_x, full_mu, epoch,
                                                   zt_list, P, T, id_covariate, varying_T)
                        except e:
                            print(e)
                            print('Saving intermediate model failed!')
                            pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m, H, best_epoch


if __name__ == "__main__":
    """
    Root file for running Continual learning (experience replay) with L-VAE. To load a model from a previous epoch, the model must include '_epochXX', XX corresponds to the epoch number. 
    The first step must already have been completed in order to load it.

    Run command: python experience_replay_LVAE.py --f=path_to_config-file.txt 
    """

    # create parser and set variables
    opt = ModelArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    #load first dataset 
    first_dataset_data = pd.read_csv(os.path.join(data_source_path, csv_file_data[0]), header=0)
    first_dataset_label = pd.read_csv(os.path.join(data_source_path, csv_file_label[0]), header=0)
    first_dataset_mask = pd.read_csv(os.path.join(data_source_path, mask_file[0]), header=0)

    #load memory dataset 
    if csv_file_data_memory != None : 
            memory_data_source = pd.read_csv(csv_file_data_memory, header=0)
            memory_label_source = pd.read_csv(csv_file_label_memory, header=0)
            memory_mask_source = pd.read_csv(csv_file_mask_memory, header=0)
            memory_dataset = HealthMNISTDatasetConv(data_source=memory_data_source, label_source=memory_label_source,
                                             mask_source=memory_mask_source, root_dir=data_source_path,
                                             transform=transforms.ToTensor(), bool_original=True, val_dataset_type='memory')
    else : 
        #create memory dataset 
        memory_data_source = pd.DataFrame(columns=first_dataset_data.columns) 
        memory_label_source = pd.DataFrame(columns=first_dataset_label.columns)  
        memory_mask_source = pd.DataFrame(columns=first_dataset_mask.columns) 

    #load first dataset 
    first_dataset = HealthMNISTDatasetConv(data_source=first_dataset_data, label_source=first_dataset_label,
                                        mask_source=first_dataset_mask, root_dir=data_source_path,
                                        transform=transforms.ToTensor(), bool_original=True, val_dataset_type='first_dataset')
    
    N = len(first_dataset)
    Q = len(first_dataset[0]['label'])


    nnet_model = ConvVAE(latent_dim, num_dim, vy_init=vy_init, vy_fixed=vy_fixed,
                                p_input=dropout_input, p=dropout).double().to(device)
    #Loading first step model 
    nnet_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')))

    
    likelihoods = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim]),
        noise_constraint=gpytorch.constraints.GreaterThan(1.000E-08)).to(device)

    likelihoods.noise = 1
    likelihoods.raw_noise.requires_grad = False

    covar_module0, covar_module1 = generate_kernel_batched(latent_dim,
                                                            cat_kernel, bin_kernel, sqexp_kernel,
                                                            cat_int_kernel, bin_int_kernel,
                                                            covariate_missing_val, id_covariate)


    if num_past_step == -1 : 
        num_past_step = ""
        
    #Loading parameters for first step model
    if len_plot_values == 5 : 
        with open(gp_model_folder+"/plot_values"+str(num_past_step)+".pkl", "rb") as f:
            [train_x, prediction_mu, log_var, Z, label_id] = pickle.load(f) 
    elif len_plot_values == 4 : 
        with open(gp_model_folder+"/plot_values"+str(num_past_step)+".pkl", "rb") as f:
            [train_x, log_var, Z, label_id] = pickle.load(f) 

    gp_model = ExactGPModel(train_x, Z.type(torch.DoubleTensor), likelihoods,
                            covar_module0 + covar_module1).to(device)

    # initialise inducing points
    zt_list = torch.zeros(latent_dim, M, Q, dtype=torch.double).to(device)
    for i in range(latent_dim):
        zt_list_idx = torch.randperm(N, device=train_x.device)[:M] 
        zt_list[i] = train_x[zt_list_idx].clone().detach() 



    covar_module0.train().double()
    covar_module1.train().double()
    likelihoods.train().double()

    try:
        gp_model.load_state_dict(torch.load((gp_model_folder+'/gp_model'+str(num_past_step)+'.pth'), map_location=torch.device(device)))
        zt_list = torch.load((gp_model_folder+ '/zt_list'+str(num_past_step)+'.pth'), map_location=torch.device(device))
        print('Loaded GP models')
    except Exception as e :
        print('GP model loading failed!')
        raise print(e)
        

    m = torch.randn(latent_dim, M, 1).double().to(device).detach()
    H = (torch.randn(latent_dim, M, M)/10).double().to(device).detach()

    H = torch.matmul(H, H.transpose(-1, -2)).detach().requires_grad_(False)

    try:
        m = torch.load((gp_model_folder+'/m'+str(num_past_step)+'.pth'), map_location=torch.device(device)).detach()
        H = torch.load((gp_model_folder+'/H'+str(num_past_step)+'.pth'), map_location=torch.device(device)).detach()
        print('Loaded natural gradient values')
    except:
        print('Loading natural gradient values failed!')
        pass

    nnet_model.train()
    adam_param_list = []
    adam_param_list.append({'params': covar_module0.parameters()})
    adam_param_list.append({'params': covar_module1.parameters()})
    adam_param_list.append({'params': nnet_model.parameters()})
    optimiser = torch.optim.Adam(adam_param_list, lr=1e-3)

    #loop for steps
    for i in range(first_t, t_steps): 
        print('\nStep ', i)
        #load new train dataset
        dataset_data = pd.read_csv(os.path.join(data_source_path, csv_file_data[i]), header=0)
        dataset_label = pd.read_csv(os.path.join(data_source_path, csv_file_label[i]), header=0)
        dataset_mask = pd.read_csv(os.path.join(data_source_path, mask_file[i]), header=0)
        dataset_data = dataset_data.reset_index(drop=True)
        dataset_label = dataset_label.reset_index(drop=True)
        dataset_mask = dataset_mask.reset_index(drop=True)
        dataset = HealthMNISTDatasetConv(data_source=dataset_data, label_source=dataset_label,
                                                    mask_source=dataset_mask, root_dir=data_source_path,
                                                    transform=transforms.ToTensor(), bool_original=True, val_dataset_type='train')
        
        #load new validation dataset
        dataset_validation_data = pd.read_csv(os.path.join(data_source_path, csv_file_validation_data[i]), header=0)
        dataset_validation_label = pd.read_csv(os.path.join(data_source_path, csv_file_validation_label[i]), header=0)
        dataset_validation_mask = pd.read_csv(os.path.join(data_source_path, validation_mask_file[i]), header=0)
        dataset_validation = HealthMNISTDatasetConv(data_source=dataset_validation_data, label_source=dataset_validation_label,
                                                    mask_source=dataset_validation_mask, root_dir=data_source_path,
                                                    transform=transforms.ToTensor(), bool_original=True, val_dataset_type='val')
        
        #load new test dataset
        dataset_prediction_data = pd.read_csv(os.path.join(data_source_path, csv_file_prediction_data[i]), header=0)
        dataset_prediction_label = pd.read_csv(os.path.join(data_source_path, csv_file_prediction_label[i]), header=0)
        dataset_prediction_mask = pd.read_csv(os.path.join(data_source_path, prediction_mask_file[i]), header=0)
        dataset_prediction = HealthMNISTDatasetConv(data_source=dataset_prediction_data, label_source=dataset_prediction_label,
                                                    mask_source=dataset_prediction_mask, root_dir=data_source_path,
                                                    transform=transforms.ToTensor(), bool_original=True, val_dataset_type='test')
                
        #P is the number of subjects in total 
        P = len(dataset_label['subject'].unique())+len(memory_label_source['subject'].unique())
        N = len(dataset)+len(memory_dataset)
        Q = len(dataset[0]['label'])
        P_train = len(dataset_label['subject'].unique())
        print(P, P_train)

        #train 
        _ = hensman_training(nnet_model, type_nnet, epochs, dataset,
                             optimiser, type_KL, num_samples, latent_dim,
                             covar_module0, covar_module1, likelihoods, m,
                             H, zt_list, P, T, varying_T, Q, weight,
                             id_covariate, loss_function, N, natural_gradient, natural_gradient_lr,
                             subjects_per_batch, memory_dbg, eps,
                             results_path, validation_dataset=dataset_validation,
                             generation_dataset=None, prediction_dataset=dataset_prediction, gp_model=gp_model, csv_file_test_data=csv_file_test_data[i],
                             csv_file_test_label=csv_file_test_label[i], test_mask_file=test_mask_file[i],
                             data_source_path=data_source_path, memory_dataset=memory_dataset, memory_batch_size=memory_batch_size, P_train=P_train)
        m, H = _[5], _[6]
        penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr = _[0], _[1], _[2], _[3], _[4]

        #save
        save_models(str(i), optimiser, nnet_model, gp_model, zt_list, m, H, train_x, log_var, Z, penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr)


        #save memory dataset 
        memory_data_source.to_csv(os.path.join(save_path, 'memory_dataset_data'+str(i)+'.csv'), index=False)
        memory_label_source.to_csv(os.path.join(save_path, 'memory_dataset_label'+str(i)+'.csv'), index=False)
        memory_mask_source.to_csv(os.path.join(save_path, 'memory_dataset_mask'+str(i)+'.csv'), index=False)


        #a sub dataset in randomly choose to be added in memory dataset 
        all_subjects = dataset_label['subject'].unique()
        shortlist_subjects, _ = train_test_split(all_subjects, train_size=n_subjects_memory)
    
        dataset_label_short = dataset_label[dataset_label['subject'].isin(shortlist_subjects)]
        dataset_data_short = deepcopy(dataset_data.loc[dataset_label_short.index.tolist()])
        dataset_mask_short = deepcopy(dataset_mask.loc[dataset_label_short.index.tolist()])
        print('Memory dataset add : ', len(dataset_label_short))
        

        #Memory updated  
        memory_data_source = pd.concat([memory_data_source, dataset_data_short], ignore_index=True)
        memory_label_source = pd.concat([memory_label_source, dataset_label_short], ignore_index=True)
        memory_mask_source = pd.concat([memory_mask_source, dataset_mask_short], ignore_index=True)
        memory_dataset = HealthMNISTDatasetConv(data_source=memory_data_source, label_source=memory_label_source,
                                            mask_source=memory_mask_source, root_dir=data_source_path,
                                            transform=transforms.ToTensor(), bool_original=True, val_dataset_type='memory')
        print('Memory update : ', len(memory_label_source))

