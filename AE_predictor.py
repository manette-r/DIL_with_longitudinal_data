import os
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch
import gpytorch
import pickle
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import argparse
import ast
import re 
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import Dataset
from copy import deepcopy
from torch import nn,optim
import pymc as pm

from GP_def import ExactGPModel
from VAE import ConvVAE, SimpleVAE
from elbo_functions import elbo, KL_closed, deviance_upper_bound
from kernel_gen import generate_kernel, generate_kernel_approx, generate_kernel_batched
from model_test import MSE_test
# from predict_HealthMNIST import gen_rotated_mnist_seqrecon_plot
from training import hensman_training, minibatch_training, standard_training, variational_inference_optimization
from validation16 import validate
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader, batch_predict_varying_T

eps = 1e-6

class LoadFromFile (argparse.Action):
    """
    Read parameters from config file
    """
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().splitlines(), namespace)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ModelArgs:
    """
    Runtime parameters for the L-VAE model
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Enter configuration arguments for the model')

        self.parser.add_argument('--data_source_path', type=str, default='./data', help='Path to data')
        self.parser.add_argument('--save_path', type=str, default='./results', help='Path to save data')
        self.parser.add_argument('--csv_file_data', type=str, help='Name of data file', required=False)
        self.parser.add_argument('--csv_file_test_data', type=str, help='Name of test data file', required=False)
        self.parser.add_argument('--csv_file_label', type=str, help='Name of label file', required=False)
        self.parser.add_argument('--csv_file_test_label', type=str, help='Name of test label file', required=False)
        self.parser.add_argument('--csv_file_prediction_data', type=ast.literal_eval, help='Name of prediction data file', required=False)
        self.parser.add_argument('--csv_file_prediction_label', type=ast.literal_eval, help='Name of prediction label file', required=False)
        self.parser.add_argument('--csv_file_validation_data', type=str, help='Name of validation data file', required=False)
        self.parser.add_argument('--csv_file_validation_label', type=str, help='Name of validation label file', required=False)
        self.parser.add_argument('--csv_file_generation_data', type=ast.literal_eval, help='Name of data file for image generation', required=False)
        self.parser.add_argument('--csv_file_generation_label', type=ast.literal_eval, help='Name of label file for image generation', required=False)
        self.parser.add_argument('--mask_file', type=str, help='Name of mask file', default=None)
        self.parser.add_argument('--test_mask_file', type=str, help='Name of test mask file', default=None)
        self.parser.add_argument('--prediction_mask_file', type=ast.literal_eval, help='Name of prediction mask file', default=None)
        self.parser.add_argument('--validation_mask_file', type=str, help='Name of validation mask file', default=None)
        self.parser.add_argument('--generation_mask_file', type=ast.literal_eval, help='Name of mask file for image generation', default=None)
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
        self.parser.add_argument('--first_t', type=int, default=0, help='xx')
        self.parser.add_argument('--t_steps', type=int, default=0, help='xx')
        self.parser.add_argument('--cl_step', type=int, default=-1, help='Continual learning step to indicate which version of object to load, if -1 load in the normal way for plot_values, -2 plot_values without prediction_mu')
        self.parser.add_argument('--col_target', type=str, help='Target for the Mixed effects model', required=False, default='angle')
        self.parser.add_argument('--mem_formula_cols', type=str, help='Columns for the Mixed effects model', required=False, default='time_age +disease')
        self.parser.add_argument('--mem_cols_df', type=ast.literal_eval, default=['angle', 'time_age', 'subject', 'disease'], help='C=List of columns of the target and columns for the formula')
        self.parser.add_argument('--mem_path', type=str, required=False, help='Path to mem model', default='')
        self.parser.add_argument('--domain_test_name_list', type=ast.literal_eval, help='List of domains test')
        self.parser.add_argument('--mlp_path', type=str, required=False, help='Path to mlp', default='')
        self.parser.add_argument('--bool_f_compare', type=str2bool, default=False,help='Compare function before test to find the right one corresponding to the data.')
        self.parser.add_argument('--nlme_trace_path', type=str, default=None, help='Pre-trained NLME')
        self.parser.add_argument('--nlme_comparaison_name', type=str, default='nlme_minitrace_', help='Mini NLME for comparison')
        self.parser.add_argument('--nlme_comparaison_time', type=str, default='nlme_minitrace_time_info.pkl', help='Time normalisation info for mini NLME for comparison')
        self.parser.add_argument('--name_model_choosen', type=str, default='exp', help='Function to apply in NLME')
        self.parser.add_argument('--mlp_cols_df', type=ast.literal_eval, default=['time_age','disease'], help='C=List of columns of the target and columns for the formula')
        self.parser.add_argument('--model_type', type=str, default='mlp', help='Type of model for the predictor (mlp/nlme/mem)')
        self.parser.add_argument('--version_run', type=str, default='', help='For the title')
        self.parser.add_argument('--mlp_optimizer_path', type=str, required=False, help='Path to mlp', default='')
        self.parser.add_argument('--csv_file_data_memory', type=str, help='Name of the generic file without the cl_step or the .csv extension', required=False)
        self.parser.add_argument('--csv_file_label_memory', type=str, help='Name of the generic file without the cl_step or the .csv extension', required=False)
        self.parser.add_argument('--csv_file_mask_memory', type=str, help='Name of the generic file without the cl_step or the .csv extension', required=False)



    def parse_options(self):
        opt = vars(self.parser.parse_args())
        return opt
    


class HealthMNISTDatasetConv(Dataset):
    """
    Dataset definiton for the Health MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 36 x 36.
    """

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
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)] 
        elif isinstance(key, int):

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
        angle = torch.Tensor(np.nan_to_num(np.array(label[np.array([2])])))
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([6, 4, 0, 5, 3, 7])])))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask, 'angle': angle}
        return sample
    


def gen_rotated_mnist_seqrecon_plot(X, recon_X, labels_recon, labels_train, save_file='recon_complete.pdf'):
    """
    Function to generate Health MNIST digits.
    
    """    
    num_sets = 8
    fig, ax = plt.subplots(4 * num_sets - 1, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
            ax__.axis('off')
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(12, 20)

    for j in range(num_sets):
        begin_data = seq_length_train*j
        end_data = seq_length_train*(j+1)

        begin_label = seq_length_full*2*j
        mid_label = seq_length_full*(2*j+1)
        end_label = seq_length_full*2*(j+1)
        
        time_steps = labels_train[begin_data:end_data, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j, int(t)].imshow(np.reshape(X[begin_data + i, :], [36, 36]), cmap='gray')
        
        time_steps = labels_train[begin_label:mid_label, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j + 1, int(t)].imshow(np.reshape(recon_X[begin_label + i, :], [36, 36]), cmap='gray')
        
        time_steps = labels_train[mid_label:end_label, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j + 2, int(t)].imshow(np.reshape(recon_X[mid_label + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file, bbox_inches='tight')
    plt.close('all')
    
def recon_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, 
                       covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, 
                       prediction_mu, epoch, zt_list, P, T, id_covariate, varying_T=False, file_name_Z='Z_pred'):
    """
    Function to generate rotated MNIST digits.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Generating images - length of dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=4)
    total_mse = 0
    total_samples = 0
    all_mse = []
    all_Z_pred = []
    all_label = []
    all_label_id = []
    all_angle = []

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            angle = sample_batched['angle']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred)
            print("prediction_mu.shape, Z_pred.shape",prediction_mu.shape, Z_pred.shape)
            print("data.shape, recon_Z.shape", data.shape, recon_Z.shape)
            string_title = validation_mask_file
            filename = 'recon_complete'+str(string_title[:7])+'.pdf' 
            
            print("recon_Z.shape[0] == 2 * data.shape[0]", recon_Z.shape[0] == 2 * data.shape[0])
            assert recon_Z.shape[0] == data.shape[0], "Pb car data[0:160] et recon_Z[0:320], recon_z contient mu et z_gp, Mismatch between prediction and ground truth"

            mse, _ = nnet_model.loss_function(recon_Z, data, mask)  
            total_mse += torch.sum(mse).item()
            total_samples += mse.shape[0]

            all_Z_pred.append(Z_pred.detach().cpu())
            all_angle.append(angle.detach().cpu())
            all_label.append(label.detach().cpu())
            all_label_id.append(label_id.detach().cpu())
            all_mse.append(mse.detach().cpu())
            # reconstruction loss
            print('160 == ',data.shape[0], 'and 360 == ', recon_Z.shape[0])

            seq_len = int(torch.max(label[:,0]).item()) + 1
            num_sets = min(8, data.shape[0] // seq_len)
            max_display = num_sets * seq_len
            gen_rotated_mnist_seqrecon_plot(data[0:max_display, :].cpu(), recon_Z[0:max_display, :].cpu(), label[0:max_display, :].cpu(), label[0:max_display, :].cpu(),
                                            save_file=os.path.join(results_path, filename))
        print("Pour ", total_samples, "samples ")

        Z_pred_all = torch.cat(all_Z_pred, dim=0)
        label_all = torch.cat(all_label, dim=0)
        angle_all = torch.cat(all_angle, dim=0)
        label_id_all = torch.cat(all_label_id, dim=0)

        label_df = pd.DataFrame(label_all.numpy(), columns=[
            "time_age",
            "disease_time",
            "subject",
            "gender",
            "disease",
            "location"
        ])
        label_df["angle"] = angle_all.numpy()

        try : 
            pd.to_pickle({"Z_pred": Z_pred_all.numpy(),"label_df": label_df, "label_id": label_id_all.numpy()}, os.path.join(results_path, file_name_Z+'.pkl'))
        except Exception as e : 
            print(e)

        all_mse = torch.cat(all_mse)
        mean_mse = torch.mean(all_mse)
        std_mse = torch.std(all_mse)
        stderr_mse = std_mse / torch.sqrt(torch.tensor(len(all_mse), dtype=torch.float))
        mse_mean = (total_mse/total_samples)
        print("MSE GP = ",mse_mean)
        print("MSE GP2 = ",mean_mse, "Std MSE = ", std_mse, "Std Error = ", stderr_mse)

        return mse_mean, Z_pred_all, label_df, label_id_all
    

    
def MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
                      covar_module0, covar_module1, likelihoods, results_path, latent_dim, prediction_x, prediction_mu,
                      zt_list, P, T, id_covariate, varying_T=False, save_file='result_error.csv'):

    """
    Function to compute Mean Squared Error of test set with GP approximationö
    
    """

    print("Running tests with a test set")
    dataset_type = 'HealthMNIST'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type_nnet == 'conv':
        if dataset_type == 'HealthMNIST':

            data_buffer = pd.read_csv(os.path.join(data_source_path,csv_file_test_data), header=0)
            label_buffer = pd.read_csv(os.path.join(data_source_path,csv_file_test_label), header=0)
            mask_buffer = pd.read_csv(os.path.join(data_source_path,test_mask_file), header=0)
            test_dataset = HealthMNISTDatasetConv(data_source=data_buffer,
                                                  label_source=label_buffer,
                                                  mask_source=mask_buffer, root_dir=data_source_path,
                                                  transform=transforms.ToTensor(), bool_original=True, val_dataset_type='test')


    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)

            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)
            
            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
            print('MSE_test_GPapprox ',pred_results)


def nlme_train(df_train, col_subject, col_target, col_time, non_linear_f):
    mean_theta0 = df_train.groupby(col_subject)[col_target].max().mean()
    std_theta0 = df_train.groupby(col_subject)[col_target].max().std()

    patient_speeds = df_train.groupby(col_subject).apply(
        lambda g: np.polyfit(g[col_time], g[col_target], 1)[0]  # pente de la rotation
    )
    mean_theta1 = patient_speeds.mean()
    std_theta1 = patient_speeds.std()

    print("theta0_pop ~", mean_theta0, "+/-", std_theta0)
    print("theta1_pop ~", mean_theta1, "+/-", std_theta1)

    n_patients_train = len(df_train[col_subject].unique())

    with pm.Model() as nlme_model:

        # rotation max
        theta0_pop = pm.Normal("theta0_pop", mu=mean_theta0, sigma=2*std_theta0)
        # speed
        theta1_pop = pm.Normal("theta1_pop", mu=mean_theta1, sigma=2*std_theta1)   

        sigma_theta0 = pm.HalfNormal("sigma_theta0", sigma=std_theta0)
        sigma_theta1 = pm.HalfNormal("sigma_theta1", sigma=std_theta1)
        
        theta0_offset = pm.Normal("theta0_offset", mu=0, sigma=1, shape=n_patients_train)
        theta1_offset = pm.Normal("theta1_offset", mu=0, sigma=1, shape=n_patients_train)
        
        theta0_i = theta0_pop + theta0_offset * sigma_theta0
        theta1_i = theta1_pop + theta1_offset * sigma_theta1
        

        patient_idx = df_train[col_subject].factorize()[0]
        time = df_train[col_time].values
        rotation_train = df_train[col_target].values
        
        y_pred = non_linear_f(time, [theta0_i[patient_idx], theta1_i[patient_idx]])
        
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=np.std(rotation_train))
        # Likelihood
        y_obs = pm.Normal("y_obs", mu=y_pred, sigma=sigma_eps, observed=rotation_train)
        
        trace = pm.sample(1000, tune=1000, target_accept=0.9, random_seed=42)
    
    return trace 


def f_linear(t, theta):
    return theta[0] + theta[1] * t

def f_exp(t, theta):
    return theta[0] * (1 - np.exp(-theta[1] * t))

def f_quadratic(t, theta):
    return theta[0] + theta[1] * t + theta[2] * t**2


def f_logistic(t, theta):
    return theta[0] / (1 + np.exp(-theta[1]*(t-theta[2])))

def f_basis_expansion(t, theta):
    return theta[0] + theta[1]*t + theta[2]*t**2 + theta[3]*t**3

import arviz as az

def compare_f(df_train, col_subject, col_target, col_time, models):

    traces, time_info = nlme_train_comparison(df_train, col_subject, col_target, col_time, 500)
    print('\nComparaison des y_pred_mean : ')
    for name in traces.keys():
        print(name, traces[name]['y_pred_mean'])
    #datframe 
    comparaison = az.compare({name: traces[name]["trace"] for name in traces}, ic="loo")
    print(comparaison)

    return traces, time_info

def nlme_train_comparison(df_train, col_subject, col_target, col_time, models, nb_echantillon=1000):

    time_raw = df_train[col_time].values
    time_mean = time_raw.mean()
    time_std = time_raw.std()
    time = (time_raw - time_mean) / time_std

    mean_theta0 = df_train.groupby(col_subject)[col_target].max().mean()
    std_theta0 = df_train.groupby(col_subject)[col_target].max().std()

    patient_speeds = df_train.groupby(col_subject).apply(
        lambda g: np.polyfit((g[col_time].values - time_mean) / time_std, g[col_target], 1)[0]  # pente de la rotation
    )
    mean_theta1 = patient_speeds.mean()
    std_theta1 = patient_speeds.std()

    mean_theta2 = 1.0  
    std_theta2 = 0.5
    mean_theta3 = 1.0
    std_theta3 = 0.5

    print("theta0_pop ~", mean_theta0, "+/-", std_theta0)
    print("theta1_pop ~", mean_theta1, "+/-", std_theta1)
    print("theta2_pop ~", mean_theta2, "+/-", std_theta2)
    print("theta3_pop ~", mean_theta3, "+/-", std_theta3)

    n_patients_train = len(df_train[col_subject].unique())
    patient_idx = df_train[col_subject].factorize()[0]
    rotation_train = df_train[col_target].values

    traces = {}

    for name, model_info in models.items():
        f = model_info["f"]
        n_theta = model_info["n_theta"]

        with pm.Model() as nlme_model:

            # rotation max
            theta0_pop = pm.Normal("theta0_pop", mu=mean_theta0, sigma=2*std_theta0)
            # speed
            theta1_pop = pm.Normal("theta1_pop", mu=mean_theta1, sigma=2*std_theta1)   

            sigma_theta0 = pm.HalfNormal("sigma_theta0", sigma=std_theta0)
            sigma_theta1 = pm.HalfNormal("sigma_theta1", sigma=std_theta1)
            
            theta0_offset = pm.Normal("theta0_offset", mu=0, sigma=1, shape=n_patients_train)
            theta1_offset = pm.Normal("theta1_offset", mu=0, sigma=1, shape=n_patients_train)
            
            theta0_i = theta0_pop + theta0_offset * sigma_theta0
            theta1_i = theta1_pop + theta1_offset * sigma_theta1

            if n_theta == 3 : 
                theta2_pop = pm.Normal("theta2_pop", mu=mean_theta2, sigma=2*std_theta2)
                y_pred = f(time, [theta0_i[patient_idx], theta1_i[patient_idx], theta2_pop])

            elif n_theta == 4 : 
                theta2_pop = pm.Normal("theta2_pop", mu=mean_theta2, sigma=2*std_theta2)
                theta3_pop = pm.Normal("theta3_pop", mu=mean_theta3, sigma=2*std_theta3)
                y_pred = f(time, [theta0_i[patient_idx], theta1_i[patient_idx], theta2_pop, theta3_pop])

            else : 
                y_pred = f(time, [theta0_i[patient_idx], theta1_i[patient_idx]])

            # Noise 
            sigma_eps = pm.HalfNormal("sigma_eps", sigma=np.std(rotation_train))
            # Likelihood
            y_obs = pm.Normal("y_obs", mu=y_pred, sigma=sigma_eps, observed=rotation_train)
            
            trace = pm.sample(nb_echantillon, tune=nb_echantillon, target_accept=0.9, random_seed=42, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

            # Posterior predictive
            ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])
            y_pred_mean = ppc.posterior_predictive["y_obs"].mean(dim=("chain", "draw")).values  # <- ici tu obtiens la moyenne pour toutes les observations
            
            traces[name] = {"trace": trace,  "y_pred_mean": y_pred_mean}

    return traces, {"mean": time_mean, "std": time_std} 


def nlme_test(model_info, df_test, df_true, trace, time_infos, col_subject, col_target, col_time, title_plot="NLME Test"):
    f = model_info["f"]
    n_theta = model_info["n_theta"] 
    time_mean = time_infos["mean"]
    time_std = time_infos["std"]

    test_patients = df_test[col_subject].unique()
    rotation_test = df_test[col_target].values
    time_idx_test = df_test[col_time].values
    patient_idx_test = df_test[col_subject].factorize()[0]

    results = {}

    pred_samples_list = []
    pred_mean_list = []

    for pid in test_patients:
        df_pid = df_test[df_test[col_subject] == pid]
        y_cond = df_pid[col_target].values
        t_cond = df_pid[col_time].values

        #true value 
        df_pid_true = df_true[df_true[col_subject] == pid]
        df_pid_test = df_test[df_test[col_subject] == pid]

        df_pid_future = df_pid_true.merge(
            df_pid_test[[col_time]].drop_duplicates(),
            on=col_time,
            how="left",
            indicator=True
        )

        df_pid_future = df_pid_future[df_pid_future["_merge"] == "left_only"]

        t_true = df_pid_future[col_time].values
        y_true = df_pid_future[col_target].values

        t_cond_scaled = (t_cond - time_mean) / time_std
        t_true_scaled = (t_true - time_mean) / time_std

        print('pour le patient ', pid)
       
        print('y_cond', y_cond)
        print('t_cond', t_cond)
        print('t_true',t_true)
        print('y_true',y_true)
        print('t_true_scaled', t_true_scaled)
        print('t_cond_scaled', t_cond_scaled)

        if len(y_cond) == 1:
            # population only
            theta0_pop_samples = trace.posterior["theta0_pop"].values.flatten()
            theta1_pop_samples = trace.posterior["theta1_pop"].values.flatten()

            if n_theta == 3 : 
                theta2_pop_samples = trace.posterior["theta2_pop"].values.flatten()

            elif n_theta == 4 : 
                theta2_pop_samples = trace.posterior["theta2_pop"].values.flatten()
                theta3_pop_samples = trace.posterior["theta3_pop"].values.flatten()

            y_samples = []

            for i in range(len(theta0_pop_samples)):
                if n_theta == 3 : 
                    mu_test = f(t_true_scaled, [theta0_pop_samples[i], theta1_pop_samples[i], theta2_pop_samples[i]])

                elif n_theta == 4 : 
                    mu_test = f(t_true_scaled, [theta0_pop_samples[i], theta1_pop_samples[i], theta2_pop_samples[i], theta3_pop_samples[i]])

                else : 
                    mu_test = f(t_true_scaled, [theta0_pop_samples[i], theta1_pop_samples[i]])

                y_samples.append(mu_test)

            y_samples = np.array(y_samples)
            mu_pred = y_samples.mean(axis=0)

        else:
            # conditional prediction
            with pm.Model() as cond_model:
                theta0_pop_mean = trace.posterior["theta0_pop"].mean().values
                theta1_pop_mean = trace.posterior["theta1_pop"].mean().values
                sigma_theta0_mean = trace.posterior["sigma_theta0"].mean().values
                sigma_theta1_mean = trace.posterior["sigma_theta1"].mean().values
                sigma_eps_mean = trace.posterior["sigma_eps"].mean().values

                theta0_offset = pm.Normal("theta0_offset", mu=0, sigma=1)
                theta1_offset = pm.Normal("theta1_offset", mu=0, sigma=1)

                theta0_i = theta0_pop_mean + theta0_offset * sigma_theta0_mean
                theta1_i = theta1_pop_mean + theta1_offset * sigma_theta1_mean

                if n_theta == 2:
                    mu = f(t_cond_scaled, [theta0_i, theta1_i])

                elif n_theta == 3:
                    theta2_pop_mean = trace.posterior["theta2_pop"].mean().values
                    mu = f(t_cond_scaled, [theta0_i, theta1_i, theta2_pop_mean])

                elif n_theta == 4:
                    theta2_pop_mean = trace.posterior["theta2_pop"].mean().values
                    theta3_pop_mean = trace.posterior["theta3_pop"].mean().values
                    mu = f(t_cond_scaled, [theta0_i, theta1_i, theta2_pop_mean, theta3_pop_mean])

                pm.Normal("y_obs_cond", mu=mu, sigma=sigma_eps_mean, observed=y_cond)

                trace_cond = pm.sample(500, tune=500, target_accept=0.9,random_seed=42, progressbar=False)

            # forecasting for a patient
            theta0_i_samples = theta0_pop_mean + \
                trace_cond.posterior["theta0_offset"].values.flatten() * sigma_theta0_mean
            theta1_i_samples = theta1_pop_mean + \
                trace_cond.posterior["theta1_offset"].values.flatten() * sigma_theta1_mean
            
            t_true_scaled_buffer = t_true_scaled[None, :]
            if n_theta == 2:
                mu_pred = np.mean(f(t_true_scaled_buffer, [theta0_i_samples[:, None], theta1_i_samples[:, None]]),axis=0)
            elif n_theta == 3:
                mu_pred = np.mean(f(t_true_scaled_buffer, [theta0_i_samples[:, None],theta1_i_samples[:, None],theta2_pop_mean]),axis=0)
            elif n_theta == 4:
                mu_pred = np.mean(f(t_true_scaled_buffer, [theta0_i_samples[:, None],theta1_i_samples[:, None],theta2_pop_mean,theta3_pop_mean]),axis=0)
        
        pred_samples_list.append(mu_pred)
        for t_val, mu_val in zip(t_true, mu_pred):
            pred_mean_list.append((pid, t_val, mu_val))
        
        print('mu_pred', mu_pred)

    pred_samples_test = np.array(pred_samples_list)  #list per patient
    pred_mean_test = np.array(pred_mean_list, dtype=object)

    results["test"] = {
        "pred_samples": pred_samples_test,
        "pred_mean": pred_mean_test
    }
    print("rotation test:", len(rotation_test))
    
    print("pred:", len(pred_mean_test))
    

    pred_df = pd.DataFrame(pred_mean_list, columns=[col_subject, col_time, "pred"])

    df_test_keys = df_test[[col_subject, col_time]].drop_duplicates()

    df_true_future = df_true.merge(
        df_test_keys,
        on=[col_subject, col_time],
        how="left",
        indicator=True
    )

    df_true_future = df_true_future[df_true_future["_merge"] == "left_only"]
    df_true_future = df_true_future.drop(columns="_merge")

    df_merge = df_true_future.merge(pred_df, on=[col_subject, col_time])
    patient_idx_merge = df_merge[col_subject].factorize()[0]
    print("patients unique (merge):",  df_merge[col_subject].nunique())
    print("Duplicates pred_df:", pred_df.duplicated([col_subject, col_time]).sum())
    print("Duplicates df_true_future:", df_true_future.duplicated([col_subject, col_time]).sum())
    
    print("len pred_df:", len(pred_df))
    print("len df_true_future:", len(df_true_future))
    print("len df_merge:", len(df_merge))

    # visualisation
    plot_nlme_scatter(time=df_merge[col_time].values,rotation=df_merge[col_target].values,patient_idx=patient_idx_merge ,patients=df_merge[col_subject].unique(),pred_mean=df_merge['pred'].values,mode="scatter",title=title_plot)
    plot_nlme_time(time=df_merge[col_time].values,rotation=df_merge[col_target].values,patient_idx=patient_idx_merge ,patients=df_merge[col_subject].unique(),pred_mean=df_merge['pred'].values,mode="plot_time",title=title_plot)

    return results


def plot_nlme_time(time, rotation, patient_idx, patients, pred_mean, mode="train", title="NLME"):
    plt.figure(figsize=(8, 6))

    for pid in patients:
        mask = patient_idx == pid

        t = time[mask]
        y = rotation[mask]
        y_pred = pred_mean[mask]

        #chronological order
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        y_pred = y_pred[order]

        plt.plot(t, y, 'o-', label=f"y P{pid}")
        plt.plot(t, y_pred, 'x--', label=f"y_pred P{pid}")

    plt.xlabel("Temps")
    plt.ylabel("Rotation")
    plt.title(title + f" ({mode})")

    if mode == "train":
        plt.legend()

    # save
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{title}_{mode}.png"),
                    dpi=300, bbox_inches='tight')

    plt.show()
    
def plot_nlme_scatter(time, rotation, patient_idx, patients, pred_mean, mode="train", title="NLME"):
    plt.figure(figsize=(6, 6))

    for pid in patients:
        mask = patient_idx == pid

        y = rotation[mask]
        y_pred = pred_mean[mask]

        plt.scatter(y, y_pred, label=f"Patient {pid}", alpha=0.7)

    all_y = np.concatenate([rotation[patient_idx == pid] for pid in patients])
    min_v, max_v = all_y.min(), all_y.max()
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', label="y = y_pred")

    plt.xlabel("Rotation réelle (y)")
    plt.ylabel("Rotation prédite (y_pred)")
    plt.title(title + f" ({mode})")

    if mode == "train":
        plt.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{title}_{mode}.png"),
                    dpi=300, bbox_inches='tight')

    plt.show()


def mem_model_train(latent_dim, formula_target, formula_cols, Z_train, cols_df, train_label): 

    formula = formula_target+" ~ " + " + ".join([f'Z{i}' for i in range(latent_dim)]) + " + "+formula_cols

    train_dataset_modified = pd.DataFrame(Z_train, columns=[f'Z{i}' for i in range(latent_dim)])
    train_dataset_modified[cols_df] = train_label[cols_df]

    # Mixed effects model 
    model = smf.mixedlm(formula, data=train_dataset_modified, groups="subject")
    result_mem = model.fit()
    print(result_mem.summary())

    fixed_df = pd.DataFrame({
    'coef': result_mem.fe_params,
    'std_err': result_mem.bse,
    't': result_mem.tvalues,
    'p': result_mem.pvalues
    })
    print('\n\n',fixed_df)

    fixed_df['coef'].plot(kind='bar', yerr=fixed_df['std_err'])
    plt.ylabel('Coefficient')
    plt.title('Fixed effects')
    plt.show()

    return result_mem


def mem_test(list_Z_test, latent_dim, cols_df, list_test_label, mem_model, col_target, title= 'Mixed effects model with LVAE on Single baseline'):

    df_result = pd.DataFrame(columns=['domain','r2', 'mse'])
   
    min_target = None 
    max_target = None

    #plot with all domains
    fig_global, ax_global = plt.subplots(figsize=(6,6))

    for i in range(len(list_Z_test)) : 

        [domain_i, full_z_GP] = list_Z_test[i]
        test_dataset_modified = pd.DataFrame(full_z_GP, columns=[f'Z{i}' for i in range(latent_dim)])

        [domain_i, test_label] = list_test_label[i]
        test_dataset_modified[cols_df] = test_label[cols_df]

        y_pred_mem = mem_model.predict(test_dataset_modified)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(test_dataset_modified[col_target], y_pred_mem, alpha=0.5)
        ax.set_title(f'{domain_i}')
        ax.set_xlabel('True '+col_target)
        ax.set_ylabel('Predicted '+col_target)
        if save_path is not None:
            domain_i_text = str(domain_i).replace(' ', '')
            fig.savefig(os.path.join(save_path, f"mem_plot_{col_target}_{domain_i_text}.png"),
                        dpi=300, bbox_inches='tight')

        plt.close(fig)  

        ax_global.scatter(test_dataset_modified[col_target], y_pred_mem, alpha=0.5, label=f"{domain_i}")

        r2 = r2_score(test_dataset_modified[col_target], y_pred_mem)
        mse = mean_squared_error(test_dataset_modified[col_target], y_pred_mem)
        print("Domain : ",domain_i)
        print("MEM R² =", r2)
        print("MEM MSE =", mse)
        df_result.loc[len(df_result)] = [domain_i, r2, mse]
        if i == 0 : 
            min_target = test_dataset_modified[col_target].min()
            max_target = test_dataset_modified[col_target].max()
        else : 
            if test_dataset_modified[col_target].max() > max_target : 
                max_target = test_dataset_modified[col_target].max()

            if test_dataset_modified[col_target].min() < min_target : 
                min_target = test_dataset_modified[col_target].min()

    ax_global.set_xlabel('True '+col_target)
    ax_global.set_ylabel('Predicted'+col_target)
    ax_global.legend()
    ax_global.set_title(title)
    if save_path is not None:
        fig_global.savefig(os.path.join(save_path, f"mem_plot_{col_target}_ALL.png"),
                    dpi=300, bbox_inches='tight')

    plt.show()

    print('Target has a minimum value of ',min_target, ' and maximum value of ', max_target)
    return df_result


def mlp_train(mlp, optimizer, full_Z_pred_train, col_added, train_label):
        
    time_age_train = torch.tensor(train_label[['time_age']].to_numpy(dtype=float), dtype=torch.double)
    disease_train = torch.tensor(train_label[['disease']].to_numpy(dtype=float), dtype=torch.double)
    X_train = torch.cat([full_Z_pred_train, time_age_train, disease_train], dim=1)
    y_train = torch.tensor(train_label[col_target].to_numpy(dtype=float).reshape(-1,1), dtype=torch.double)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()
    #MLP train 
    epochs = 100

    for epoch in range(epochs):
        mlp.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = mlp(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_loader.dataset)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss = {epoch_loss:.4f}")

    print('MLP training finished.')
    return mlp, optimizer


def mlp_test(mlp, col_target, list_Z_test, list_test_label, device, title ='MLP with LVAE on Single baseline'):

    df_result = pd.DataFrame(columns=['domain','r2', 'mse'])

    #MLP simple test
    mlp.eval()

    #plot with all domains 
    fig_global, ax_global = plt.subplots(figsize=(6,6))
    
    for i in range(len(list_Z_test)) : 

        [domain_i, full_z_GP] = list_Z_test[i]


        [domain_i, test_label] = list_test_label[i]
        time_age_test  = torch.tensor(test_label[['time_age']].to_numpy(dtype=float), dtype=torch.double)
        disease_test  = torch.tensor(test_label[['disease']].to_numpy(dtype=float), dtype=torch.double)

        X_test  = torch.cat([full_z_GP, time_age_test, disease_test], dim=1)

        y_test  = torch.tensor(test_label[col_target].to_numpy(dtype=float).reshape(-1,1), dtype=torch.double)

        test_dataset = TensorDataset(X_test, y_test)

        test_loader = DataLoader(test_dataset, batch_size=64)
        
        with torch.no_grad():
            y_pred = mlp(X_test.to(device)).cpu().numpy()


        r2_test = r2_score(y_test.numpy(), y_pred)
        mse_test = mean_squared_error(y_test.numpy(), y_pred)

        print("MLP Test R² =", r2_test)
        print("MLP Test MSE =", mse_test)
        df_result.loc[len(df_result)] = [domain_i, r2_test, mse_test]

        #plot just for one domain
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(y_test.numpy(), y_pred, alpha=0.5)
        ax.set_title(f'{domain_i}')
        ax.set_xlabel('True '+col_target)
        ax.set_ylabel('Predicted '+col_target)
        if save_path is not None:
            domain_i_text = str(domain_i).replace(' ', '')
            fig.savefig(os.path.join(save_path, f"mlp_plot_{col_target}_{domain_i_text}.png"),
                        dpi=300, bbox_inches='tight')

        plt.close(fig)  

        ax_global.scatter(y_test.numpy(), y_pred, alpha=0.5, label=f"{domain_i}")

    ax_global.set_xlabel('True '+col_target)
    ax_global.set_ylabel('Predicted '+col_target)
    ax_global.legend()
    ax_global.set_title(title)
    if save_path is not None:
        fig_global.savefig(os.path.join(save_path, f"mlp_plot_{col_target}_ALL.png"),
                    dpi=300, bbox_inches='tight')

    plt.show()

    return df_result

def test_LVAE(idx_pred_gen, nnet_model, covar_module0, covar_module1, likelihoods, zt_list, Q, file_name_Z='Z_pred'):

    #alors c'est train 
    if idx_pred_gen == -1 : 
        data_buffer_pred = pd.read_csv(os.path.join(data_source_path,csv_file_data), header=0)
        label_buffer_pred = pd.read_csv(os.path.join(data_source_path,csv_file_label), header=0)
        mask_buffer_pred = pd.read_csv(os.path.join(data_source_path,mask_file), header=0)
        data_buffer_gen = deepcopy(data_buffer_pred)
        label_buffer_gen = deepcopy(label_buffer_pred)
        mask_buffer_gen = deepcopy(mask_buffer_pred)
    else : 
        data_buffer_pred = pd.read_csv(os.path.join(data_source_path,csv_file_prediction_data[idx_pred_gen]), header=0)
        label_buffer_pred = pd.read_csv(os.path.join(data_source_path,csv_file_prediction_label[idx_pred_gen]), header=0)
        mask_buffer_pred = pd.read_csv(os.path.join(data_source_path,prediction_mask_file[idx_pred_gen]), header=0)

        data_buffer_gen = pd.read_csv(os.path.join(data_source_path,csv_file_generation_data[idx_pred_gen]), header=0)
        label_buffer_gen = pd.read_csv(os.path.join(data_source_path,csv_file_generation_label[idx_pred_gen]), header=0)
        mask_buffer_gen = pd.read_csv(os.path.join(data_source_path,generation_mask_file[idx_pred_gen]), header=0)

    prediction_dataset = HealthMNISTDatasetConv(data_source=data_buffer_pred,
                                                label_source=label_buffer_pred,
                                                mask_source=mask_buffer_pred,
                                                root_dir=data_source_path,
                                                transform=transforms.ToTensor(), bool_original=True, val_dataset_type='prediction')
    
    generation_dataset = HealthMNISTDatasetConv(data_source=data_buffer_gen,
                                                label_source=label_buffer_gen,
                                                mask_source=mask_buffer_gen,
                                                root_dir=data_source_path,
                                                transform=transforms.ToTensor(), bool_original=True, val_dataset_type='test')

    prediction_dataloader = DataLoader(prediction_dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(prediction_dataset, id_covariate), subjects_per_batch), num_workers=4)
    full_mu = torch.zeros(len(prediction_dataset), latent_dim, dtype=torch.double).to(device)
    prediction_x = torch.zeros(len(prediction_dataset), Q, dtype=torch.double).to(device)

    with torch.no_grad():
        
        for batch_idx, sample_batched in enumerate(prediction_dataloader):
            label_id = sample_batched['idx']
            prediction_x[label_id] = sample_batched['label'].double().to(device)
            data = sample_batched['digit'].double().to(device)
            covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate+1:]), dim=1)


            mu, log_var = nnet_model.encode(data)
            full_mu[label_id] = mu

    # MSE test

    with torch.no_grad():
        covar_module0.eval()
        covar_module1.eval()   
        MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet,
                                  nnet_model, covar_module0, covar_module1, likelihoods,  results_path, latent_dim, prediction_x,
                                  full_mu, zt_list, P, T, id_covariate, varying_T)
    with torch.no_grad():
        covar_module0.eval()
        covar_module1.eval()                
        mse_tt, Z_pred_all, label_df, label_id_all = recon_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, full_mu, -1, zt_list, P, T, id_covariate, varying_T, file_name_Z=file_name_Z)
       
        return  Z_pred_all, label_df


if __name__ == "__main__":
    """
    Root file for running L-VAE.
    
    Run command: python LVAE_test.py --f=path_to_config-file.txt 
    """

    # create parser and set variables
    opt = ModelArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))
    
    data_buffer = pd.read_csv(os.path.join(data_source_path,csv_file_data), header=0)
    label_buffer = pd.read_csv(os.path.join(data_source_path,csv_file_label), header=0)
    mask_buffer = pd.read_csv(os.path.join(data_source_path,mask_file), header=0)

    #load memory dataset 
    if csv_file_data_memory != None : 

        
        memory_data_source = pd.read_csv(gp_model_folder+"/"+csv_file_data_memory+str(cl_step)+".csv", header=0)
        memory_label_source = pd.read_csv(gp_model_folder+"/"+csv_file_label_memory+str(cl_step)+".csv", header=0)
        memory_mask_source = pd.read_csv(gp_model_folder+"/"+csv_file_mask_memory+str(cl_step)+".csv", header=0)

        #concat directement avec train 
        data_buffer = pd.concat([memory_data_source, data_buffer], ignore_index=True)
        label_buffer = pd.concat([memory_label_source, label_buffer], ignore_index=True)
        mask_buffer = pd.concat([memory_mask_source, mask_buffer], ignore_index=True)

    dataset = HealthMNISTDatasetConv(data_source=data_buffer,
                                                label_source=label_buffer,
                                                mask_source=mask_buffer,
                                                root_dir=data_source_path,
                                                transform=transforms.ToTensor(), bool_original=True, val_dataset_type='dataset')
    col_subject = 'subject'
    col_target = 'angle'
    col_time = 'time_age'
    #subject,digit,angle,disease,disease_time,gender,time_age,location,domain,dataset

    df_train = label_buffer[[col_subject, col_target, col_time]]


    if model_type == 'mlp' or model_type == 'mem' :
        
        #Models
        N = len(dataset)
        Q = len(dataset[0]['label'])


        nnet_model = ConvVAE(latent_dim, num_dim, vy_init=vy_init, vy_fixed=vy_fixed,
                                    p_input=dropout_input, p=dropout).double().to(device)
        nnet_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')))

        #GP déjà entrainé
        likelihoods = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim]),
            noise_constraint=gpytorch.constraints.GreaterThan(1.000E-08)).to(device)

        likelihoods.noise = 1
        likelihoods.raw_noise.requires_grad = False

        covar_module0, covar_module1 = generate_kernel_batched(latent_dim,
                                                                cat_kernel, bin_kernel, sqexp_kernel,
                                                                cat_int_kernel, bin_int_kernel,
                                                                covariate_missing_val, id_covariate)

        if cl_step == -1 : 
            with open(gp_model_folder+"/plot_values.pkl", "rb") as f:
                [train_x, prediction_mu, log_var, Z, label_id] = pickle.load(f) 
        elif cl_step == -2 : 
            with open(gp_model_folder+"/plot_values.pkl", "rb") as f:
                [train_x, log_var, Z, label_id] = pickle.load(f)         
        else :
            with open(gp_model_folder+"/plot_values"+str(cl_step)+".pkl", "rb") as f:
                [train_x, log_var, Z, label_id] = pickle.load(f) 

        #si c'est la valeur par défaut alors ne rien ajouter 
        if cl_step == -1 or cl_step == -2 :
            cl_step = ''
            
        gp_model = ExactGPModel(train_x, Z.type(torch.DoubleTensor), likelihoods,
                                covar_module0 + covar_module1).to(device)


        covar_module0.train().double()
        covar_module1.train().double()
        likelihoods.train().double()

        try:
            gp_model.load_state_dict(torch.load(gp_model_folder+'/gp_model'+str(cl_step)+'.pth', map_location=torch.device(device)))
            zt_list = torch.load(gp_model_folder+'/zt_list'+str(cl_step)+'.pth', map_location=torch.device(device))
            print('Loaded GP models')
        except:
            print('GP model loading failed!')
            zt_list = torch.zeros(latent_dim, M, Q, dtype=torch.double).to(device)
            for i in range(latent_dim):
                zt_list_idx = torch.randperm(N, device=train_x.device)[:M]  
                zt_list[i] = train_x[zt_list_idx].clone().detach() 

            pass



        try:
            m = torch.load(gp_model_folder+'/m'+str(cl_step)+'.pth', map_location=torch.device(device)).detach()
            H = torch.load(gp_model_folder+'/H'+str(cl_step)+'.pth', map_location=torch.device(device)).detach()
            print('Loaded natural gradient values')
        except:
            print('Loading natural gradient values failed!')
            m = torch.randn(latent_dim, M, 1).double().to(device).detach()
            H = (torch.randn(latent_dim, M, M)/10).double().to(device).detach()

            H = torch.matmul(H, H.transpose(-1, -2)).detach().requires_grad_(False)
            pass

        
        #LVAE Test 
        list_all_Z_pred = []
        global_input_Z = []
        global_input_label = []
        for i in range(len(csv_file_prediction_data)):
            print('For : ', domain_test_name_list[i])
            #file name to save Z_pred 
            file_name_Z='Z_pred_label'+str(domain_test_name_list[i]).replace(' ', '')+str(version_run)
            #Test avec LVAE 
            Z_pred_all, label_all = test_LVAE(i, nnet_model, covar_module0, covar_module1, likelihoods, zt_list, Q, file_name_Z=file_name_Z)
            list_all_Z_pred.append(Z_pred_all)
            global_input_Z.append((domain_test_name_list[i], Z_pred_all))
            global_input_label.append((domain_test_name_list[i],label_all))


        if model_type == 'mem' : 
            print('\nPredictor : Linear mixed effect model\n')
            if mem_path != '' : 
                with open(mem_path, "rb") as f:
                    mem_model = pickle.load(f)
                print('MEM loaded.')
            else : 
                Z_pred_train, label_train = test_LVAE(-1, nnet_model, covar_module0, covar_module1, likelihoods, zt_list, Q)
                #Train and save
                mem_model = mem_model_train(latent_dim, col_target, mem_formula_cols, Z_pred_train, mem_cols_df, label_train) 
                with open(os.path.join(results_path,"mem_model"+str(version_run)+".pkl"), "wb") as f:
                    pickle.dump(mem_model, f)
                print('MEM training finished.')

            #MEM Test 
            print('Test for MEM')
            df_result_mem = mem_test(global_input_Z, latent_dim, mem_cols_df, global_input_label, mem_model, col_target) 
            print(df_result_mem)

        elif model_type == 'mlp':
            print('\nPredictor : MLP non longitudinal\n')
            #Load or training MLP
            Z_pred_train, label_train = test_LVAE(-1, nnet_model, covar_module0, covar_module1, likelihoods, zt_list, Q)

            latent_dim_total = Z_pred_train.shape[1] + len(mlp_cols_df)  # +1 pour time_age

            mlp_model = nn.Sequential(
                nn.Linear(latent_dim_total, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ).double().to(device)
            mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)

            if mlp_path != '' : 
                mlp_model.load_state_dict(torch.load(mlp_path, map_location=torch.device('cpu')))
                mlp_optimizer.load_state_dict(torch.load(mlp_optimizer_path, map_location=torch.device('cpu')))
                print('MLP loaded.')

            mlp_model, mlp_optimizer = mlp_train(mlp_model, mlp_optimizer, Z_pred_train, mlp_cols_df, label_train)

            torch.save(mlp_model.state_dict(), results_path+"/mlp_model"+str(version_run)+".pkl")
            torch.save(mlp_optimizer.state_dict(), results_path+"/mlp_optimizer"+str(version_run)+".pkl")
            

            #MLP 
            print('Test for MLP')
            df_result_mlp = mlp_test(mlp_model, col_target, global_input_Z, global_input_label, device)
            print(df_result_mlp)

    #nlme sans representation latente en entrée      
    elif model_type == 'nlme' :  
        print('\nPredictor : Non linear mixed effect model\n')
        model_info = {
            "linear": {
                "f": f_linear,
                "n_theta": 2
            },
            "quadratic": {
                "f": f_quadratic,
                "n_theta": 3
            },
            "exp": {
                "f": f_exp,
                "n_theta": 2
            },
            "log": {
                "f": f_logistic,
                "n_theta": 3
            },
            "basis_exp": {
                "f": f_basis_expansion,
                "n_theta": 4
            }
        }
        if bool_f_compare : 
            
            traces_f = {} 
            for filename in os.listdir(save_path):
                if filename.startswith(nlme_comparaison_name):
                    print('File found : ', filename, ' path : ', (save_path+'/'+filename))

                    with open(save_path+'/'+filename, "rb") as f:
                        [nlme_trace] = pickle.load(f) 
                    match  = re.search(re.escape(nlme_comparaison_name) + r"(.*?)" + '.pkl', filename, re.DOTALL)
                    name = match.group(1)
                    traces_f[name] = {'trace': nlme_trace}
                    print('Trace of ',name, ' loaded.')
            try : 
                with open(save_path+'/'+nlme_comparaison_time, "rb") as f:
                    [time_info] = pickle.load(f) 
            except : 
                
                # normalisation
                time_raw = df_train[col_time].values
                time_mean = time_raw.mean()
                time_std = time_raw.std()
                time_info = {"mean": time_mean, "std": time_std} 
                
            if traces_f == {} :

                traces_f, time_info = compare_f(df_train, col_subject, col_target, col_time, model_info)
                for name in traces_f.keys():
                    nlme_trace = traces_f[name]['trace']
                    pd.to_pickle([nlme_trace], os.path.join(save_path, nlme_comparaison_name+str(name)+str(version_run)+'.pkl'))
                pd.to_pickle([time_info], os.path.join(save_path, nlme_comparaison_name+str(version_run)+'_time_info.pkl'))
            
            for name in traces_f.keys():
                nlme_trace = traces_f[name]['trace']

                #LVAE Test 
                df_test_all = []
                result_all = []
                for i in range(len(csv_file_prediction_data)):
                    print('For : ', domain_test_name_list[i])
                    csv_test_i = pd.read_csv(os.path.join(data_source_path,csv_file_prediction_label[i]), header=0)
                    csv_test_true_i = pd.read_csv(os.path.join(data_source_path,csv_file_generation_label[i]), header=0)

                    df_test_i = csv_test_i[[col_subject, col_target, col_time]]
                    df_test_true_i = csv_test_true_i[[col_subject, col_target, col_time]]

                    df_test_all.append(df_test_i)

                    #NLME test
                    try : 
                        result_test_i = nlme_test(model_info[name], df_test_i, df_test_true_i, nlme_trace, time_info, col_subject, col_target, col_time, 'nlme_plot_'+name+'_'+str(domain_test_name_list[i]).replace(' ', '')+str(version_run))       
                        print(result_test_i)
                        result_all.append(result_test_i)
                    except Exception as e : 
                        raise e

        else : 
            if nlme_trace_path is not None: 
                with open(save_path+'/'+nlme_trace_path, "rb") as f:
                        [nlme_trace] = pickle.load(f)

            else :
                mini_model = {name_model_choosen: model_info.get(name_model_choosen)}
                traces_f, time_info = nlme_train_comparison(df_train, col_subject, col_target, col_time, mini_model, nb_echantillon=1000)
                nlme_trace = traces_f[name_model_choosen]['trace']
                pd.to_pickle([nlme_trace], os.path.join(save_path, 'nlme_model_'+name_model_choosen+str(version_run)+'train.pkl'))

        
            #LVAE Test 
            df_test_all = []
            result_all = []
            for i in range(len(csv_file_prediction_data)):
                print('For : ', domain_test_name_list[i])
                csv_test_i = pd.read_csv(os.path.join(data_source_path,csv_file_prediction_label[i]), header=0)
                csv_test_true_i = pd.read_csv(os.path.join(data_source_path,csv_file_generation_label[i]), header=0)

                df_test_i = csv_test_i[[col_subject, col_target, col_time]]
                df_test_true_i = csv_test_true_i[[col_subject, col_target, col_time]]

                df_test_all.append(df_test_i)


                #NLME test
                try : 
                    result_test_i = nlme_test(model_info[name_model_choosen], df_test_i, df_test_true_i, nlme_trace, time_info, col_subject, col_target, col_time, 'nlme_plot_'+name_model_choosen+'_'+str(domain_test_name_list[i]).replace(' ', '')+str(version_run))       
                    print(result_test_i)
                    result_all.append(result_test_i)
                except Exception as e : 
                    raise e