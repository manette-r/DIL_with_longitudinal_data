import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import gpytorch
import pickle
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import argparse
import ast
from GP_def import ExactGPModel
from VAE import ConvVAE, SimpleVAE
from dataset_def import HealthMNISTDatasetConv, RotatedMNISTDatasetConv, HealthMNISTDataset, RotatedMNISTDataset, \
    PhysionetDataset
from elbo_functions import elbo, KL_closed, deviance_upper_bound
from kernel_gen import generate_kernel, generate_kernel_approx, generate_kernel_batched
from model_test import MSE_test
from training import hensman_training, minibatch_training, standard_training, variational_inference_optimization
from validation import validate
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
        self.parser.add_argument('--csv_file_prediction_data', type=str, help='Name of prediction data file', required=False)
        self.parser.add_argument('--csv_file_prediction_label', type=str, help='Name of prediction label file', required=False)
        self.parser.add_argument('--csv_file_validation_data', type=str, help='Name of validation data file', required=False)
        self.parser.add_argument('--csv_file_validation_label', type=str, help='Name of validation label file', required=False)
        self.parser.add_argument('--csv_file_generation_data', type=str, help='Name of data file for image generation', required=False)
        self.parser.add_argument('--csv_file_generation_label', type=str, help='Name of label file for image generation', required=False)
        self.parser.add_argument('--mask_file', type=str, help='Name of mask file', default=None)
        self.parser.add_argument('--test_mask_file', type=str, help='Name of test mask file', default=None)
        self.parser.add_argument('--prediction_mask_file', type=str, help='Name of prediction mask file', default=None)
        self.parser.add_argument('--validation_mask_file', type=str, help='Name of validation mask file', default=None)
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
        self.parser.add_argument('--first_t', type=int, default=0, help='xx')
        self.parser.add_argument('--t_steps', type=int, default=0, help='xx')
        self.parser.add_argument('--csv_file_data_memory', type=str, help='Path to save data', required=False)
        self.parser.add_argument('--csv_file_label_memory', type=str, help='Path to save data', required=False)
        self.parser.add_argument('--csv_file_mask_memory', type=str, help='Path to save data', required=False)
        self.parser.add_argument('--cl_step', type=int, default=-1, help='Continual learning step to indicate which version of object to load, if -1 load in the normal way for plot_values, -2 plot_values without prediction_mu')
        


    def parse_options(self):
        opt = vars(self.parser.parse_args())
        return opt
    

#import VAE déjà entrainés

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
                       prediction_mu, epoch, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to generate rotated MNIST digits.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Generating images - length of dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=4)
    total_mse = 0
    total_samples = 0
    all_mse = []

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
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
            all_mse.append(mse.detach().cpu())
            # reconstruction loss
            print('160 == ',data.shape[0], 'and 360 == ', recon_Z.shape[0])

            seq_len = int(torch.max(label[:,0]).item()) + 1
            num_sets = min(8, data.shape[0] // seq_len)
            max_display = num_sets * seq_len
            gen_rotated_mnist_seqrecon_plot(data[0:max_display, :].cpu(), recon_Z[0:max_display, :].cpu(), label[0:max_display, :].cpu(), label[0:max_display, :].cpu(),
                                            save_file=os.path.join(results_path, filename))
        print("Pour ", total_samples, "samples ")
        all_mse = torch.cat(all_mse)
        mean_mse = torch.mean(all_mse)
        std_mse = torch.std(all_mse)
        stderr_mse = std_mse / torch.sqrt(torch.tensor(len(all_mse), dtype=torch.float))
        mse_mean = (total_mse/total_samples)
        print("MSE GP = ",mse_mean)
        print("MSE GP2 = ",mean_mse, "Std MSE = ", std_mse, "Std Error = ", stderr_mse)
        return mse_mean
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
            test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                                  csv_file_label=csv_file_test_label,
                                                  mask_file=test_mask_file, root_dir=data_source_path,
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

            print('label\n', label, 'data', len(data))

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
            # np.savetxt(os.path.join(results_path, save_file), pred_results)


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

    dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_data,
                                                csv_file_label=csv_file_label,
                                                mask_file=mask_file,
                                                root_dir=data_source_path,
                                                transform=transforms.ToTensor(), bool_original=True, val_dataset_type='dataset')
    
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

        # initialise inducing points
        zt_list = torch.zeros(latent_dim, M, Q, dtype=torch.double).to(device)
        for i in range(latent_dim):
            zt_list_idx = torch.randperm(train_x.shape[0], device=train_x.device)[:M]  
            zt_list[i] = train_x[zt_list_idx].clone().detach() 

        pass

    assert train_x.shape[1] == Q, "Mismatch covariates dimension"
    assert zt_list.shape[1] == M, "Mismatch inducing points M"


    m = torch.randn(latent_dim, M, 1).double().to(device).detach()
    H = (torch.randn(latent_dim, M, M)/10).double().to(device).detach()

    H = torch.matmul(H, H.transpose(-1, -2)).detach().requires_grad_(False)

    try:
        m = torch.load(gp_model_folder+'/m'+str(cl_step)+'.pth', map_location=torch.device(device)).detach()
        H = torch.load(gp_model_folder+'/H'+str(cl_step)+'.pth', map_location=torch.device(device)).detach()
        print('Loaded natural gradient values')
    except:
        print('Loading natural gradient values failed!')
        pass


    #Test avec code de LVAE 

    prediction_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_prediction_data,
                                                csv_file_label=csv_file_prediction_label,
                                                mask_file=prediction_mask_file,
                                                root_dir=data_source_path,
                                                transform=transforms.ToTensor(), bool_original=True, val_dataset_type='prediction')
    
    generation_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_generation_data,
                                                csv_file_label=csv_file_generation_label,
                                                mask_file=generation_mask_file,
                                                root_dir=data_source_path,
                                                transform=transforms.ToTensor(), bool_original=True, val_dataset_type='test')

    prediction_dataloader = DataLoader(prediction_dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(prediction_dataset, id_covariate), subjects_per_batch), num_workers=4)
    full_mu = torch.zeros(len(prediction_dataset), latent_dim, dtype=torch.double).to(device)
    prediction_x = torch.zeros(len(prediction_dataset), Q, dtype=torch.double).to(device)

    print("train_x shape:", train_x.shape)
    print("prediction_x shape:", prediction_x.shape)
    print("zt_list shape:", zt_list.shape)
    
    with torch.no_grad():
        print("Test première boucle")
        for batch_idx, sample_batched in enumerate(prediction_dataloader):
            label_id = sample_batched['idx']
            prediction_x[label_id] = sample_batched['label'].double().to(device)
            data = sample_batched['digit'].double().to(device)
            covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate+1:]), dim=1)

            # print("label",prediction_x[label_id], "data", len(data))

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
        mse_tt = recon_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, full_mu, -1, zt_list, P, T, id_covariate, varying_T)
        




