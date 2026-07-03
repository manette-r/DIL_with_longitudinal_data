# Domain incremental learning with longitudinal data
---

This repository contains Python scripts developed to conduct initial experiments on the DomHealth MNIST dataset, demonstrating that continual learning methods outperform non-continual baselines.

## Overview 
---

In this repository, you can find a Longitudinal-VAE* updated and its continual learning version, as well as regression model (a non-longitudinal MLP, Linear and Non-linear Mixed effects model).   

* originally from *Ramchandran, S., Tikhonov, G., Kujanpää, K., Koskinen, M., & Lähdesmäki, H. (2021). Longitudinal Variational Autoencoder. Proceedings of the Twenty Fourth International Conference on Artificial Intelligence and Statistics (AISTATS)*


## AE_predictor file 

It requires the same arguments as LVAE.py and some additional specific arguments.
```cl_step``` indicates which version of continual learning to load.
To apply naïve replay continual learning, you can load the memory with ```csv_file_data_memory```, please specify the generic file name (excluding the cl_step and csv extensions).
To apply regularization continual learning methods, you must specify it in ```regularization```. Arguments related to EWC (Elastic Weight Consolidation) method are ```lambda_ewc``` and ```old_fisher_path```. For LwF (Learning without forgetting) method, the only argument is ```lambda_lwf```. 

For the regression model, ```col_target``` specifies the name of the target column, and ```domain_test_name_list``` is used to display plots and save data.
The arguments depend on the regression model used. You can specify the model type using a string (mlp, nlme or mem) in ```model_type```. 

For an MLP model, if an MLP model needs to be loaded, fill in the ```mlp_path``` and ```mlp_optimizer_path```.

For a non-linear mixed effects model, the arguments are:
- ```nlme_trace_path```: the path for pre-trained nlme
- ```name_model_chosen```: a string of functions to apply to nlme (quadratic, exp, log or basis_exp)

For a linear mixed effects model, the arguments are:
- ```mem_formula_cols```: a string of columns to add to the formula
- ```mem_cols_df```: a list of label columns for the target and formula columns
- ```mem_path```: the path for pre-trained mem

## LVAE_test file 


It requires the same arguments as LVAE.py and the ```cl_step``` argument indicates the version of continual learning to be loaded. If there is no continual learning, it indicates how many values are in the plot values list (-1 for 5 values and -2 for 4 values).


## experience_replay_LVAE file 

It requires the same arguments as LVAE.py and some additional specific arguments. Some of these relate to the number of steps: 
- ```first_t```: the number of the first step in the learning process
- ```t_steps```: the total number of steps
- ```num_past_step```: the number of previous steps to load the model
If a previous step is loaded, set ```len_plot_values``` = 4.
Regarding memory, ```n_subject_memory``` is the number of patients to be stored in memory.
If previous memory data needs to be loaded, use ```csv_file_data_memory```, ```csv_file_label_memory``` and ```csv_file_mask_memory```.
