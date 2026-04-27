from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(project_root)

print(project_root)
import domHealth_MNIST_generate 
# from dil_replay.synthetic_dataset import domHealth_MNIST_generate 

class PhysionetDataset(Dataset):
    """
    Dataset definition for the Physionet Challenge 2012 dataset.
    """

    def __init__(self, data_file, root_dir, transform=None):
        data = np.load(os.path.join(root_dir, data_file))
        self.data_source = data['data_readings'].reshape(-1, data['data_readings'].shape[-1])
        self.label_source = data['outcome_attrib'].reshape(-1, data['outcome_attrib'].shape[-1])
        self.mask_source = data['data_mask'].reshape(-1, data['data_mask'].shape[-1])
        self.label_mask_source = data['outcome_mask'].reshape(-1, data['outcome_mask'].shape[-1])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient_data = self.data_source[idx, :]
        patient_data = torch.Tensor(np.array(patient_data))

        mask = self.mask_source[idx, :]
        mask = np.array(mask, dtype='uint8')

        label = self.label_source[idx, :]
        label[8] = label[8] - 24
        label_mask = self.label_mask_source[idx, :]

        label = torch.Tensor(np.concatenate((label, label_mask)))

        if self.transform:
            patient_data = self.transform(patient_data)

        sample = {'data': patient_data, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class RotatedMNISTDataset(Dataset):
    """
    Dataset definition for the rotated MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 784.
    """

    def __init__(self, data_file, label_file, root_dir, mask_file=None, transform=None):

        data = np.load(os.path.join(root_dir, data_file))
        label = np.load(os.path.join(root_dir, label_file))
        self.data_source = data.reshape(-1, data.shape[-1])
        self.label_source = label.reshape(label.shape[0], -1).T
        if mask_file is not None:
            self.mask_source = np.load(os.path.join(root_dir, mask_file))
        else:
            self.mask_source = np.ones_like(self.data_source)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source[idx, :]
        digit = np.array([digit])

        mask = self.mask_source[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source[idx, :]
        label = torch.Tensor(np.array(label))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class RotatedMNISTDatasetConv(Dataset):
    """
    Dataset definiton for the rotated MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 28 x 28.
    """

    def __init__(self, data_file, label_file, root_dir, mask_file=None, transform=None):

        data = np.load(os.path.join(root_dir, data_file))
        label = np.load(os.path.join(root_dir, label_file))
        self.data_source = data.reshape(-1, data.shape[-1])
        self.label_source = label.reshape(label.shape[0], -1).T
        if mask_file is not None:
            self.mask_source = np.load(os.path.join(root_dir, mask_file))
        else:
            self.mask_source = np.ones_like(self.data_source)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source[idx, :]
        digit = np.array(digit)
        digit = digit.reshape(28, 28)
        digit = digit[..., np.newaxis]

        mask = self.mask_source[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source[idx, :]
        label = torch.Tensor(np.array(label))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class HealthMNISTDataset(Dataset):
    """
    Dataset definition for the Health MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 1296.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None):

        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source.iloc[idx, :]
        digit = np.array([digit], dtype='uint8')

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source.iloc[idx, :]
        # changed
        # time_age,  disease_time,  subject,  gender,  disease,  location
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([6, 4, 0, 5, 3, 7])])))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample

class HealthMNISTDatasetConv(Dataset):
    """
    Dataset definiton for the Health MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 36 x 36.
    """

    # Si on veut donner des dataframe au lieu de donner des path 
    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None, bool_original=True, df_data = None, df_mask = None, df_label = None, val_dataset_type = None):
        
        self.bool_original = bool_original
        self.val_dataset_type = val_dataset_type

        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=0)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=0)
        # self.data_source = pd.DataFrame(self.data_source['data'].tolist())
        # self.mask_source = pd.DataFrame(self.mask_source['mask'].tolist())

        # if csv_file_data == None : 
        #     self.data_source = df_data
        #     self.label_source = df_label 
        #     self.mask_source = df_mask

        # else : 

        #     self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        #     print(('len(self.label_source)', len(self.label_source)))
        #     if val_dataset_type != None and 'dataset_type' in self.label_source.columns : 
        #         self.label_source = self.label_source[self.label_source['dataset_type'] == val_dataset_type]

        #     if 'dataset_type' in self.label_source.columns : 
        #         self.label_source = self.label_source.drop(columns=['dataset_type'])

        #     if self.bool_original : 
        #         self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        #         self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        #     else : 
        #         #data 
        #         data_source = pd.read_csv(os.path.join(root_dir, csv_file_data))
        #         data_source = domHealth_MNIST_generate.df_delete_rows(data_source,'data')
        #         self.data_source = data_source
        #         self.data_source['data'] = data_source['data'].apply(lambda x: domHealth_MNIST_generate.strList_to_numberList(x))

        #         #mask
        #         mask_source = pd.read_csv(os.path.join(root_dir, mask_file))
        #         mask_source = domHealth_MNIST_generate.df_delete_rows(mask_source,'mask')
        #         self.mask_source = mask_source
        #         self.mask_source['mask'] = mask_source['mask'].apply(lambda x: domHealth_MNIST_generate.strList_to_numberList(x))

        #         if val_dataset_type != None : 
        #             self.data_source = self.data_source.loc[self.label_source.index.tolist()]
        #             self.mask_source = self.mask_source.loc[self.label_source.index.tolist()]

        #         self.data_source = pd.DataFrame(self.data_source['data'].tolist())
        #         self.mask_source = pd.DataFrame(self.mask_source['mask'].tolist())
        

        # print((len(self.data_source), len(self.label_source), len(self.mask_source)))
        self.root_dir = root_dir
        self.transform = transform

    
    def __len__(self):
        print('(len(self.data), len(self.labels), len(self.mask_df))')
        print((len(self.data_source), len(self.label_source), len(self.mask_source)))
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
    #     print("getitem:", idx,
    #   len(self.data_source),
    #   len(self.label_source),
    #   len(self.mask_source))

        # if self.bool_original : 
        #     digit = self.data_source.iloc[idx, :]
        #     mask = self.mask_source.iloc[idx, :]
        # else : 
        #     digit = self.data_source['data'].iloc[idx]
        #     mask = self.mask_source['mask'].iloc[idx]
        
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
    

# class HealthMNISTDatasetConv(Dataset):
#     """
#     Dataset definiton for the Health MNIST dataset when using CNN-based VAE.

#     Data formatted as dataset_length x 36 x 36.
#     """

#     def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None):

#         self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
#         self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
#         self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_source)

#     def __getitem__(self, key):
#         if isinstance(key, slice):
#             start, stop, step = key.indices(len(self))
#             return [self.get_item(i) for i in range(start, stop, step)] 
#         elif isinstance(key, int):
#             return self.get_item(key)
#         else:
#             raise TypeError

#     def get_item(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         digit = self.data_source.iloc[idx, :]
#         digit = np.array(digit, dtype='uint8')
#         digit = digit.reshape(36, 36)
#         digit = digit[..., np.newaxis]

#         mask = self.mask_source.iloc[idx, :]
#         mask = np.array([mask], dtype='uint8')

#         label = self.label_source.iloc[idx, :]
#         # CHANGED
#         # time_age,  disease_time,  subject,  gender,  disease,  location
#         label = torch.Tensor(np.nan_to_num(np.array(label[np.array([6, 4, 0, 5, 3, 7])])))

#         if self.transform:
#             digit = self.transform(digit)

#         sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
#         return sample
