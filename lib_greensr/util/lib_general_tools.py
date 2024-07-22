#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:47:21 2021

@author: wei
"""

# include old lib_statistics, part of lib_spatial_correlation, part of lib_tool_box

import pickle
import torch
from torch.nn import functional as f
import numpy as np
from scipy.spatial.distance import cdist
import os
import gc
from PIL import Image

# import lib.lib_saab_channel_wise_v5_realbias_updated as lib_saab_cw


# SPATIAL CROPPING, TRANSFER, ETC.
# STATISTICS OPERATION / ANALYSIS
# SOME VISUALIZATIONS


#%%

def get_dist_to_ref(ref, samples):
    dist_samples_to_ref = \
        np.sqrt(
            np.sum(
                np.power(samples - ref, 2),
                axis=-1)
            )
    return dist_samples_to_ref




#%%
# data = np.array([[1,0],[3,4]])
# a = data
# b = data[1]
# a *= 3
# # a = a * 3
# b = b * 4
# print(data)

#%%
# FOLDER / FILE RELATED

def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


# def parallel_by_samples(total_sample_mat):
#     cumu_sample_num = get_cumu_sample_num_batchwise(len(input_patch), batch_size=10000)
#     print('SO: cumu_sample_num', len(cumu_sample_num), cumu_sample_num[:5])
#     results = Parallel(n_jobs=NUM_PARALLEL)\
#         (delayed(self.multiring_patchalign_feature_extraction)\
#          (n,
#           input_patch[i:j],
#           saab_foldername,
#           rft_foldername,
#           setting_dirc['simple_saab_feat_setting'],
#           setting_dirc['neighbor_saab_feat_setting'],
#           setting_dirc['pixel_neighbor_setting'],
#           aligning_flag=setting_dirc['align_flag']) for i,j in zip(cumu_sample_num[:-1], cumu_sample_num[1:]))
#     input_feat_reg = np.concatenate(results, axis=0)
#     print('SO: aft saab, rft (chl selection & spatial selection): input_aug_feat_reg',
#           input_feat_reg.shape)
#     print()


from joblib import Parallel, delayed
def parallel_by_samples(batchsize, num_parallel,
                        functions, samples, **kwargs):
    # print('lib_gtools.parallel_by_samples, kwargs', kwargs)
    cumu_sample_num = get_cumu_sample_num_batchwise(len(samples),
                                                    batch_size=batchsize)
    results = Parallel(n_jobs=num_parallel)\
            (delayed(functions)\
              (samples[i:j], kwargs) for i,j in zip(cumu_sample_num[:-1],
                                                    cumu_sample_num[1:]))
    results = np.concatenate(results, axis=0)
    gc.collect()
    return results










#%% numpy bool array operation

def np_and_list(bool_list):
    bool_ind_final = np.logical_and(bool_list[0], bool_list[1])
    if len(bool_list) >= 3:
        for bool_ind in bool_list[2:]:
            bool_ind_final = np.logical_and(bool_ind_final, bool_ind)
    return bool_ind_final

def np_or_list(bool_list):
    bool_ind_final = np.logical_or(bool_list[0], bool_list[1])
    if len(bool_list) >= 3:
        for bool_ind in bool_list[2:]:
            bool_ind_final = np.logical_or(bool_ind_final, bool_ind)
    return bool_ind_final

def np_xor_list(bool_list):
    bool_ind_final = np.logical_xor(bool_list[0], bool_list[1])
    if len(bool_list) >= 3:
        for bool_ind in bool_list[2:]:
            bool_ind_final = np.logical_xor(bool_ind_final, bool_ind)
    return bool_ind_final

def np_xor_2list_raw(bool_list):
    '''
    the 2nd list has more elements then the 1st list
    '''
    assert set(bool_list[0]).issubset(set(bool_list[1]))
    return np.array(list(set(bool_list[1]) - set(bool_list[0])))



#%% LOADING




def obj_saving(file_name, obj):
    if file_name[-3:] == 'pkl':
        with open(file_name, 'wb') as fw:
            pickle.dump(obj, fw, protocol=4)
    elif file_name[-3:] == 'npy':
        np.save(file_name, obj)

def obj_loading(file_name):
    if file_name[-3:] == 'pkl':
        with open(file_name, 'rb') as fw:
            obj = pickle.load(fw)
    elif file_name[-3:] == 'npy':
        obj = np.load(file_name, allow_pickle=True)
    return obj


def img_loading(file_name, mode='npy'):
    img = Image.open(file_name)
    if mode == 'npy':
        img = np.array(img)
    return img




# # specific load, img, train, valid
# def get_hr_train_all(data_direct):
#     with open(data_direct + 'training_HR/RESULT_training_images_hr_bsd200.pkl', 'rb') as fr:
#         hr_img_4d_train_all = pickle.load(fr)
#     print('hr_img_4d_train_all', hr_img_4d_train_all.shape)
#     return hr_img_4d_train_all

# def get_hr_train_valid(data_direct):
#     train_img_idx, valid_img_idx = get_sample_idx_train_valid(data_direct)
#     hr_img_4d = get_hr_train_all(data_direct)
#     train_hr_img_4d = hr_img_4d[train_img_idx]
#     valid_hr_img_4d = hr_img_4d[valid_img_idx]
#     print('train_hr_img_4d valid_hr_img_4d', train_hr_img_4d.shape, valid_hr_img_4d.shape)
#     return train_hr_img_4d, valid_hr_img_4d

# def get_ilr_train_all(data_direct, ilr_type_flag=''):
#     if ilr_type_flag == 'lanc':
#         with open(data_direct + 'training_ILR_2_lanczos/RESULT_training_images_ilr_lanczos_bsd200.pkl', 'rb') as fr:
#             ilr_img_4d_train_all = pickle.load(fr)
#     else:
#         with open(data_direct + 'training_ILR_2/RESULT_training_images_ilr_bsd200.pkl', 'rb') as fr:
#             ilr_img_4d_train_all = pickle.load(fr)
#     print('ilr_img_4d_train_all', ilr_img_4d_train_all.shape)
#     return ilr_img_4d_train_all

# def get_ilr_train_valid(data_direct, ilr_type=''):
#     train_img_idx, valid_img_idx = get_sample_idx_train_valid(data_direct)
#     ilr_img_4d = get_ilr_train_all(data_direct, ilr_type)
#     train_ilr_img_4d = ilr_img_4d[train_img_idx]
#     valid_ilr_img_4d = ilr_img_4d[valid_img_idx]
#     print('train_ilr_img_4d valid_ilr_img_4d', train_ilr_img_4d.shape, valid_ilr_img_4d.shape)
#     return train_ilr_img_4d, valid_ilr_img_4d


# load, sample_idx, train, valid
def get_sample_idx_train_valid(data_direc):
    train_img_idx = np.load(data_direc + 'training_validation_split/train_img_idx_bsd200.npy', allow_pickle=True)
    valid_img_idx = np.load(data_direc + 'training_validation_split/valid_img_idx_bsd200.npy', allow_pickle=True)
    return train_img_idx, valid_img_idx






def load_train_ilr_img_clct(loc='hpc'):
    if loc == 'server':
        main_data_folder = '/mnt/wei_gpu/SR_Denoising/DATASET/local/'
    elif loc== 'local':
        main_data_folder = '/media/wei/Data/SR/DATASET/'
    elif loc == 'hpc':
        main_data_folder = '/project/jckuo_84/weiwang/DATASETS/SR/'
    return obj_loading(main_data_folder + 'training_2/training_ILR_lanczos/RESULT_training_images_ilr_bsd200_train.pkl')

def load_test_ilr_img_clct(testing_dataset, loc='hpc'):
    if loc == 'server':
        main_data_folder = '/mnt/wei_gpu/SR_Denoising/DATASET/'
    elif loc== 'local':
        main_data_folder = '/media/wei/Data/SR/DATASET/' 
    elif loc == 'hpc':
        main_data_folder = '/project/jckuo_84/weiwang/DATASETS/SR/'
    return obj_loading(main_data_folder + 'testing_2/testing_ILR_lanczos/RESULT_testing_images_ilr_'+str(testing_dataset)+'.pkl')

def load_totaltrain_ilr_img_clct(loc='hpc'):
    if loc == 'server':
        main_data_folder = '/mnt/wei_gpu/SR_Denoising/DATASET/local/'
    elif loc== 'local':
        main_data_folder = '/media/wei/Data/SR/DATASET/'
    elif loc == 'hpc':
        main_data_folder = '/project/jckuo_84/weiwang/DATASETS/SR/'
    return obj_loading(main_data_folder + 'training_2/training_ILR_lanczos/RESULT_training_images_ilr_bsd200.pkl')







def load_train_hr_img_clct(loc='hpc'):
    if loc == 'server':
        main_data_folder = '/mnt/wei_gpu/SR_Denoising/DATASET/local/'
    elif loc== 'local':
        main_data_folder = '/media/wei/Data/SR/DATASET/'
    elif loc == 'hpc':
        main_data_folder = '/project/jckuo_84/weiwang/DATASETS/SR/'
    return obj_loading(main_data_folder + 'training_HR/RESULT_training_images_hr_bsd200_train.pkl')

def load_test_hr_img_clct(testing_dataset, loc='hpc'):
    # testing_dataset = 'bsd100'
    if loc == 'server':
        main_data_folder = '/mnt/wei_gpu/SR_Denoising/DATASET/'
    elif loc== 'local':
        main_data_folder = '/media/wei/Data/SR/DATASET/'
    elif loc == 'hpc':
        main_data_folder = '/project/jckuo_84/weiwang/DATASETS/SR/'
    return obj_loading(main_data_folder + 'testing_HR/RESULT_testing_images_hr_'+str(testing_dataset)+'.pkl')

def load_totaltrain_hr_img_clct(loc='hpc'):
    if loc == 'server':
        main_data_folder = '/mnt/wei_gpu/SR_Denoising/DATASET/local/'
    elif loc== 'local':
        main_data_folder = '/media/wei/Data/SR/DATASET/'
    elif loc == 'hpc':
        main_data_folder = '/project/jckuo_84/weiwang/DATASETS/SR/'
    return obj_loading(main_data_folder + 'training_HR/RESULT_training_images_hr_bsd200.pkl')







#%%
# BATCHLIZATION
def get_cumu_sample_num_batchwise(sample_num, batch_size=1):
    # cond_sample_idx = np.arange(sample_num)
    batch_num = int(np.ceil(sample_num *1.0 / batch_size))
    cumu_sample_num_batchwise = [i*batch_size for i in range(batch_num)]
    cumu_sample_num_batchwise.append(np.min([sample_num, batch_num * batch_size]))
    return np.array(cumu_sample_num_batchwise)

# cumu_sample_num_list = get_cumu_sample_num_batchwise(1001, batch_size=10000)
# print(cumu_sample_num_list)







#%% BLOCK MAT TRANSFER
################################################################################
# block manipulation
def mat4Dto2D_simple(Mat):
    '''
    Mat(np.array): [num, c, h, w]
    '''
    shape_ori = Mat.shape
    Mat_new = np.moveaxis(Mat, 1,3) #-> [num, h, w, c]
    Mat_new = Mat_new.reshape((-1, shape_ori[1])) #-> [num*h*w, c]
    return Mat_new

def mat4Dto2D(Mat):
    '''
    Mat(np.array): [num, c, h, w]
    '''
    shape_ori = Mat.shape
    Mat_new = np.moveaxis(Mat, 1,3) #-> [num, h, w, c]
    Mat_new = Mat_new.reshape((-1, shape_ori[1])) #-> [num*h*w, c]
    return Mat_new, shape_ori

def mat2Dto4D(Mat, shape_4d):
    '''
    @Arg:
        Mat(np.array): [num*h*w, c]
    @Returns:
        Mar(np.array): [num, c, h, w]
    '''
    Mat_new = Mat.reshape((-1, shape_4d[-2], shape_4d[-1], shape_4d[1])) #-> [num, h, w, c]
    assert Mat_new.shape[-1] == shape_4d[1]
    Mat_new = np.moveaxis(Mat_new, -1, 1) #-> [num, c, h, w]
    return Mat_new




#%% IMG --> PATCHES

# def window_process_3d_patch(img_4d, patch_size, stride, dilate, pad_flag):
#     feat_patch = lib_saab_cw.window_process4_ori(img_4d, patch_size, stride, dilate, padFlag=pad_flag)  # (n,output_h, output_w, c, kernel_size//dilate, kernel_size//dilate)
#     s = feat_patch.shape
#     patch_cropped = np.squeeze(feat_patch.reshape((s[0]*s[1]*s[2], s[3], s[4]*s[5])))
#     return patch_cropped

def window_process_4d_patch(img_4d, patch_size, stride, dilate, pad_flag):
    '''
    if pad_flag:
        padding the input img_4d 
    else:
        No padding
    torch.nn.functional.unfold generates patches not for each pixel, but within the whole image boarders

    Parameters
    ----------
    img_4d : TYPE
        DESCRIPTION.
    patch_size : TYPE
        DESCRIPTION.
    stride : TYPE
        DESCRIPTION.
    dilate : TYPE
        DESCRIPTION.
    pad_flag : TYPE
        DESCRIPTION.

    Returns
    -------
    patch_cropped : TYPE
        DESCRIPTION.

    '''
    # feat_patch = lib_saab_cw.window_process4_ori(img_4d, patch_size, stride, dilate, padFlag=pad_flag)  
    # (n, output_h, output_w, c, kernel_size//dilate, kernel_size//dilate)
    # s = feat_patch.shape
    # patch_cropped = feat_patch.reshape((s[0]*s[1]*s[2], s[3], s[4], s[5]))
    
    # (n*output_h*output_w, c, kernel_size//dilate, kernel_size//dilate)
    if pad_flag:
        padding_width = patch_size // 2
        img_4d_padded = f.pad(torch.from_numpy(img_4d), (padding_width, padding_width, padding_width, padding_width), 'reflect')
    else:
        img_4d_padded = torch.from_numpy(img_4d)
    patch_cropped = f.unfold(img_4d_padded, kernel_size=(patch_size, patch_size), stride=stride, dilation=dilate).permute(0, 2, 1).reshape(-1, img_4d.shape[1], patch_size, patch_size).numpy()
    # print(img_4d.shape, img_4d_padded.shape, patch_cropped.shape)
    return patch_cropped

# arr_shape = (2, 1, 325, 484)
# a = np.random.rand(np.prod(arr_shape)).reshape(arr_shape)
# p = window_process_4d_patch(a, 7, 3, 1, False)
# print(p.shape) # (342400, 1, 7, 7)
# print(a[0, 0, :5, :5])
# p = p * 100
# print(a[0, 0, :5, :5])


def window_process_4d_patch_wrapper(img_4d, patch_setting):
    return window_process_4d_patch(img_4d,
                            patch_setting['patch_size'],
                            patch_setting['stride'],
                            patch_setting['dilate'],
                            patch_setting['pad_flag'])

def window_process_4d_patch_batch_by_samples(img_4d, patch_setting):
    patch4d = \
        parallel_by_samples(20,
                    8,
                    window_process_4d_patch_wrapper,
                    img_4d,
                    **patch_setting)
    return patch4d






# def crop_selected_patches(sample_idx, patch_neighbor_setting, input_multiImg4d_or_oneImg4dlist_finalpadded):
#     '''
#     patch_neighbor_setting = {'sampling_size':7, 'sampling_stride':3, 'corepatch_for_autopadding':3}
#     '''
#     h_finalpadded_lists, w_finalpadded_lists = \
#         [i.shape[-2] for i in input_multiImg4d_or_oneImg4dlist_finalpadded], \
#         [i.shape[-1] for i in input_multiImg4d_or_oneImg4dlist_finalpadded]

#     if len(h_finalpadded_lists) == 1:
#         temp = h_finalpadded_lists[0]
#         h_finalpadded_lists = [temp for i in range(len(input_multiImg4d_or_oneImg4dlist_finalpadded[0]))]
#     if len(w_finalpadded_lists) == 1:
#         temp = w_finalpadded_lists[0]
#         w_finalpadded_lists = [temp for i in range(len(input_multiImg4d_or_oneImg4dlist_finalpadded[0]))]
#     print('h_finalpadded_lists', h_finalpadded_lists[:5])
#     print('w_finalpadded_lists', w_finalpadded_lists[:5])

#     orinum_patch_h = [(h - patch_neighbor_setting['sampling_size']) // patch_neighbor_setting['sampling_stride'] + 1 for h in h_finalpadded_lists]
#     orinum_patch_w = [(w - patch_neighbor_setting['sampling_size']) // patch_neighbor_setting['sampling_stride'] + 1 for w in w_finalpadded_lists]
#     print('orinum_patch_h', orinum_patch_h[:5])
#     print('orinum_patch_w', orinum_patch_w[:5])
#     # get original idx
#     sample_img_idx, sample_h_idx, sample_w_idx = \
#         transf_sampleidx_to_imghwidx(sample_idx, orinum_patch_h, orinum_patch_w)

#     # get original locations on the final-padded images
#     sample_h_1 = patch_neighbor_setting['sampling_stride'] * sample_h_idx # : (sample_h + patch_neighbor_setting['sampling_size'])
#     sample_h_2 = sample_h_1 + patch_neighbor_setting['sampling_size']
#     sample_w_1 = patch_neighbor_setting['sampling_stride'] * sample_w_idx # : (sample_w + patch_neighbor_setting['sampling_size'])
#     sample_w_2 = sample_w_1 + patch_neighbor_setting['sampling_size']

#     # crop out the larger patches
#     if len(input_multiImg4d_or_oneImg4dlist_finalpadded) > 1: # [(1, 1, 233, 233), (1, 1, 320, 450)]
#         larger_regions_list = \
#             [np.squeeze(input_multiImg4d_or_oneImg4dlist_finalpadded[img_idx])[h_1:h_2, w_1:w_2] \
#              for img_idx, h_1, h_2, w_1, w_2 in zip(sample_img_idx, sample_h_1, sample_h_2, sample_w_1, sample_w_2)]
#     else: # [(200, 1, 320, 480)] or [(1, 1, 233, 233)]
#         larger_regions_list = \
#             [np.squeeze(input_multiImg4d_or_oneImg4dlist_finalpadded[0][img_idx])[h_1:h_2, w_1:w_2] \
#              for img_idx, h_1, h_2, w_1, w_2 in zip(sample_img_idx, sample_h_1, sample_h_2, sample_w_1, sample_w_2)]
#     larger_regions_list = [i.reshape((1, 1, i.shape[-2], i.shape[-1])) for i in larger_regions_list]

#     selected_patches = np.concatenate(larger_regions_list, axis=0)
#     del larger_regions_list
#     gc.collect()
#     return selected_patches

# arr_shape = (2, 1, 325, 484)
# a = np.random.rand(np.prod(arr_shape)).reshape(arr_shape)
# print(a[0, 0, :5, :5])
# print()

# sample_id = np.arange(10)
# p = crop_selected_patches(sample_id,
#                          {'sampling_size':7,
#                           'sampling_stride':1,
#                           'corepatch_for_autopadding':1},
#                          a)
# print()

# print(p[0, 0, :5, :5])
# print()

# p = p*10

# print(a[0, 0, :5, :5])
# print()

# print(p[0, 0, :5, :5])
# print()



#%%


# def window_process_4d_img(img_4d, patch_size, stride, dilate, pad_flag):
#     feat_patch = lib_saab_cw.window_process4_ori(img_4d, patch_size, stride, dilate, padFlag=pad_flag)  
#     # (n, output_h, output_w, c, kernel_size//dilate, kernel_size//dilate)
#     feat_patch_shape = feat_patch.shape
#     feat_patch = feat_patch.reshape((feat_patch_shape[0], feat_patch_shape[1], feat_patch_shape[2], -1))
#     new_img_4d = np.moveaxis(feat_patch, -1, 1)
#     return new_img_4d





# original from: part_redesign_pixeldomain_var_and_cox_cleanversion_removeDC_variant_LR_v3_functions
def images_to_patches_multiobj(obj_list):
    '''
    Padding and selection for each element 

    Parameters
    ----------
    obj_list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
#def helper_crop_patch_multiimg(obj_list):
#    obj_list: a list of [img_4d, patch_border, patch_size, patch_stride, padding_flag, selected_patch_idx]
    
#    for img_4d, patch_border, patch_size, patch_stride in obj_list:
#        patch_4d = lib_gtools.window_process_4d_patch(np.pad(img_4d, ((0,0), (0,0), (patch_border, patch_border), (patch_border, patch_border)), 'reflect'), 
#                                                         patch_size, patch_stride, 1, False)
#    selected_patch_list = [(lib_gtools.window_process_4d_patch(np.pad(img_4d, ((0,0), (0,0), (patch_border, patch_border), (patch_border, patch_border)), 'reflect'), \
#                                                       patch_size, patch_stride, 1, False))[selected_patch_idx] \
#                           for img_4d, patch_border, patch_size, patch_stride, selected_patch_idx in obj_list]
    # print('crop img to patch')
    # print('img ready for cropping', np.pad((obj_list[0])[0], ((0,0), (0,0), ((obj_list[0])[1], (obj_list[0])[1]), ((obj_list[0])[1], (obj_list[0])[1])), 'reflect').shape)
    # print('cropping patch_size, patch_stride', (obj_list[0])[2], (obj_list[0])[3])
    selected_patch_list = [(window_process_4d_patch(np.pad(img_4d, ((0,0), (0,0), (patch_border, patch_border), (patch_border, patch_border)), 'reflect'), \
                            patch_size, patch_stride, 1, padding_flag)) if selected_patch_idx == 'full' else \
                            (window_process_4d_patch(np.pad(img_4d, ((0,0), (0,0), (patch_border, patch_border), (patch_border, patch_border)), 'reflect'), \
                            patch_size, patch_stride, 1, padding_flag))[selected_patch_idx] \
                           for img_4d, patch_border, patch_size, patch_stride, padding_flag, selected_patch_idx in obj_list]
    
    print([i.shape for i in selected_patch_list])
    return selected_patch_list


#%% PATCHES --> IMG
# original from: partition_with_xgbreg_haar_iter_multipatchsize_alldata_functions
# same as get_feat_arr4dcollect
def img4dlist_to_patch4dlist(feat_img4d_list, sampling_setting, padding_flag=False):
    '''
    work for single channel, each elemetn in feat_img_singchl_set is [n, 1, h, w]
    can work for different img_arr with different n, h, w but the same corepatch_size
    generate 2-D input and output feat for reg.

    Parameters
    ----------
    feat_img4d_list : TYPE
        DESCRIPTION.
    sampling_setting : TYPE
        DESCRIPTION.

    Returns
    -------
    feat_patch4d_list: a list of 4d patches

    '''

    # feat_patch4d_list = \
    # images_to_patches_multiobj([[i,
    #                              sampling_setting['sampling_size']//2,
    #                              sampling_setting['sampling_size'],
    #                              sampling_setting['sampling_stride']//sampling_setting['corepatch_stride'],
    #                              padding_flag, 'full'] for i in feat_img4d_list])#,
    # #[img_4d, patch_border, patch_size, patch_stride, selected_patch_idx]

    feat_patch4d_list = [window_process_4d_patch(i,
                                                 sampling_setting['sampling_size'],
                                                 sampling_setting['sampling_stride']//sampling_setting['corepatch_stride'],
                                                 1,
                                                 padding_flag)\
                         for i in feat_img4d_list]
    print([i.shape for i in feat_patch4d_list])
    return feat_patch4d_list



#%% PADDING, IMG --> PATCHES with PADDING
def get_padding_info_two_ends(border_length, patch_size, stride):
    padded_boarder_length = int(np.ceil((border_length - patch_size) * 1.0 / stride) * stride + patch_size)
    total_padding_pixels = padded_boarder_length - border_length
    padding_left = total_padding_pixels // 2
    padding_right = total_padding_pixels - padding_left
    # print(padding_left, padding_right)
    return padding_left, padding_right

def get_img4d_arr_padding_width(img4d_arr, patch_size, stride):
    h_p_lr = get_padding_info_two_ends(img4d_arr.shape[-2], patch_size, stride)
    w_p_lr = get_padding_info_two_ends(img4d_arr.shape[-1], patch_size, stride)
    return h_p_lr, w_p_lr

def autopad_img4d_arr(img4d_arr, patch_size, stride):
    h_p_lr, w_p_lr = get_img4d_arr_padding_width(img4d_arr, patch_size, stride)
    return np.pad(img4d_arr, ((0,0), (0,0), h_p_lr, w_p_lr), 'reflect')

def autopad_img4dlist_arr(img4d_arr_list, patch_size, stride):
    return [autopad_img4d_arr(i, patch_size, stride) for i in img4d_arr_list]

def autopad_img4dgroup_arr(img4d_arr_group, patch_size, stride):
    if isinstance(img4d_arr_group, np.ndarray):
        return autopad_img4d_arr(img4d_arr_group, patch_size, stride)
    else:
        return autopad_img4dlist_arr(img4d_arr_group, patch_size, stride)





def additionalpad_img4d_arr(img4d_arr, core_patch_size, sampling_patch_size, stride):
    # check if need additional padding
    p = int((sampling_patch_size - core_patch_size) / 2.0)
    print('additional padding p', p)
    if p > 0:
        img4d_arr_padded = np.pad(img4d_arr, ((0,0), (0,0), (p,p), (p,p)), 'reflect')
        # print('additional padding', [i.shape for i in img4d_arr_padded_list])
        return img4d_arr_padded
    else:
        return img4d_arr

def additionalpad_img4dlist_arr(img4d_arr_list, core_patch_size, sampling_patch_size, stride):
    # check if need additional padding
    p = int((sampling_patch_size - core_patch_size) / 2.0)
    if p > 0:
        img4d_arr_padded_list = [np.pad(i, ((0,0), (0,0), (p,p), (p,p)), 'reflect')
                                  for i in img4d_arr_list]
        # print('additional padding', [i.shape for i in img4d_arr_padded_list])
        return img4d_arr_padded_list
    else:
        return img4d_arr_list

def additionalpad_img4dgroup_arr(img4d_arr_group, core_patch_size, sampling_patch_size, stride):
    if isinstance(img4d_arr_group, np.ndarray):
        return additionalpad_img4d_arr(img4d_arr_group, core_patch_size, sampling_patch_size, stride)
    else:
        return additionalpad_img4dlist_arr(img4d_arr_group, core_patch_size, sampling_patch_size, stride)





def autopad_img4d_arr_inverse(img4d_arr_padded, hw_ori, patch_size, stride):
    h_padded, w_padded = img4d_arr_padded.shape[-2:]
    h_ori, w_ori = hw_ori
    h_p_lr = get_padding_info_two_ends(h_ori, patch_size, stride)
    w_p_lr = get_padding_info_two_ends(w_ori, patch_size, stride)
    return img4d_arr_padded[:, :, h_p_lr[0]:h_padded-h_p_lr[-1], w_p_lr[0]:w_padded-w_p_lr[-1]]




def img4dlist_to_patch4dlist_general(feat_img4d_list, sampling_setting, hw_collect_flag=False):
    '''
    - Work for single channel, each elemetn in feat_img_singchl_set is [n, 1, h, w]
    - Can work for different img_arr with different n, h, w but the same corepatch_size
    
    - Automatically pad the images:
        to ensure afterwards patches (based on sampling_setting['sampling_size'], the real patch size)
        are of integer number (sampling_setting['corepatch_for_autopadding'] --> the auto-padded images
                               & sampling_setting['sampling_size'] --> the additional padded images) 
    
    - Addtional padding after the auto-padding is designed for sampling_setting['sampling_size']
    
    - The same stride sampling_setting['sampling_stride'] is shared 
        between sampling_setting['corepatch_for_autopadding'] 
        and sampling_setting['sampling_size'].
    
    Parameters
    ----------
    feat_img4d_list : TYPE
        DESCRIPTION.
    sampling_setting : TYPE
        e.g. {'sampling_size':3, 'sampling_stride':3, 'corepatch_for_autopadding':3}.
             {'sampling_size':7, 'sampling_stride':3, 'corepatch_for_autopadding':3}.

    Returns
    -------
    feat_patch4d_list: a list of 4d patches

    '''

    # auto padding, also work for images of which original size is suitable for integer num. patches
    # feat_img4d_padded_list = [autopad_img4d_arr(i,
    #                                         sampling_setting['corepatch_for_autopadding'],
    #                                         sampling_setting['sampling_stride'])
    #                           for i in feat_img4d_list]
    feat_img4d_padded_list = autopad_img4dlist_arr(feat_img4d_list,
                                                   sampling_setting['corepatch_for_autopadding'],
                                                   sampling_setting['sampling_stride'])
    print('aft auto padding', [i.shape for i in feat_img4d_padded_list])
    hw_autopadded_lists = [[i.shape[-2] for i in feat_img4d_padded_list],
                           [i.shape[-1] for i in feat_img4d_padded_list]]

    # additional padding
    feat_img4d_padded_list = additionalpad_img4dlist_arr(feat_img4d_padded_list,
                                                   sampling_setting['corepatch_for_autopadding'],
                                                   sampling_setting['sampling_size'],
                                                   sampling_setting['sampling_stride'])
    print('aft additional padding', [i.shape for i in feat_img4d_padded_list])


    feat_patch4d_list = [window_process_4d_patch(i,
                                                  sampling_setting['sampling_size'],
                                                  sampling_setting['sampling_stride'],
                                                  1,
                                                  False)\
                          for i in feat_img4d_padded_list]
    # feat_patch4d_list = [window_process_4d_patch_batch_by_samples(i,
    #                                              {'patch_size':sampling_setting['sampling_size'],
    #                                               'stride':sampling_setting['sampling_stride'],
    #                                               'dilate':1,
    #                                               'pad_flag':False})\
    #                      for i in feat_img4d_padded_list]
    print([i.shape for i in feat_patch4d_list])
    if hw_collect_flag:
        n_list = [i.shape[0] for i in feat_img4d_list]
        hw_lists = [[i.shape[-2] for i in feat_img4d_list],
                    [i.shape[-1] for i in feat_img4d_list]]
        hw_finalpadded_lists = [[i.shape[-2] for i in feat_img4d_padded_list],
                                [i.shape[-1] for i in feat_img4d_padded_list]]

        return feat_patch4d_list, n_list, hw_lists, hw_autopadded_lists, hw_finalpadded_lists, feat_img4d_padded_list
    else:
        return feat_patch4d_list


# resign to class_data_loader.DATA_LOADER.collect_patches
def img4dgroup_to_patch4dlist_general(multiImg4d_or_oneImg4dlist, group_sampling_setting, group_hw_collect_flag=False):
    if isinstance(multiImg4d_or_oneImg4dlist, np.ndarray):
        return img4dlist_to_patch4dlist_general([multiImg4d_or_oneImg4dlist],
                                                group_sampling_setting, hw_collect_flag=group_hw_collect_flag)
    else:
        return img4dlist_to_patch4dlist_general(multiImg4d_or_oneImg4dlist,
                                                group_sampling_setting, hw_collect_flag=group_hw_collect_flag)





def window_process_4d_patch_nonoverlap_inverse(patch4d, h_img, w_img):
    # patch4d (n, c, h, w)
    patch_size = patch4d.shape[-1]
    c_img = patch4d.shape[-3]
    num_patch_h = h_img // patch_size
    num_patch_w = w_img // patch_size
    imgs_unfold = torch.from_numpy(patch4d.reshape((-1, num_patch_h*num_patch_w, c_img*patch_size*patch_size))).permute(0, 2, 1)
    imgs = f.fold(imgs_unfold, (h_img, w_img), patch_size, stride=patch_size, dilation=1, padding=0).numpy()
    return imgs


def patch4dlist_to_img4dlist_general(feat_patch4d_arr, sampling_setting, n_list, hw_lists, hw_finalpadded_lists):
    '''
    sampling_setting : TYPE
        e.g. {'sampling_size':3, 'sampling_stride':3, 'corepatch_for_autopadding':3}.
             {'sampling_size':7, 'sampling_stride':3, 'corepatch_for_autopadding':3}.
    '''
    h_list, w_list = hw_lists
    h_padded_list, w_padded_list = hw_finalpadded_lists
    patch_size = feat_patch4d_arr.shape[-1]
    num_sample_list = [n*h*w // patch_size**2 for n, h, w in zip(n_list, h_padded_list, w_padded_list)]
    cumnum_sample_list = [0]
    cumnum_sample_list.extend(np.cumsum(num_sample_list))
    print('cumnum_sample_list', cumnum_sample_list)
    print('total patches', feat_patch4d_arr.shape)
    # patches --> auto-padded images
    img4d_list = []
    for set_idx in range(len(n_list)):
        print('selected patches', feat_patch4d_arr[cumnum_sample_list[set_idx]:cumnum_sample_list[set_idx+1]].shape)
        img4d=\
            window_process_4d_patch_nonoverlap_inverse(feat_patch4d_arr[cumnum_sample_list[set_idx]:cumnum_sample_list[set_idx+1]],
                                                       h_padded_list[set_idx], w_padded_list[set_idx])
        img4d_list.append(img4d)

    # crop out borders due to addtional padding
    p = int((sampling_setting['sampling_size'] - sampling_setting['corepatch_for_autopadding']) / 2.0)
    if p > 0:
        img4d_list = [i[:, :, p:-1, p:-p] for i in img4d_list]

    # crop out borders due to auto-padding
    img4d_cropped_list = [autopad_img4d_arr_inverse(i,
                                                    [h, w],
                                                    sampling_setting['corepatch_for_autopadding'],
                                                    sampling_setting['sampling_stride'])
                          for i,h,w in zip(img4d_list, h_list, w_list)]
    return img4d_cropped_list


# shape = (2, 1, 10, 10)
# a = np.arange(np.prod(shape)).reshape(shape).astype(float)
# a_unfold = f.unfold(torch.from_numpy(a), kernel_size=(2, 2), stride=2, dilation=1)
# # torch.Size([2, 4, 25]) #(N,C×∏(kernel_size),L)
# a_rec = f.fold(a_unfold, (10, 10), 2, stride=2, dilation=1, padding=0)
# # torch.Size([2, 1, 10, 10])
# a_patches = a_unfold.permute(0, 2, 1).reshape(-1, a.shape[1], 2, 2).numpy()
# # (50, 1, 2, 2)
# # (50, 1, 2, 2) --> (2*25, 1*2*2) --> torch.Size([2, 4, 25])
# a_rec_from_patches = window_process_4d_patch_nonoverlap_inverse(a_patches, 10, 10)

# from sklearn.metrics import mean_squared_error as MSE
# shape = (2, 1, 10, 10)
# a = np.arange(np.prod(shape)).reshape(shape).astype(float) * 10
# a_list = [a, a*10]
# sampling_setting = {'sampling_size':3, 'sampling_stride':3, 'corepatch_for_autopadding':3}
# p_list, n_list, hw_lists, hw_autopadded_lists, hw_finalpadded_lists = \
#     img4dlist_to_patch4dlist_general(a_list, sampling_setting, hw_collect_flag=True)
# a_list_rec = patch4dlist_to_img4dlist_general(np.concatenate(p_list, axis=0), sampling_setting, n_list, hw_lists, hw_finalpadded_lists)
# print([MSE(i.reshape(-1), j.reshape(-1)) for i, j in zip(a_list, a_list_rec)])



#%%

def crop_ring_to_rect(patch4d, cropping_setting):
    ring_width_pair, local_block_size = cropping_setting['ring_width_pair'], cropping_setting['local_block_size']
    # patch4d (n, 1, 15, 15)
    ring_width_outside, ring_width_inside = ring_width_pair
    ring_pad = (patch4d.shape[-1] - ring_width_outside) // 2
    # ring_width = (ring_width_outside - ring_width_inside) // 2

    if ring_width_inside == 0:
    # a whole patch
        return patch4d[:, :,
                       ring_pad:(ring_pad+ring_width_outside),
                       ring_pad:(ring_pad+ring_width_outside)]
    else:
        # only when non-overlapping of local blocks
        # a real ring
        assert (ring_width_outside - ring_width_inside) // 2 == local_block_size


        num_local_1d = ring_width_outside // local_block_size
        local_idx_selected = [i for i in range(num_local_1d)]
        local_idx_selected.extend([i*num_local_1d for i in np.arange(1, num_local_1d-1)])
        local_idx_selected.extend([i*num_local_1d - 1 for i in np.arange(2, num_local_1d)])
        local_idx_selected.extend([num_local_1d*(num_local_1d-1) + i for i in np.arange(num_local_1d)])
        local_idx_selected = np.array(local_idx_selected)
        # print(local_idx_list)

        # (n, 1, 9, 9) --> (n*9, 1, 3, 3) if local_block is 3
        local_block_nonoverlap = \
            window_process_4d_patch(patch4d[:, :,
                                            ring_pad:(ring_pad+ring_width_outside),
                                            ring_pad:(ring_pad+ring_width_outside)],
                                    local_block_size,
                                    local_block_size, 1, False)

        # --> (n, 8, 1, 3, 3)
        local_block_nonoverlap = \
            local_block_nonoverlap.reshape((patch4d.shape[0],
                                            -1,
                                            patch4d.shape[1],
                                            local_block_size,
                                            local_block_size))

        # --> (n, 8, 1, 3, 3)
        local_block_nonoverlap = \
            local_block_nonoverlap[:, local_idx_selected]

        # --> (n, 1, 3, 3, 8) --> (n, 1, 3, 3*8)
        local_block_nonoverlap = np.moveaxis(local_block_nonoverlap, 1, -1).reshape((patch4d.shape[0],
                                                                   patch4d.shape[1],
                                                                   local_block_size,
                                                                   -1))
        # (n*8, 1, 3, 3)
        gc.collect()
        return local_block_nonoverlap#, len(local_idx_selected)




#%%
# a = np.arange(2*15*15, dtype=float).reshape(2, 1, 15, 15)
# # b = window_process_4d_patch(a, 3, 3, 1, False)
# b = crop_ring_to_rect(a, [9, 3], local_block_size=3)
# print(a)
# print(b)














#%% PIXE --> IMG

def get_spatial_size_info(multiImg4dArr_or_oneImg4dList):
    if isinstance(multiImg4dArr_or_oneImg4dList, list):
        h_info = [i.shape[-2] for i in multiImg4dArr_or_oneImg4dList]
        w_info = [i.shape[-1] for i in multiImg4dArr_or_oneImg4dList]
    elif isinstance(multiImg4dArr_or_oneImg4dList, np.ndarray):
        h_info, w_info = multiImg4dArr_or_oneImg4dList.shape[-2:]
    return h_info, w_info


def pixellist_to_images(pred_list, sample_id_list, h_info, w_info, n=0):
    if isinstance(h_info, int):
        assert n > 0
        img_arr = np.zeros(h_info * w_info * n)
    else:
        num_pixel_list = [h*w for h,w in zip(h_info, w_info)]
        cumu_num_pixel_list = [0]
        cumu_num_pixel_list.extend(np.cumsum(num_pixel_list))
        cumu_num_pixel_list = np.array(cumu_num_pixel_list)
        img_arr = np.zeros(np.sum(num_pixel_list))

    for pred, sample_id in zip(pred_list, sample_id_list):
        img_arr[sample_id] = pred

    if isinstance(h_info, int):
        img_arr_new = img_arr.reshape(n, h_info, w_info)
    else:
        img_arr_new = [img_arr[i:j].reshape(h, w) for i,j,h,w in zip(cumu_num_pixel_list[:-1], cumu_num_pixel_list[1:], h_info, w_info)]
    return img_arr_new


def pixellist_to_images_update(pred_list, sample_id_list, img_arr_old):
    h_info, w_info = get_spatial_size_info(img_arr_old)

    if isinstance(h_info, int):
        n = len(h_info)
        img_arr = np.zeros(h_info * w_info * n)
    else:
        num_pixel_list = [h*w for h,w in zip(h_info, w_info)]
        cumu_num_pixel_list = [0]
        cumu_num_pixel_list.extend(np.cumsum(num_pixel_list))
        cumu_num_pixel_list = np.array(cumu_num_pixel_list)
        img_arr = np.zeros(np.sum(num_pixel_list))

    for pred, sample_id in zip(pred_list, sample_id_list):
        img_arr[sample_id] = pred

    print('h_info', h_info)
    print('w_info', w_info)
    print('num_pixel_list', num_pixel_list)
    print('cumu_num_pixel_list', cumu_num_pixel_list)
    print('cumu_num_pixel_list[:-1]', cumu_num_pixel_list[:-1])
    print('cumu_num_pixel_list[1:]', cumu_num_pixel_list[1:])
    print('img_arr', img_arr.shape)
    for i,j in zip(cumu_num_pixel_list[:-1], cumu_num_pixel_list[1:]):
        print(i, j, img_arr[i:j].shape)


    if isinstance(h_info, int):
        img_arr_new = img_arr.reshape(img_arr_old.shape)
        img_arr_new += img_arr_old
    else:
        img_arr_new = [img_arr[i:j].reshape(img_arr_o.shape) + img_arr_o for i,j,img_arr_o in zip(cumu_num_pixel_list[:-1], cumu_num_pixel_list[1:], img_arr_old)]
    return img_arr_new









#%%

def transf_sampleidx_to_imghwidx(sample_idx, h, w):
    '''
    Find the img_idx, h_idx, and w_idx for each sample
    Can work for a list of images with variant h and w

    Parameters
    ----------
    sample_idx : list or 1-D np.ndarray
        w.r.t samples.
    h : int or a list (or 1-D np.ndarray) of height of all images.
        w.r.t images.
    w : int or a list (or 1-D np.ndarray) of height of all images.
        w.r.t images.

    Returns
    -------
    sample_img_idx : 1-D np.ndarray
        w.r.t samples.
    sample_h_idx : 1-D np.ndarray
        w.r.t samples.
    sample_w_idx : 1-D np.ndarray
        w.r.t samples.

    '''
    sample_idx = np.array(sample_idx)
    
    if isinstance(h, int) and isinstance(w, int):
        sample_img_idx = sample_idx // (h * w)
        sample_h_idx = sample_idx % (h * w) // w
        sample_w_idx = sample_idx % (h * w) % w
        return sample_img_idx, sample_h_idx, sample_w_idx
    else:
        
        if isinstance(h, int): # but w is not an int
            h = np.ones(len(w), dtype=int) * h # w.r.t. images
        else:
            w = np.ones(len(h), dtype=int) * w # w.r.t. images
        h = np.array(h)
        w = np.array(w)
        
        num_pixel_list = [i*j for i,j in zip(h,w)]
        cumu_num_pixel_arr = np.cumsum(num_pixel_list)
        cumu_num_pixel_arr = np.insert(cumu_num_pixel_arr, 0, 0)
        sample_img_idx = np.searchsorted(cumu_num_pixel_arr[1:], sample_idx, side='right')
        
        residue = sample_idx - cumu_num_pixel_arr[sample_img_idx]
        # sample_img_height = h[sample_img_idx]
        sample_img_width = w[sample_img_idx]
        sample_h_idx = residue // sample_img_width
        sample_w_idx = residue % sample_img_width
        return sample_img_idx, sample_h_idx, sample_w_idx


# transf_sampleidx_to_imghwidx([0, 2500-1, 
#                               2500, 5100-1,
#                               5100, 7800-1], [10, 11, 12], [250, 260, 270])



def get_sample_for_one_img(sample_imghw_idx, img_idx_seleted=0):
    sample_img_idx, sample_h_idx, sample_w_idx = sample_imghw_idx
    sample_idx_cond = np.where(sample_img_idx==img_idx_seleted)[0]
    # print('selected sample num', len(sample_idx_cond), 'ratio', len(sample_idx_cond)/len(sample_img_idx))
    sample_img_idx_selected = np.ones(len(sample_idx_cond), dtype=int)
    sample_h_idx_selected = sample_h_idx[sample_idx_cond]
    sample_w_idx_selected = sample_w_idx[sample_idx_cond]
    return sample_img_idx_selected, sample_h_idx_selected, sample_w_idx_selected







#%%
# visualize selected block in one image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import cv2







def visualize_box_in_one_img(img_arr, sample_imghw_idx, box_size, img_idx=0, fig_name=[]):
    img_show = np.squeeze(img_arr[img_idx])
    sample_img_idx_toshow, sample_h_idx_toshow, sample_w_idx_toshow = get_sample_for_one_img(sample_imghw_idx, img_idx_seleted=img_idx)
    print('single img, ratio', len(sample_img_idx_toshow)/np.product(img_show.shape))
    
    boxes = [Rectangle((w, h), box_size, box_size) for w, h in zip(sample_w_idx_toshow, sample_h_idx_toshow)]
    pc = PatchCollection(boxes, linewidth=1, facecolor='none', edgecolor='r')
    
    plt.figure(figsize=(10,6))
#    plt.imshow(np.zeros((160, 240)))
    plt.imshow(img_show)
    plt.gca().add_collection(pc)
    # plt.colorbar()
    if len(fig_name) > 0:
        plt.savefig(fig_name)
    plt.show()
    
    
def visualize_dot_in_one_img(img_arr, sample_imghw_idx, marker_size, img_idx=0, color='r', fig_name=[]):
    img_show = np.squeeze(img_arr[img_idx])
    sample_img_idx_toshow, sample_h_idx_toshow, sample_w_idx_toshow = get_sample_for_one_img(sample_imghw_idx, img_idx_seleted=img_idx)
    print('single img, ratio', len(sample_img_idx_toshow)/np.product(img_show.shape))
    
    plt.figure(figsize=(10,6))
    plt.imshow(img_show)
    # since there is plt.imshow in advance, the plotting origin is changed to
    # the upper left
    plt.scatter(sample_w_idx_toshow, sample_h_idx_toshow, s=marker_size, c=color, alpha=0.5)
    plt.grid(False)
    if len(fig_name) > 0:
        plt.savefig(fig_name)
    plt.show()








def visualize_box_in_one_img_standard(img_arr, box_size_list, sample_idx=[], sample_imghw_idx=[], img_idx=0, fig_name=[]):
    img_show = np.squeeze(img_arr[img_idx])
    if len(sample_imghw_idx) > 0:
        sample_img_idx_toshow, sample_h_idx_toshow, sample_w_idx_toshow = sample_imghw_idx
    else:
        sample_img_idx_toshow, sample_h_idx_toshow, sample_w_idx_toshow = get_sample_for_one_img(sample_idx, img_idx_seleted=img_idx)

    box_size_h_list, box_size_w_list = box_size_list
    print('single img, ratio', len(sample_img_idx_toshow)/np.product(img_show.shape))

    boxes = [Rectangle((w, h), box_size_w, box_size_h) for w, h, box_size_w, box_size_h in zip(sample_w_idx_toshow, sample_h_idx_toshow, box_size_h_list, box_size_w_list)]
    pc = PatchCollection(boxes, linewidth=3, facecolor='none', edgecolor='r')
    
    plt.figure(figsize=(10,6))
#    plt.imshow(np.zeros((160, 240)))
    plt.imshow(img_show, cmap='gray', vmin=0, vmax=255)
    plt.gca().add_collection(pc)
    # plt.colorbar()
    if len(fig_name) > 0:
        plt.savefig(fig_name)
    plt.show()




#%%

def plot_scatter_equal_axes(arr_info, label_info, axis_lim, fig_title='', fig_file_name=''):
    x_arr, y_arr = arr_info
    x_label, y_label = label_info
    plt.figure()
    plt.scatter(x_arr, y_arr)
    plt.xlim(axis_lim)
    plt.ylim(axis_lim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if len(fig_file_name) > 0:
        plt.savefig(fig_file_name)
    plt.show()


def plot_hist2d_equal_axes(arr_info, label_info, axis_lim, fig_title='', fig_file_name=''):
    x_arr, y_arr = arr_info
    x_label, y_label = label_info
    bin_edges = np.arange(axis_lim[0], axis_lim[1]+2, 2)
    plt.figure()#(figsize=(5, 5))
    plt.hist2d(x_arr, y_arr, bins=[bin_edges, bin_edges])
    plt.xlim(axis_lim)
    plt.ylim(axis_lim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.tight_layout()
    if len(fig_file_name) > 0:
        plt.savefig(fig_file_name)
    plt.show()
    


# plt.xlim(axis_lim)
# plt.ylim(axis_lim)
# plt.xlabel(x_label)
# plt.ylabel(y_label)
# plt.title(fig_title)
# plt.legend()
# plt.tight_layout()
# plt.savefig(fig_file_name)
# plt.show()





#%%
# SPATIAL CROPPING, TRANSFER, ETC.
'''
FUNCTION LIST:
    # patch 1d <-> 3d
    idx_transf_patch1d_to_patch3d(patch_idx_arr, img_h_local, img_w_local) = img_idx, h_idx, w_idx
    idx_transf_patch3d_to_patch1d(img_idx, h_idx, w_idx, img_h_local, img_w_local) = patch_idx_arr
    
    # parent patch -> children patch
    idx_transf_patch3d_to_smallerpatch3d(img_idx_patch, h_idx_patch, w_idx_patch, patch_size_ratio) = img_idx_smallerpatch, h_idx_smallerpatch, w_idx_smallerpatch
    idx_transf_patch1d_to_smallerpatch1d(patch_idx_arr, img_h, img_w, patch_size_big, patch_size_small) = patch_idx_arr_smallerpatch
    
    # (helper) patch -> pixel
    idx_transf_patch1d_to_pixel3d(patch_idx_arr, patch_img_h, patch_img_w, patch_size) = pixel_img_idx, pixel_h_idx, pixel_w_idx
'''
# -------------------
def idx_transf_patch1d_to_patch3d(patch_idx_arr, img_h_local, img_w_local):
    img_idx = patch_idx_arr // (img_h_local*img_w_local)
    h_idx = patch_idx_arr % (img_h_local*img_w_local) // img_w_local
    w_idx = (patch_idx_arr % (img_h_local*img_w_local) % img_w_local).astype(int)
    return img_idx, h_idx, w_idx
    
def idx_transf_patch3d_to_patch1d(img_idx, h_idx, w_idx, img_h_local, img_w_local):
    patch_idx_arr = img_idx * (img_h_local*img_w_local) + h_idx*img_w_local + w_idx
    return patch_idx_arr

# -------------------
# need to change ? the input?
def idx_transf_patch3d_to_smallerpatch3d(img_idx_patch, h_idx_patch, w_idx_patch, patch_size_ratio):
    img_idx = [i for i in img_idx_patch for k in np.arange(patch_size_ratio) for j in np.arange(patch_size_ratio)]
    h_idx = [i*patch_size_ratio + k for i in h_idx_patch for k in np.arange(patch_size_ratio) for j in np.arange(patch_size_ratio)]
    w_idx = [i*patch_size_ratio + j for i in w_idx_patch for k in np.arange(patch_size_ratio) for j in np.arange(patch_size_ratio)]
    return np.array(img_idx), np.array(h_idx), np.array(w_idx)

def idx_transf_patch1d_to_smallerpatch1d(patch_idx_arr, img_h, img_w, patch_size_big, patch_size_small):
    img_idx_patch, h_idx_patch, w_idx_patch = idx_transf_patch1d_to_patch3d(patch_idx_arr, img_h//patch_size_big, img_w//patch_size_big)
#    print(img_idx_patch, h_idx_patch, w_idx_patch)
    img_idx, h_idx, w_idx = idx_transf_patch3d_to_smallerpatch3d(img_idx_patch, h_idx_patch, w_idx_patch, patch_size_big//patch_size_small)
    patch_idx_arr_smallerpatch = idx_transf_patch3d_to_patch1d(img_idx, h_idx, w_idx, img_h//patch_size_small, img_w//patch_size_small)
    return patch_idx_arr_smallerpatch

#a = np.arange(1)+3
#idx_3d = idx_transf_patch1d_to_smallerpatch1d(a, 8, 8, 4, 1)
#print(idx_3d)

# --------------------
def idx_transf_patch1d_to_pixel3d(patch_idx_arr, img_h, img_w, patch_size):
    img_idx_wrt_patch, h_idx_wrt_patch, w_idx_wrt_patch = idx_transf_patch1d_to_patch3d(patch_idx_arr, img_h//patch_size, img_w//patch_size)
    print('range of img_idx', np.min(img_idx_wrt_patch), np.max(img_idx_wrt_patch))
    print('range of h_idx', np.min(h_idx_wrt_patch), np.max(h_idx_wrt_patch))
    print('range of w_idx', np.min(w_idx_wrt_patch), np.max(w_idx_wrt_patch))
    img_idx, h_idx, w_idx = idx_transf_patch3d_to_smallerpatch3d(img_idx_wrt_patch, h_idx_wrt_patch, w_idx_wrt_patch, patch_size)
    print('range of img_idx', np.min(img_idx), np.max(img_idx))
    print('range of h_idx', np.min(h_idx), np.max(h_idx))
    print('range of w_idx', np.min(w_idx), np.max(w_idx))
    return img_idx, h_idx, w_idx


# ----------------------

    
#%%
#a = np.arange(2, dtype=int)
#b = np.arange(2, dtype=int) + 10
#comb = np.array(np.meshgrid(h_idx_cand, w_idx_cand)).T.reshape(-1,2)

h_idx_wrt_patch = np.arange(100, dtype=int)
w_idx_wrt_patch = np.arange(100, dtype=int) +1000

#c = np.array(np.meshgrid(h_idx_cand, w_idx_cand)).T
#comb = c.reshape(-1,2)



#%%
# STATISTICS OPERATION / ANALYSIS

'''
FUNCTION LIST:
    binary_partition_byperc_fit(arr, perc_thre) = value_thre, [sample_num_perc, sample_idx_smallvalue, sample_idx_largevalue]
    binary_partition_byvalue(arr, value_thre) = sample_num_perc, sample_idx_smallvalue, sample_idx_largevalue
    
    ### get bins with given cumu perc distributions (w.r.t. general feature arr)
    get_value_bins_byperc_fit(cumu_perc_list, feat_arr) = bin_info, sample_idx_sorted_all
    get_value_bins_pred(bin_info, feat_arr) = bin_idx_samplewise
    
    ### get bins with given cumu perc distributions (w.r.t. distance to the feature mean)
    get_distbins_byperc_fit(cumu_perc_list, feat_arr) = bin_info, sample_idx_sorted_all
    get_distbins_pred(bin_info, feat_arr) = bin_idx_samplewise
    
    ## get cluster-wise sample info (sample feature dimension is n-D)
    get_clstwise_sampleinfo(label_arr, unique_labels) = sample_idx_all, sample_number_all, sample_number_perc_all
    
    ## get histogram info and quantized histogram bins info (sample feature dimension is 1-D)
    get_hist_info(feat_arr, num_bins_or_bin_edges) = pdf, bin_edges, bin_centers
    
    (class)
    headtail_perc_cutter(thre_perc, head_tail_sign).fit/predict(arr)
'''



# subspace learning




def binary_partition_byperc_fit(arr, perc_thre):
    value_thre = np.sort(arr.reshape(-1))[int(len(arr)*perc_thre)-1]
    return value_thre, binary_partition_byvalue(arr, value_thre)
    

def binary_partition_byvalue(arr, value_thre):
    sample_idx_smallvalue = np.where(arr.reshape(-1) <= value_thre)[0]
    sample_idx_largevalue = np.where(arr.reshape(-1) > value_thre)[0]
    return len(sample_idx_smallvalue)*1.0 / len(arr), sample_idx_smallvalue, sample_idx_largevalue
    






### get bins with given cumu perc distributions (w.r.t. general feature arr)
def get_value_bins_byperc_fit(cumu_perc_list, feat_arr):
    '''
    increasingly sort samples with 1-D feat
    cut by cumu perc
    '''
    feat_1d = feat_arr.reshape(-1)
    sample_idx_sorted = np.argsort(feat_1d.reshape(-1))
    cumu_num_list = (np.array(cumu_perc_list) * len(sample_idx_sorted)).astype(int)
    sample_idx_sorted_all = [sample_idx_sorted[cumu_num_list[i] : cumu_num_list[i+1]] for i in range(len(cumu_perc_list)-1)]
    dist_bin_edges = [(feat_1d[sample_idx_sorted_all[0]])[0]]
    dist_bin_edges.extend([(feat_1d[i])[-1] for i in sample_idx_sorted_all])
    dist_bin_edges = np.array(dist_bin_edges).reshape(-1)
    bin_info = {'mean':np.mean(feat_arr.reshape((len(feat_arr), -1)), axis=0, keepdims=True), 'bin_edges':dist_bin_edges}
    return bin_info, sample_idx_sorted_all

def get_value_bins_pred(bin_info, feat_arr):
    bin_idx_samplewise = np.searchsorted(bin_info['bin_edges'], feat_arr.reshape(-1), side='left') - 1
    bin_idx_samplewise[bin_idx_samplewise < 0] = 0
    bin_idx_samplewise[bin_idx_samplewise > len(bin_info['bin_edges'])-2] = len(bin_info['bin_edges'])-2
    return bin_idx_samplewise
#a = np.arange(20) - 10
#bininfo = {'bin_edges':np.arange(4)*2}
#bin_idx_samplewise = get_value_bins_pred(bininfo, a)


#a = np.random.choice(np.arange(11)/10, 11, replace=False) -0.5
##a = np.arange(11)/10
#b,c = get_value_bins_byperc_fit([0, 0.33, 0.67, 1], a)
#print('a', a)
#print('b', b)
#print('c', c)
#print([a[i] for i in c])
#print()
#
#d = get_value_bins_pred(b, a)
#print(d)
#print([np.where(d==i)[0] for i in np.unique(d)])



### get bins with given cumu perc distributions (w.r.t. distance to the feature mean)
def get_distbins_byperc_fit(cumu_perc_list, feat_arr):
    '''
    cumu_perc_list: e.g. [0, 0.1, 0.2, ..., 1]
    '''
    feat_dist = cdist(feat_arr.reshape((len(feat_arr), -1)), np.mean(feat_arr.reshape((len(feat_arr), -1)), axis=0, keepdims=True))
    return get_value_bins_byperc_fit(cumu_perc_list, feat_dist)

def get_distbins_pred(bin_info, feat_arr):
    feat_dist = cdist(feat_arr.reshape((len(feat_arr), -1)), bin_info['mean'])
    return get_value_bins_pred(bin_info, feat_dist)

#print()
#a = np.random.choice(np.arange(11)/10, 11, replace=False)
##a = np.arange(11)/10
#b,c = get_bins_equalperc_fit([0, 0.33, 0.67, 1], a)
#print('a', a)
#print('b', b)
#print('c', c)
#print([a[i] for i in c])
#print()
#
#d = get_bins_pred(b, a)
#print(d)
#print([np.where(d==i)[0] for i in np.unique(d)])

def get_bins_pred_bybinedges(bin_info, feat_arr):
    bin_idx_samplewise = np.searchsorted(bin_info['bin_edges'], feat_arr.reshape(-1), side='left') - 1
    bin_idx_samplewise[bin_idx_samplewise < 0] = 0
    bin_idx_samplewise[bin_idx_samplewise > len(bin_info['bin_edges'])-2] = len(bin_info['bin_edges']) - 2
    return bin_idx_samplewise

## get cluster-wise sample info (sample feature dimension is n-D)
def get_clstwise_sampleinfo(label_arr, unique_labels):
    sample_idx_all = np.array([np.where(label_arr==i)[0] for i in unique_labels], dtype=object)
    sample_number_all = np.array([len(i) for i in sample_idx_all])
#    print('np.sum(sample_number_all)', np.sum(sample_number_all))
    sample_number_perc_all = sample_number_all / np.sum(sample_number_all)
    return sample_idx_all, sample_number_all, sample_number_perc_all

## get histogram info and quantized histogram bins info (sample feature dimension is 1-D)
def get_hist_info(feat_arr, num_bins_or_bin_edges, cond_hist_flag=False, base_num=0):
    sample_num, bin_edges = np.histogram(feat_arr.reshape(-1), bins=num_bins_or_bin_edges)
    if cond_hist_flag:
        base_num = len(feat_arr)
    pdf = sample_num / base_num
    bin_centers = np.diff(bin_edges) / 2.0 + bin_edges[:-1]
    return pdf, bin_edges, bin_centers




def get_bin_qualified_samples(right_bin_edges, samples_arr, bin_idx):
    '''
    each bin [, )
    right_bin_edges: (nun_bins, )
        each element is the right edge of the bins,
        but only the first nun_bins-1 inner bin edges are useful
    samples_arr: (num_samples, )
    bin_idx: int
        bin_idx = 1, 2, ..., num_bins
    '''
    cond_sample_idx = np.arange(len(samples_arr))
    assert bin_idx > 0

    if bin_idx == 1:
        sample_ind = samples_arr < right_bin_edges[bin_idx - 1]
    elif bin_idx == len(right_bin_edges):
        sample_ind = samples_arr >= right_bin_edges[bin_idx - 2]
    else:
        sample_ind = \
            np.logical_and(samples_arr < right_bin_edges[bin_idx - 1],
                           samples_arr >= right_bin_edges[bin_idx - 2])
    return cond_sample_idx[sample_ind]


def get_bin_leakage_samples(right_bin_edges, samples_arr, bin_idx):
    '''
    each bin [, )
    right_bin_edges: (nun_bins, )
        each element is the right edge of the bins,
        but only the first nun_bins-1 inner bin edges are useful
    samples_arr: (num_samples, )
    bin_idx: int
        bin_idx = 1, 2, ..., num_bins
    '''
    cond_sample_idx = np.arange(len(samples_arr))
    assert bin_idx > 0

    if bin_idx == 1:
        sample_ind = samples_arr >= right_bin_edges[bin_idx - 1]
    elif bin_idx == len(right_bin_edges):
        sample_ind = samples_arr < right_bin_edges[bin_idx - 2]
    else:
        sample_ind = \
            np.logical_or(samples_arr < right_bin_edges[bin_idx - 2],
                           samples_arr >= right_bin_edges[bin_idx - 1])
    return cond_sample_idx[sample_ind]


#%%
def get_selected_sample_idx_by_label(label_arr, interest_label_arr):
    '''
    label_arr: (num_samples, )
    interest_label_arr: (num_interest_samples, )
    '''
    sample_idx_selected = np.array([i for i in range(len(label_arr)) if label_arr[i] in interest_label_arr])
    return sample_idx_selected
# a = [0, 1, 0, 2, 5, 3, 1, 0]
# b = [0, 1]
# print(get_selected_sample_idx_by_label(a, b))

#%%

### 1-d clustering with cluster boundaries 
#def get_bin_inner_edges_byperc(obj_arr, cumu_perc_thre_list_inner):
#    print('cumu_perc_thre_list_inner', cumu_perc_thre_list_inner)
#    obj_arr_sorted = np.sort(obj_arr.reshape(-1))
#    num_sample = len(obj_arr)
#    sample_num_thre_list = (np.array(cumu_perc_thre_list_inner) * num_sample).astype(int)
#    print('sample_num_thre_list', sample_num_thre_list)
#    bin_inner_edge_list = obj_arr_sorted[sample_num_thre_list-1]
#    print('bin_inner_edge_list', bin_inner_edge_list)
#    bin_idx = np.searchsorted(bin_inner_edge_list.reshape(-1), np.array(obj_arr).reshape(-1))
#    return bin_inner_edge_list, bin_idx
#
#def get_bin_idx_withinner_edges(obj_arr, bin_inner_edge_list):
#    print('obj_arr', obj_arr.shape)
#    print('bin_inner_edge_list', bin_inner_edge_list.shape)
#    bin_idx = np.searchsorted(bin_inner_edge_list.reshape(-1), np.array(obj_arr).reshape(-1))
#    return bin_idx
#
#def get_bin_label(edges, arr_to_quan):
#    bin_label = np.searchsorted(edges, arr_to_quan, side='right') - 1
#    bin_label[bin_label < 0] = 0
#    bin_label[bin_label >= len(edges) -1] = len(edges) -2
#    return bin_label


class headtail_perc_cutter(object):
    
    def __init__(self, thre_perc, head_tail_sign):
        '''
        head_tail_sign = -1, 1 --> head, tail
        '''
        self.thre_perc = thre_perc
        self.head_tail_sign = head_tail_sign
        self.thre_value = 0
    
    def fit(self, arr):
        sorting_order = np.argsort(arr.reshape(-1))
        num_thre = int(self.thre_perc * len(arr))
        self.thre_value = (arr.reshape(-1))[sorting_order[(num_thre * self.head_tail_sign * (-1))-1]]
    
    def predict(self, arr):
        if self.head_tail_sign < 0:
            selected_sample_idx = np.where(arr.reshape(-1) <= self.thre_value)[0]
        else:
            selected_sample_idx = np.where(arr.reshape(-1) >= self.thre_value)[0]
        return selected_sample_idx




#%%
# SOME VISUALIZATIONS
import matplotlib.pyplot as plt
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', titlesize=10)  # fontsize of the figure title

import cv2

'''
FUNCTION LIST:
    equalscale_scatter(arr_set, axis_label_set, axis_range=[-200, 200])
    equalscale_hist2d(arr_set, axis_label_set, axis_range=[-200, 200])
    hist1d(arr, axis_label_set, arr_range=[-200, 200], hist_range=[0, 0.5])
'''

def equalscale_scatter(arr_set, axis_label_set, axis_range=[-200, 200]):
    arr1, arr2 = arr_set
    x_label, y_label = axis_label_set
    
#    plt.figure()
    plt.scatter(arr1.reshape(-1), arr2.reshape(-1))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(axis_range)
    plt.ylim(axis_range)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
#    if saving_flag:
#        assert len(file_name) > 0
#        plt.savefig(file_name)
#    plt.show()
    

def equalscale_hist2d(arr_set, axis_label_set, axis_range=[-200, 200], hist_flag=False):
    arr1, arr2 = arr_set
    x_label, y_label = axis_label_set
    
    if hist_flag:
        hist2d, xedges, yedges = np.histogram2d(arr1.reshape(-1), arr2.reshape(-1), bins=[np.arange(axis_range[0], axis_range[1]+5, 5), np.arange(axis_range[0], axis_range[1]+5, 5)])
        hist2d = hist2d / len(arr1)
        plt.imshow(hist2d)
    else:
#    plt.figure()
        plt.hist2d(arr1.reshape(-1), arr2.reshape(-1), bins=[np.arange(axis_range[0], axis_range[1]+5, 5), np.arange(axis_range[0], axis_range[1]+5, 5)])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(axis_range)
    plt.ylim(axis_range)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.tight_layout()
#    if saving_flag:
#        assert len(file_name) > 0
#        plt.savefig(file_name)
#    plt.show()
    

#plt.figure()
#plt.hist2d(train_hr_feat_chl.reshape(-1), train_target_pred.reshape(-1), bins=[np.arange(-200, 205, 5), np.arange(-200, 205, 5)])
#plt.xlabel('predHR')
#plt.ylabel('HR')
#plt.xlim([-200, 200])
#plt.ylim([-200, 200])
#plt.gca().set_aspect('equal', adjustable='box')
#plt.colorbar()
#plt.tight_layout()
##plt.savefig(FIG_FOLDER + 'HR_vs_predHR_hist2d_round'+str(iter_idx)+'_xgbreg.jpg')
#plt.show()


def hist1d(arr, axis_label_set, arr_range=[-200, 200], hist_range=[0, 0.5], cond_flag=True, cond_num=0, linecolor='C0', linetype='-', linelabel=''):
    x_label, y_label = axis_label_set
    pdf, bin_edges, bin_centers = get_hist_info(arr, np.arange(arr_range[0], arr_range[1]+5, 5), cond_hist_flag=cond_flag, base_num=cond_num)
    
#    plt.figure()
    if len(linelabel) > 0:
        plt.plot(bin_centers, pdf, linetype, c=linecolor, label=linelabel)
    else:
        plt.plot(bin_centers, pdf, linetype, c=linecolor)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(hist_range)
    plt.tight_layout()
    print(np.sum(pdf))
    






#%%
# gaussian

def gene_gaussian_2d(w=5, sig=1.):
    """\
    creates gaussian kernel with side length `w` and a sigma of `sig`
    """
    ax = np.linspace(-(w - 1) / 2., (w - 1) / 2., w)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel# / np.sum(kernel)

# gaus_kernel_1 = gene_gaussian_2d(w=21, sig=10.0)
# print(np.max(gaus_kernel_1), np.sum(gaus_kernel_1))
# plt.figure()
# plt.imshow(gaus_kernel_1)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('./fig/gaus/patch21_gausSigma10.jpg')
# plt.show()

# gaus_kernel_2 = gene_gaussian_2d(w=11, sig=5.0)
# print(np.max(gaus_kernel_2), np.sum(gaus_kernel_2))
# plt.figure()
# plt.imshow(gaus_kernel_2)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('./fig/gaus/patch11_gausSigma5.jpg')
# plt.show()

# shape = (1, 1, 21, 21)
# a = np.arange(np.prod(shape)).reshape(shape)
# a_weighted = a*gaus_kernel_1
# a = np.squeeze(a)
# a_weighted = np.squeeze(a_weighted)



#%%
# 8 directions

def gene_horiver_directs(patch_size):
    '''
    patch3X3 --> r=1; patch5X5 --> r=2
    | False | True  | False 
    | True  | False | True 
    | False | True  | False 
    '''
    r = patch_size // 2 # e.g. 1
    h_list = []
    w_list = []
    for h in np.arange(r, dtype=int):
        h_list.append(h)
        w_list.append(r)
    for h in np.arange(r, dtype=int) + r + 1:
        h_list.append(h)
        w_list.append(r)

    for w in np.arange(r, dtype=int):
        h_list.append(r)
        w_list.append(w)
    for w in np.arange(r, dtype=int) + r + 1:
        h_list.append(r)
        w_list.append(w)

    return np.array(h_list), np.array(w_list)

# s = 7
# h, w = gene_horiver_directs(s)
# a = np.arange(s*s).reshape(s, s)
# print(a)
# a[h, w] = 100
# print(a)
#%%
def gene_diag_directs(patch_size):
    '''
    patch3X3 --> r=1; patch5X5 --> r=2
    | True  | False | True 
    | False | False | False 
    | True  | False | True 
    '''
    r = patch_size // 2 # e.g. 1
    h_list = []
    w_list = []
    for h in np.arange(r, dtype=int):
        h_list.append(h)
        w_list.append(h)
    for h in np.arange(r, dtype=int) + r + 1:
        h_list.append(h)
        w_list.append(h)

    for w in np.arange(r, dtype=int):
        h_list.append(w)
        w_list.append(patch_size - 1 - w)
    for w in np.arange(r, dtype=int) + r + 1:
        h_list.append(w)
        w_list.append(patch_size - 1 - w)

    return np.array(h_list), np.array(w_list)

# s = 7
# h, w = gene_diag_directs(s)
# a = np.arange(s*s).reshape(s, s)
# print(a)
# a[h, w] = 100
# print(a)

#%%
def gene_center_direct(patch_size):
    r = patch_size // 2 # e.g. 1
    h_list = [r]
    w_list = [r]
    return np.array(h_list), np.array(w_list)

# s = 7
# h, w = gene_center_direct(s)
# a = np.arange(s*s).reshape(s, s)
# print(a)
# a[h, w] = 100
# print(a)



#%%

def pixel_to_imageset(pixel_arr, ref_img_set):
    pixel_arr = np.array(pixel_arr).reshape(-1)

    if isinstance(ref_img_set, np.ndarray):
        shape = ref_img_set.shape
        pixel_reshaped = pixel_arr.reshape(shape)
    else:
        shape_list = [i.shape for i in ref_img_set]
        sample_num_list = [np.prod(i) for i in shape_list]
        cumu_sample_num_list = [0]
        cumu_sample_num_list.extend(np.cumsum(sample_num_list))
        pixel_reshaped = []
        for sing_shape, sample_idx_st, sample_idx_ed in zip(shape_list, cumu_sample_num_list[:-1], cumu_sample_num_list[1:]):
            pixel_reshaped.append(pixel_arr[sample_idx_st:sample_idx_ed].reshape(sing_shape))
    return pixel_reshaped

# s_list = [(2, 1, 3, 4), (3, 1, 2, 2), (4, 2, 3, 1)]
# ref = [np.arange(np.prod(i)).reshape(i) for i in s_list]
# p = np.concatenate([i.copy().reshape(-1) for i in ref])
# p_reshaped = pixel_to_imageset(p, ref)
# print([np.array_equal(i, j) for i,j in zip(ref, p_reshaped)])





#%%
# pure numerical cal

def mae_to_mse(mae_arr):
    return np.sum(np.power(mae_arr, 2)) * 1.0 / len(mae_arr)

def get_top_arg_idx_cumuthre(ref_arr, thre_value,
                             sorting_mode='min_first',
                             sel_mode='<=',
                             normalize_flag=True):
    '''
    sorting_nodee='min_first' or 'max_first'
    sel_mode='<=' or '>='
    '''
    if sorting_mode == 'min_first':
        sorting_order = np.argsort(ref_arr)
    elif sorting_mode == 'max_first':
        sorting_order = np.argsort(ref_arr*-1)

    ref_arr_sorted = ref_arr[sorting_order]
    if normalize_flag:
        ref_arr_sorted = ref_arr_sorted * 1.0 / np.sum(ref_arr_sorted)
    ref_arr_sorted_cumu = np.cumsum(ref_arr_sorted)

    if sel_mode == '<=':
        return sorting_order[ref_arr_sorted_cumu<=thre_value]
    elif sel_mode == '>=':
        return sorting_order[ref_arr_sorted_cumu>=thre_value]












#%%
'''
函数的参数: 
    可以用必选参数、默认参数、可变参数、关键字参数和命名关键字参数，
这5种参数都可以组合使用。
但是请注意，参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数。
https://www.liaoxuefeng.com/wiki/1016959663602400/1017261630425888





'''




