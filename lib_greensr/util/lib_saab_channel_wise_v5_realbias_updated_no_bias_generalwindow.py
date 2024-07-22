#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:15:43 2020
@author: Wei Wang
"""


'''
Revised on 05/23/20, Wei
1. each DC component goes to next stage, undifferentiated from AC components, otherwise reconstruction would be a problem
2. determine intermediate nodes by global energy percentage:
   --> for each node, track global energy percentage, track if splittable, --> by r.v. intermediate_ind
3. no max pooling
4. has reconstruction

IMPORTANT
can work for testing images (inference_chl_wise) with different spatial size from training images (used in multi_Saab_chl_wise)
'''

import numpy as np
from numpy import linalg as LA
from skimage.util.shape import view_as_windows
from skimage.measure import block_reduce
from itertools import product

# from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA as PCA

import gc

#%%
def window_process4(samples, kernel_size, stride, dilate, padFlag=True):
    '''
    @ Args:
        samples(np.array) [num, c, h, w]
        kernel_size [kernel_size_h, kernel_size_w]
    @ Returns:
        patches(np.array) [n, h, w, c]
    '''
    kernel_size_h, kernel_size_w = kernel_size

    if padFlag:
        samples2 = np.pad(samples,((0,0),(0,0),(int(kernel_size_h/2),int(kernel_size_h/2)),(int(kernel_size_w/2),int(kernel_size_w/2))),'reflect')
    else:
        samples2 = samples
#    print('-- window_process4 cuboid after patching', samples2.shape) 
    n, c, h, w= samples2.shape
    output_h = (h - kernel_size_h) // stride + 1
    output_w = (w - kernel_size_w) // stride + 1
    patches = view_as_windows(np.ascontiguousarray(samples2), (1, c, kernel_size_h, kernel_size_w), step=(1, c, stride, stride))
#    patches = view_as_windows(np.ascontiguousarray(samples2), (2, c//2, kernel_size, kernel_size), step=(2, c//2, stride, stride))
#    print('-- window_process4 cuboid after view_as_windows', patches.shape) 
    # --> [output_n=n, output_c==1, output_h, output_w, 4d_kernel_n==1, 4d_kernel_c==c, 4d_kernel_h, 4d_kernel_w]
    patches = patches.reshape(n,output_h, output_w, c, kernel_size_h, kernel_size_w)
    assert dilate >=1
    patches = patches[:,:,:,:,::dilate,::dilate]   # arbitary dilate, not necessary to take 9 positions
    patches = patches.reshape(n,output_h,output_w,-1)   # [n,output_h,output_w,c*9]
    return patches  # [n,output_h,output_w,c*9]

# def window_process4_ori(samples, kernel_size, stride, dilate, padFlag=True):
#     '''
#     @ Args:
#         samples(np.array) [num, c, h, w]
#     @ Returns:
#         patches(np.array) [n, h, w, c]
#     '''
#     if padFlag:
#         samples2 = np.pad(samples,((0,0),(0,0),(int(kernel_size/2),int(kernel_size/2)),(int(kernel_size/2),int(kernel_size/2))),'reflect')
#     else:
#         samples2 = samples
#     n, c, h, w= samples2.shape
#     output_h = (h - kernel_size) // stride + 1
#     output_w = (w - kernel_size) // stride + 1
#     patches = view_as_windows(np.ascontiguousarray(samples2), (1, c, kernel_size, kernel_size), step=(1, c, stride, stride))
# #    patches = view_as_windows(np.ascontiguousarray(samples2), (2, c//2, kernel_size, kernel_size), step=(2, c//2, stride, stride))
# #    print('-- window_process4 cuboid after view_as_windows', patches.shape) 
#     # --> [output_n, output_c, output_h, output_w, 4d_kernel_n, 4d_kernel_c, 4d_kernel_h, 4d_kernel_w]
#     # i.e. [n, 1, output_h, output_w, 1, c, kernel_size, kernel_size]
#     patches = patches.reshape(n,output_h, output_w, c, kernel_size, kernel_size)
#     assert dilate >=1
#     patches = patches[:,:,:,:,::dilate,::dilate]   # arbitary dilate, not necessary to take 9 positions
# #    patches_new = patches.reshape(n,output_h,output_w,-1)   # [n,output_h,output_w,c*9]
#     return patches  # (n,output_h, output_w, c, kernel_size//dilate, kernel_size//dilate)




# arbitary dilate, not necessary to take 9 positions
# Reconstruct process
def unpatchify(patches, imsize, stride, dilate, kernel_size):
    '''
    work for single channel (mean pooling for one pixel)
    @Args:
        patches(4d np.array, np.float64)[out_h,out_w,kernel_size,kernel_size]: 
            current stage patches for previous stage reconstruction
        imsize(2d tuple, int): image (h,w) for previous(target) layer, e.g. (200,100)
    @Returns:
        recImg_chn(2d np.array, np.float64)[imsize[0], imsize[1]]
    '''
    #print('--unpatchify: patches.shape: ', patches.shape)
    (i_h, i_w) = imsize # previous layer h, w
#    print('previous layer h, w', imsize)
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.ones(imsize, dtype=patches.dtype)
    n_h, n_w = patches.shape[:2]  # (output_h, output_w, 3, 3)
    
    s_h = stride
    s_w = stride

    for i, j in product(range(n_h), range(n_w)):
        image[(i*s_h):(i*s_h)+kernel_size:dilate, (j*s_w):(j*s_w)+kernel_size:dilate] += patches[i,j]
        divisor[(i*s_h):(i*s_h)+kernel_size:dilate, (j*s_w):(j*s_w)+kernel_size:dilate] += 1
    
    divisor[divisor>1] -= 1
    recImg_chn = image * 1.0 / divisor
    return recImg_chn


def inverse_window_process4(samples, kernel_size, stride, dilate, outputHW, preHW, padFlag=True):
    '''
    @Args:
        samples(np.array, np.float64)[n*out_h*out_w, c*kernel_size_valid*kernel_size_valid]
        outputHW(list, int): h,w for current stage
        preHW(list, int): h,w for previous (target)stage
    @Returns:
        newImgs(np.array, np.float64) [n, c_pre, h_pre, w_pre], c_pre usually is 1
    '''
    output_h = outputHW[0]
    output_w = outputHW[1]
    pre_h = preHW[0]
    pre_w = preHW[1]
#    print('--inverse_window_process: patches_input.shape: ', samples.shape)  
    # 2d, [n*out_h*out_w, c*kernel_size*kernel_size]
    
    # 2d -> view_as_windows results
    n = int(samples.shape[0] / output_h / output_w)
    if dilate == 1:
        patches = samples.reshape(n,output_h, output_w, -1, kernel_size, kernel_size)
    else:
        kernel_size_valid = int((kernel_size-1)/dilate)+1
        patches = samples.reshape(n,output_h, output_w, -1, kernel_size_valid, kernel_size_valid)  
    #c = patches.shape[3]
#    print('--inverse_window_process: patches_breakAxes.shape: ', patches.shape)
    del samples
    gc.collect()
    
    # inverse view_as_windows
    patches = np.moveaxis(patches, 3, 1)
    # (n, c, output_h, output_w, kernel_size, kernel_size)
    # move "c" axis above, since need to do channel-wise averaging for each pixel w.r.t. all pred.
#    print('--inverse_window_process: output_h output_w stride kernel_size: ', output_h, output_w, stride, kernel_size)
    h = (output_h-1)*stride+kernel_size # padded_h (or real effective h cropped for deeper layer) at previous layer, NOT pre_h
    w = (output_w-1)*stride+kernel_size # padded_w (or real effective h cropped for deeper layer) at previous layer, NOT pre_w
#    print('--inverse_window_process: effective previous h,w: ', h, w)
#    print('cheching', h, output_h, stride, kernel_size)
    newImgs = []
    # arbitary dilate, not necessary to take 9 positions
    
#    print('+++padFlag+++', padFlag)
    if padFlag:
        pad = int(kernel_size/2)  # deal with border padding in window_process_4
        for eachImg in patches:
            newImg = np.array([unpatchify(eachChn, (h,w), stride, dilate, kernel_size)[pad:pad+pre_h, pad:pad+pre_w] for eachChn in eachImg])
            # crop out the outside padding
            newImgs.append(newImg)
        newImgs = np.array(newImgs)
        
    else:
#        print('come here')
        for eachImg in patches:
            newImg = np.array([unpatchify(eachChn, (h,w), stride, dilate, kernel_size) for eachChn in eachImg])
#            print('newImg', newImg.shape)
            # crop out the outside padding
            newImgs.append(newImg)
        newImgs = np.array(newImgs)
    
    return newImgs # (n, 1, h_pre, w_pre)


def remove_mean(features, axis):
    '''
    Remove the dataset mean.
    :param features [num_samples,...]
    :param axis the axis to compute mean
    '''
    feature_mean = np.mean(features, axis=axis, keepdims=True)
    feature_remove_mean = features - feature_mean
    return feature_remove_mean, feature_mean


def find_kernels_pca(samples, num_kernel, rank_ac, recFlag):
    '''
    Train PCA based on the provided samples.
    If num_kernel can be int (w.r.t. absolute channel number) or float (w.r.t. preserved energy) type.
    If want to keep all energy, feed string 'full' for num_kernel. 
    @Args:
        samples(numpy.array, numpy.float64), [num_samples, num_feature_dimension]: 2D AC sample matrix for PCA
        num_kernel(int, float, or string 'full'): num of channels (or energy percentage) to be preserved
    @Returns:
        pca(sklearn.decomposition.PCA object): trained PCA object
        num_components(int): number of valid AC components, considering constrains of num_kernel(int or float by energy) and rank_ac (if recFlag==True)
        intermediate_ind(1d np.array): conditional energy of valid AC components
    '''
    print('find_kernels_pca/samples.shape:', samples.shape)
    # determination by number of kernels
    if num_kernel == 'full' or num_kernel > min(samples.shape):
        pca = PCA()#svd_solver = 'randomized')
    # determination by energy threshold
    else:
        pca = PCA(n_components = num_kernel)#, svd_solver = 'randomized')
        #pca = IncrementalPCA(n_components = num_kernel, batch_size=100)
    pca.fit(samples)
    
    num_components = len(pca.components_)
    # consider matrix rank, control orthonormal basis
    if recFlag: 
        num_components = min(pca.n_components_, rank_ac)
    
#    energy_perc = pca.explained_variance_ratio_[:num_components]
#    intermediate_ind = energy_perc > ene_thre
#    print('energy_perc', energy_perc)
#    eneP = np.cumsum(pca.explained_variance_ratio_[:num_components])[-1]
    intermediate_ind = pca.explained_variance_ratio_[:num_components]
    return pca, num_components, intermediate_ind


def conv_with_bias(layer_idx, channel_idx, 
                   sample_init,
                   bias_pre, param_dict, 
                   train_pack=None, train_flag=True):
    '''
    work for each intermediate node (one channel)
    determine intermediate and leaf at each stage when training, set indicator in intermediate_ind_channel to be zero if < energy_thre
    @ Args:
        sample_init: [n,1,h,w]
    @ Returns:
        sample_channel: [n, c, h_new, w_new]
    '''
    if train_flag:
        energy_parent = train_pack
    
    # add bias
    sample = sample_init# + bias_pre
    
    sample_patches = window_process4(sample, 
                                     param_dict['kernel_size'][layer_idx], 
                                     param_dict['stride'][layer_idx], 
                                     param_dict['dilate'][layer_idx], 
                                     padFlag=param_dict['padFlag'][layer_idx])  # collect neighbor info
    # --> [n,h,w,c]
    [output_n, output_h, output_w, conved_c] = sample_patches.shape
#    print('- conv_with_bias Sample/cuboid_current.shape:', sample_patches.shape)
    
    # Flatten
    sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])
#    sample_patches = sample_patches[interest_sample_idx]
#    print('sample norm', LA.norm(sample_patches, axis=1))
#    print('- conv_with_bias Sample/flatten.shape:', sample_patches.shape)
    
    # get dc response
    sample_patches_ac, dc = remove_mean(sample_patches, axis=1)  # Remove patch mean
    del sample_patches
    gc.collect()
    
    # get ac response (& filters)
    ## Remove feature mean (Set E(X)=0 for each dimension)
    if train_flag:
        sample_patches_centered, feature_expectation = remove_mean(sample_patches_ac, axis=0)
        param_dict['Layer_%d/Slice_%d/feature_expectation' % (layer_idx, channel_idx)] = feature_expectation
    else:
        feature_expectation = param_dict['Layer_%d/Slice_%d/feature_expectation' % (layer_idx, channel_idx)]
        sample_patches_centered = sample_patches_ac - feature_expectation
    del sample_patches_ac
    gc.collect()
    
    # Compute PCA kernel for AC components
    if train_flag:
        rank_ac = np.linalg.matrix_rank(sample_patches_centered)
        ac_pca, num_acKernel, intermediate_ind_channel = find_kernels_pca(sample_patches_centered, param_dict['num_kernels'][layer_idx], rank_ac, param_dict['recFlag'])
        param_dict['Layer_%d/Slice_%d/rank_ac' % (layer_idx, channel_idx)] = rank_ac
        param_dict['Layer_%d/Slice_%d/num_acKernel' % (layer_idx, channel_idx)] = num_acKernel
        param_dict['Layer_%d/Slice_%d/ac_kernels' % (layer_idx, channel_idx)] = ac_pca.components_[:num_acKernel]
        
        # adjust intermediate_ind_channel for dc component, that keep dc component as one leaf node
        intermediate_ind_channel = intermediate_ind_channel.tolist()
        intermediate_ind_channel.insert(0, 1)
#        print('intermediate_ind_channel', intermediate_ind_channel)
        intermediate_ind_channel = np.array(intermediate_ind_channel) * energy_parent
#        intermediate_ind_channel[intermediate_ind_channel < param_dict['energy_perc_thre']] = 0
#        print('intermediate_ind_channel', intermediate_ind_channel)
        param_dict['Layer_%d/Slice_%d/intermediate_ind' % (layer_idx, channel_idx)] = intermediate_ind_channel
#    else:
    ac_kernels = param_dict['Layer_%d/Slice_%d/ac_kernels' % (layer_idx, channel_idx)]

    # transform
    ## calc DC response: i.e. dc
    ## calc AC response
    num_channels = ac_kernels.shape[-1]
    dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
    ac = np.matmul(sample_patches_centered, np.transpose(ac_kernels))
    transformed = np.concatenate([dc, ac], axis=1)
    del sample_patches_centered, ac_kernels
    gc.collect()
#    print('- conv_with_bias Sample/cuboid_transformed_2d.shape:', transformed.shape)
    
    if train_flag:
        # Compute bias term for next layer
        # bias = np.max(LA.norm(transformed, axis=1)) * np.ones(transformed.shape[-1])
        bias = np.zeros(transformed.shape[-1])

    # Reshape back as a 4-D feature map
    feature_channel = np.array(transformed)
    sample_channel = feature_channel.reshape((output_n, output_h, output_w, -1)) # -> [num, h, w, c]
    sample_channel = np.moveaxis(sample_channel, 3, 1) # -> [num, c, h, w]
    
    if train_flag:
        return sample_channel, bias
    else:
        return sample_channel #, intermediate_ind_channel#, output_h, output_w



def decov_with_bias(layer_idx, channel_idx,
                    sample_channel, bias_pre, param_dict):
    '''
    @ Args:
        sample_channel [n, c, h, w]
    @ Returns:
        rec_channel [n, c, h_pre, w_pre], c_pre is usually 1
    '''
    sample_channel = np.moveaxis(sample_channel, 1,3) # -> [num, h, w, c]
    h = sample_channel.shape[1]
    w = sample_channel.shape[2]
    h_pre = param_dict['Layer_%d/h' % (layer_idx-1)]
    w_pre = param_dict['Layer_%d/w' % (layer_idx-1)]
    
    # Flatten
    sample_channel = sample_channel.reshape(-1, sample_channel.shape[-1])
#    print('Sample/sample_channel_flatten.shape:', sample_channel.shape)
    
    # get kernels
    ac_kernel = param_dict['Layer_%d/Slice_%d/ac_kernels' % (layer_idx, channel_idx)].astype(np.float64)
    dc_comp = sample_channel[:, 0:1] * 1.0 / np.sqrt(ac_kernel.shape[-1])
    ac_comp = np.matmul(sample_channel[:, 1:], ac_kernel)
#    print('dc_comp', dc_comp.shape)
#    print('ac_comp', ac_comp.shape)
    
#    np.save('dc_comp_rec', dc_comp)
#    np.save('ac_comp_rec', ac_comp)
    
    feature_expectation = param_dict['Layer_%d/Slice_%d/feature_expectation' % (layer_idx, channel_idx)].astype(np.float64)
#    print('feature_expectation', feature_expectation.shape)
    rec_channel = dc_comp + (ac_comp + feature_expectation)
    del feature_expectation
    gc.collect()
#    print('Sample/cuboid_transformed_2d.shape:', rec_channel.shape)
    
    # unpatchify: current 2d -> previous 4d (mean pooling for each pixel)
    rec_channel = inverse_window_process4(rec_channel,
                                          param_dict['kernel_size'][layer_idx], 
                                          param_dict['stride'][layer_idx], 
                                          param_dict['dilate'][layer_idx],
                                          outputHW = [h,w],
                                          preHW = [h_pre, w_pre],
                                          padFlag = param_dict['padFlag'][layer_idx])
    
    # check
    # rec_channel -= bias_pre
#    print('Sample/rec_channel.shape:', rec_channel.shape)
    
    return rec_channel #[n, 1, h_pre, w_pre]


#%%
# def conv_with_bias_selected(layer_idx, channel_idx, 
#                    sample_init, interest_sample_idx, 
#                    bias_pre, param_dict, 
#                    train_pack=None, train_flag=True):
#     '''
#     work for each intermediate node (one channel)
#     determine intermediate and leaf at each stage when training, set indicator in intermediate_ind_channel to be zero if < energy_thre
#     @ Args:
#         sample_init: [n,1,h,w]
#     @ Returns:
#         sample_channel: [n, c, h_new, w_new]
#     '''
#     if train_flag:
#         energy_parent = train_pack
    
#     # add bias
#     sample = sample_init + bias_pre
    
#     sample_patches = window_process4(sample, 
#                                      param_dict['kernel_size'][layer_idx], 
#                                      param_dict['stride'][layer_idx], 
#                                      param_dict['dilate'][layer_idx], 
#                                      padFlag=param_dict['padFlag'][layer_idx])  # collect neighbor info
#     # --> [n,h,w,c]
#     [output_n, output_h, output_w, conved_c] = sample_patches.shape
# #    print('- conv_with_bias Sample/cuboid_current.shape:', sample_patches.shape)
    
#     # Flatten
#     sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])
# #    sample_patches = sample_patches[interest_sample_idx]
# #    print('sample norm', LA.norm(sample_patches, axis=1))
# #    print('- conv_with_bias Sample/flatten.shape:', sample_patches.shape)
    
#     # get dc response
#     sample_patches_ac_all, dc = remove_mean(sample_patches, axis=1)  # Remove patch mean
#     sample_patches_ac = sample_patches_ac_all[interest_sample_idx]   # selected samples
#     del sample_patches
#     gc.collect()
    
#     # get ac response (& filters)
#     ## Remove feature mean (Set E(X)=0 for each dimension)
#     if train_flag:
#         sample_patches_centered, feature_expectation = remove_mean(sample_patches_ac, axis=0)
#         param_dict['Layer_%d/Slice_%d/feature_expectation' % (layer_idx, channel_idx)] = feature_expectation
#         del sample_patches_ac
#         gc.collect()
    
#         # Compute PCA kernel for AC components
# #    if train_flag:
#         rank_ac = np.linalg.matrix_rank(sample_patches_centered)
#         ac_pca, num_acKernel, intermediate_ind_channel = find_kernels_pca(sample_patches_centered, param_dict['num_kernels'][layer_idx], rank_ac, param_dict['recFlag'])
#         param_dict['Layer_%d/Slice_%d/rank_ac' % (layer_idx, channel_idx)] = rank_ac
#         param_dict['Layer_%d/Slice_%d/num_acKernel' % (layer_idx, channel_idx)] = num_acKernel
#         param_dict['Layer_%d/Slice_%d/ac_kernels' % (layer_idx, channel_idx)] = ac_pca.components_[:num_acKernel]
        
#         # adjust intermediate_ind_channel for dc component, that keep dc component as one leaf node
#         intermediate_ind_channel = intermediate_ind_channel.tolist()
#         intermediate_ind_channel.insert(0, 1)
# #        print('intermediate_ind_channel', intermediate_ind_channel)
#         intermediate_ind_channel = np.array(intermediate_ind_channel) * energy_parent
# #        intermediate_ind_channel[intermediate_ind_channel < param_dict['energy_perc_thre']] = 0
# #        print('intermediate_ind_channel', intermediate_ind_channel)
#         param_dict['Layer_%d/Slice_%d/intermediate_ind' % (layer_idx, channel_idx)] = intermediate_ind_channel
# #    else:

#     feature_expectation = param_dict['Layer_%d/Slice_%d/feature_expectation' % (layer_idx, channel_idx)]
#     sample_patches_centered = sample_patches_ac_all - feature_expectation
#     ac_kernels = param_dict['Layer_%d/Slice_%d/ac_kernels' % (layer_idx, channel_idx)]

#     # transform
#     ## calc DC response: i.e. dc
#     ## calc AC response
#     num_channels = ac_kernels.shape[-1]
#     dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
#     ac = np.matmul(sample_patches_centered, np.transpose(ac_kernels))
#     transformed = np.concatenate([dc, ac], axis=1)
#     del sample_patches_centered, ac_kernels
#     gc.collect()
#     print('Sample/cuboid_transformed_2d.shape:', transformed.shape)
    
#     if train_flag:
#         # Compute bias term for next layer
#         bias = np.max(LA.norm(transformed, axis=1)) * np.ones(transformed.shape[-1])
    
#     # Reshape back as a 4-D feature map
#     feature_channel = np.array(transformed)
#     sample_channel = feature_channel.reshape((output_n, output_h, output_w, -1)) # -> [num, h, w, c]
#     sample_channel = np.moveaxis(sample_channel, 3, 1) # -> [num, c, h, w]
    
#     if train_flag:
#         return sample_channel, bias
#     else:
#         return sample_channel #, intermediate_ind_channel#, output_h, output_w



# def decov_with_bias_selected(layer_idx, channel_idx,
#                     sample_channel, interest_sample_idx, bias_pre, param_dict):
#     '''
#     @ Args:
#         sample_channel [n, c, h, w]
#     @ Returns:
#         rec_channel [n, c, h_pre, w_pre], c_pre is usually 1
#     '''
#     sample_channel = np.moveaxis(sample_channel, 1,3) # -> [num, h, w, c]
#     h = sample_channel.shape[1]
#     w = sample_channel.shape[2]
#     h_pre = param_dict['Layer_%d/h' % (layer_idx-1)]
#     w_pre = param_dict['Layer_%d/w' % (layer_idx-1)]
    
#     # Flatten
#     sample_channel = sample_channel.reshape(-1, sample_channel.shape[-1])
#     print('Sample/sample_channel_flatten.shape:', sample_channel.shape)
    
#     # get kernels
#     ac_kernel = param_dict['Layer_%d/Slice_%d/ac_kernels' % (layer_idx, channel_idx)].astype(np.float64)
#     dc_comp = sample_channel[:, 0:1] * 1.0 / np.sqrt(ac_kernel.shape[-1])
#     ac_comp = np.matmul(sample_channel[:, 1:], ac_kernel)
# #    print('dc_comp', dc_comp.shape)
# #    print('ac_comp', ac_comp.shape)
    
# #    np.save('dc_comp_rec', dc_comp)
# #    np.save('ac_comp_rec', ac_comp)
    
#     feature_expectation = param_dict['Layer_%d/Slice_%d/feature_expectation' % (layer_idx, channel_idx)].astype(np.float64)
#     print('feature_expectation', feature_expectation.shape)
#     rec_channel = dc_comp + (ac_comp + feature_expectation)
#     pesudo_sample_idx = np.array(list(set(np.arange(len(rec_channel))) - set(interest_sample_idx)))
#     rec_channel[pesudo_sample_idx] = 0
# #    rec_channel[interest_sample_idx] = 0
#     del feature_expectation
#     gc.collect()
#     print('Sample/cuboid_transformed_2d.shape:', rec_channel.shape)
    
#     # unpatchify: current 2d -> previous 4d (mean pooling for each pixel)
#     rec_channel = inverse_window_process4(rec_channel,
#                                           param_dict['kernel_size'][layer_idx], 
#                                           param_dict['stride'][layer_idx], 
#                                           param_dict['dilate'][layer_idx],
#                                           outputHW = [h,w],
#                                           preHW = [h_pre, w_pre],
#                                           padFlag = param_dict['padFlag'][layer_idx])
    
#     # check
#     rec_channel -= bias_pre
#     print('Sample/rec_channel.shape:', rec_channel.shape)
    
#     return rec_channel #[n, 1, h_pre, w_pre]




# def decov_with_bias_selected_simplified(layer_idx, channel_idx,
#                     sample_channel, interest_sample_idx_list, bias_pre, param_dict_list):
#     '''
#     @ Args:
#         sample_channel [n, c, h, w]
#     @ Returns:
#         rec_channel [n, c, h_pre, w_pre], c_pre is usually 1
#     '''
#     param_dict_one = param_dict_list[0]
#     sample_channel = np.moveaxis(sample_channel, 1,3) # -> [num, h, w, c]
#     h = sample_channel.shape[1]
#     w = sample_channel.shape[2]
#     h_pre = param_dict_one['Layer_%d/h' % (layer_idx-1)]
#     w_pre = param_dict_one['Layer_%d/w' % (layer_idx-1)]
    
#     # Flatten
#     sample_channel = sample_channel.reshape(-1, sample_channel.shape[-1])
#     print('Sample/sample_channel_flatten.shape:', sample_channel.shape)
    
#     rec_channel = []
#     for interest_sample_idx, param_dict in zip(interest_sample_idx_list, param_dict_list):
#         sample_channel_subspace = sample_channel[interest_sample_idx]
#         # get kernels
#         ac_kernel = param_dict['Layer_%d/Slice_%d/ac_kernels' % (layer_idx, channel_idx)].astype(np.float64)
#         ac_comp = np.matmul(sample_channel_subspace[:, 1:], ac_kernel)
#         feature_expectation = param_dict['Layer_%d/Slice_%d/feature_expectation' % (layer_idx, channel_idx)].astype(np.float64)
#         print(sample_channel_subspace.shape)
#         print('feature_expectation', feature_expectation.shape)
#         dc_comp = sample_channel_subspace[:, 0:1] * 1.0 / np.sqrt(ac_kernel.shape[-1])
# #    print('dc_comp', dc_comp.shape)
# #    print('ac_comp', ac_comp.shape)
# #    np.save('dc_comp_rec', dc_comp)
# #    np.save('ac_comp_rec', ac_comp)
#         rec_channel_subspace = dc_comp + (ac_comp + feature_expectation)
#     #    rec_channel[interest_sample_idx] = 0
#         del feature_expectation
#         gc.collect()
#         rec_channel.append(rec_channel_subspace)
#     sorting_order = np.argsort(np.concatenate(interest_sample_idx_list, axis=0))
#     rec_channel = (np.concatenate(rec_channel, axis=0))[sorting_order]
#     print('Sample/cuboid_transformed_2d.shape:', rec_channel.shape)
    
#     # unpatchify: current 2d -> previous 4d (mean pooling for each pixel)
#     rec_channel = inverse_window_process4(rec_channel,
#                                           param_dict_one['kernel_size'][layer_idx], 
#                                           param_dict_one['stride'][layer_idx], 
#                                           param_dict_one['dilate'][layer_idx],
#                                           outputHW = [h,w],
#                                           preHW = [h_pre, w_pre],
#                                           padFlag = param_dict_one['padFlag'][layer_idx])
    
#     # check
#     rec_channel -= bias_pre
#     print('Sample/rec_channel.shape:', rec_channel.shape)
    
#     return rec_channel #[n, 1, h_pre, w_pre]



#%%
def multi_saab_chl_wise(sample_images_init, 
                        stride, 
                        kernel_size, 
                        dilate,
                        num_kernels,
                        energy_perc_thre, # int(num_nodes) or float(presE, var) for nodes growing down
                        padFlag,
                        recFlag = True,
                        collectFlag = False,
                        init_bias = 0):
    '''
    Do the Saab "training".
    the length should be equal to kernel_sizes.
    @Args:
        sample_images_init(np.array, np.float64),[num_images, channel, height, width]
        stride(list, int): stride for each stage
        kernel_size(list, int): subspace size for each stage, the length defines how many stages conducted
        dilate(list, int): dilate for each stage
        num_kernels(list, float or int): used in pca for number of valid AC components preservation. int w.r.t. number, float w.r.t. cumulative energy
        energy_perc_thre(float < 1): energy percantage threshold for a single node, if global ene perc < eneray_perc_threshold, then stop splitting
                                     only work for AC nodes
        padFlag(list, bool): indicator of padding for each stage
        recFlag(boolean): If true, the AC kernel number is limited by the AC feature matrix rank. This is necessary for reconstruction.
    @Returns: 
        pca_params(dict): residue saab transform parameters
        feature_all: a list of feature (all nodes) at each stage
        sample_images[num_images, channel, height, width]: feature at last stage
    '''
    num_layers = len(kernel_size)
    pca_params = {}
    
#    pca_params['num_layers'] = num_layers
    pca_params['stride'] = stride
    pca_params['kernel_size'] = kernel_size
    pca_params['dilate'] = dilate
    pca_params['num_kernels'] = num_kernels
    pca_params['energy_perc_thre'] = energy_perc_thre
    pca_params['padFlag'] = padFlag
    pca_params['recFlag'] = recFlag
    
    sample_images = sample_images_init.copy()
    pca_params['Layer_-1/h'] = sample_images.shape[-2]
    pca_params['Layer_-1/w'] = sample_images.shape[-1]
    pca_params['Layer_-1/intermediate_ind'] = np.ones(sample_images.shape[1])
#    pca_params['Layer_-1/Slice_0/bias'] = init_bias
    pca_params['Layer_-1/bias'] = np.array([init_bias])
    feature_all = []
    
    # for each layer
    i_valid = -1
    for i in range(num_layers):
        print('--------stage %d --------' % i)
        print('Sample/cuboid_previous.shape:', sample_images.shape)
        
        # prepare for next layer
        intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % (i-1)]  # float list
        bias_layer = pca_params['Layer_%d/bias' % (i-1)]
        
        if np.sum(intermediate_ind_layer) > 0:
            i_valid += 1
            intermediate_index_layer = np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0]
            sample_images = sample_images[:, intermediate_index_layer, :, :]
#            print('intermediate_ind_layer:', intermediate_index_layer)
            print('Sample/cuboid_forNextStage.shape:', sample_images.shape)
            
            # Maxpooling
#            if i > 0:
#                sample_images = block_reduce(sample_images, (1, 1, 2, 2), np.max)
#                print('Sample/max_pooling.shape:', sample_images.shape)
                
            [n, num_node, h, w] = sample_images.shape
            
            feature_layer = [] # (unsorteed) features in this new hop from previous sorted feat.
            intermediate_ind_layer_new = []
            num_node_list = []
            bias_layer_new = []
            
#            print(bias_layer[intermediate_index_layer])
            
            # for each channel
            for channel_idx, intermediate_ene, bias in zip(np.arange(num_node), intermediate_ind_layer[intermediate_index_layer], bias_layer[intermediate_index_layer]):  # only partial samples going down
#                print('##### intermediate channel %d #####' % channel_idx)
                # Create patches in this channel
                sample_chl = sample_images[:, channel_idx:channel_idx+1, :, :]
#                sample_chl = np.expand_dims(sample_chl, axis=1)
                # extract filter and store in pca_params
                feature_channel, bias_chl = conv_with_bias(i, channel_idx,
                                                 sample_chl, bias,
                                                 pca_params, 
                                                 train_pack=intermediate_ene, train_flag=True)
                intermediate_ind_channel = pca_params['Layer_%d/Slice_%d/intermediate_ind' % (i, channel_idx)]
                # collect all slice feature in this layer
                feature_layer.append(feature_channel)
                intermediate_ind_layer_new.append(intermediate_ind_channel)
                num_node_list.append(len(intermediate_ind_channel)) # include DC, include small AC
                bias_layer_new.append(bias_chl)
                # end with all valid parent nodes (ALL CHANNELS)
            
            # one layer summary
            sample_images = np.concatenate(feature_layer, axis=1)  # only control features for further growing
            intermediate_ind_layer_new = np.concatenate(intermediate_ind_layer_new, axis=0)
            num_node_list = np.array(num_node_list)
            bias_layer_new = np.concatenate(bias_layer_new)
            pca_params['Layer_%d/intermediate_ind' % i] = intermediate_ind_layer_new # float list
            pca_params['Layer_%d/num_node_list' % i] = num_node_list
            pca_params['Layer_%d/bias' % i] = bias_layer_new
            pca_params['Layer_%d/h'% i] = sample_images.shape[-2]
            pca_params['Layer_%d/w'% i] = sample_images.shape[-1]
            print('Sample/cuboid_toNextStage.shape:', sample_images.shape)
#            print()
            
            if collectFlag:
                feature_all.append(sample_images)
            # end with ONE LAYER conditionally
        # end with ONE LAYER
        
#    #### adjust last layer intermediate_ind to be all none zero
#    intermediate_ind_layer_last = pca_params['Layer_%d/intermediate_ind' % i_valid]
#    intermediate_ind_layer_last[intermediate_ind_layer_last==0] = -1
#    pca_params['Layer_%d/intermediate_ind' % i_valid] = intermediate_ind_layer_last
    pca_params['num_layers'] = i_valid+1
    return pca_params, feature_all, sample_images



# feature generation
def inference_chl_wise(pca_params_init,
                        sample_images_init, intermediate_flag,
                        current_stage, target_stage,
                        collectFlag=True):
    '''
    Based on pca_params, generate saab feature for target_stage from current_stage
    stage index: -1(initial image), 0(1st saab), 1(2nd saab), ...
    @Args:
        pca_params(dict)
        sample_images_init(np.array, np.float64)[num,c,h,w]: 4-D feature of current_stage, of only intermediate nodes or all nodes (cannot be only leaf nodes)
        intermediate_flag(bool): [True] only gives intermediate nodes, or [False] gives all nodes
        current_stage(int): current stage index
        target_stage(int): target stage index
    @Returns:
        pca_params: modified c/w Saab filters on spatial sizes based on testing data, spectral operation is preserverd
        feature_all(list, np.array, np.float64)[sample_images0, sample_images1, ...]: 
            4-D feature of each stage in the processtarget_stage (exclude current_stage, include current_stage)
        sample_images(np.array, np.float64)[num, c, h, w]: 4-D feature of target_stage
    '''
    pca_params = pca_params_init.copy()
    
    # custermize spatial info
    sample_images = sample_images_init.copy()
    pca_params['Layer_%d/h'% current_stage] = sample_images.shape[-2]
    pca_params['Layer_%d/w'% current_stage] = sample_images.shape[-1]

    intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % current_stage] # float list
    num_intermediate_init = len(np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0])
    if intermediate_flag:
        assert num_intermediate_init == sample_images.shape[1]
    else:
        assert len(intermediate_ind_layer) == sample_images.shape[1]
    
    feature_all = []
    # for each layer
    for i in range(current_stage+1, target_stage+1, 1):
#        print('--------stage %d --------' % i)
#        print('Sample/cuboid_previous.shape:', sample_images.shape)
#        
        intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % (i-1)] # float list
        intermediate_index_layer = np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0]
        bias_layer = pca_params['Layer_%d/bias' % (i-1)]
        # prepare for next layer, take out intermediate nodes
        if i == current_stage+1 and (intermediate_flag): # current_stage, only gives intermediate nodes
            pass
        else: # else gives all nodes
            sample_images = sample_images[:, intermediate_index_layer, :, :]
#            print('intermediate_ind_layer:', intermediate_index_layer)
#            print('Sample/cuboid_forNextStage.shape:', sample_images.shape)
        
        if sample_images.shape[1] > 0:
            # Maxpooling
#            if i > 0:
#                sample_images = block_reduce(sample_images, (1, 1, 2, 2), np.max)
#                print('Sample/max_pooling.shape:', sample_images.shape)
                
            [n, num_node, h, w] = sample_images.shape
            feature_layer = []
    
            for channel_idx, bias in zip(np.arange(num_node),bias_layer[intermediate_index_layer]):
#                print('##### intermediate channel %d #####' % channel_idx)
                # Create patches in this slice
                sample_chl = sample_images[:, channel_idx, :, :]
                sample_chl = np.expand_dims(sample_chl, axis=1)
                
                feature_channel = conv_with_bias(i, channel_idx,
                                                 sample_chl, bias,
                                                 pca_params, 
#                                                 [kernel_size[i], stride[i], dilate[i], num_kernels[i], padF],
                                                 train_flag = False)
                # collect all slice feature in this layer
                feature_layer.append(feature_channel)
                
                # end with all valid parent nodes (ALL CHANNELS)
                
            # one layer summary
            sample_images = np.concatenate(feature_layer, axis=1)  # only control features for further growing
            pca_params['Layer_%d/h'% i] = sample_images.shape[-2]
            pca_params['Layer_%d/w'% i] = sample_images.shape[-1]
#            print('Sample/cuboid_toNextStage.shape:', sample_images.shape)
#            print()
            
            if collectFlag:
                feature_all.append(sample_images)
            # end with ONE LAYER conditionally
        # end with ONE LAYER
        
    ### last layer  intermediate_ind is not used
    return pca_params, feature_all, sample_images #, sample_images # last stage feature [num, c, h, w]



def inference_chl_wise_simplified(pca_params_init,
                        sample_images_init, intermediate_flag,
                        current_stage, target_stage):
    '''
    Based on pca_params, generate saab feature for target_stage from current_stage
    stage index: -1(initial image), 0(1st saab), 1(2nd saab), ...
    @Args:
        pca_params(dict)
        sample_images_init(np.array, np.float64)[num,c,h,w]: 4-D feature of current_stage, of only intermediate nodes or all nodes (cannot be only leaf nodes)
        intermediate_flag(bool): [True] only gives intermediate nodes, or [False] gives all nodes
        current_stage(int): current stage index
        target_stage(int): target stage index
    @Returns:
        pca_params: modified c/w Saab filters on spatial sizes based on testing data, spectral operation is preserverd
        feature_all(list, np.array, np.float64)[sample_images0, sample_images1, ...]: 
            4-D feature of each stage in the processtarget_stage (exclude current_stage, include current_stage)
        sample_images(np.array, np.float64)[num, c, h, w]: 4-D feature of target_stage
    '''
    pca_params = pca_params_init.copy()
    
    # custermize spatial info
    sample_images = sample_images_init.copy()
    pca_params['Layer_%d/h'% current_stage] = sample_images.shape[-2]
    pca_params['Layer_%d/w'% current_stage] = sample_images.shape[-1]

    intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % current_stage] # float list
    num_intermediate_init = len(np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0])
    if intermediate_flag:
        assert num_intermediate_init == sample_images.shape[1]
    else:
        assert len(intermediate_ind_layer) == sample_images.shape[1]
    
    # feature_all = []
    # for each layer
    for i in range(current_stage+1, target_stage+1, 1):
#        print('--------stage %d --------' % i)
#        print('Sample/cuboid_previous.shape:', sample_images.shape)
#        
        intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % (i-1)] # float list
        intermediate_index_layer = np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0]
        bias_layer = pca_params['Layer_%d/bias' % (i-1)]
        # prepare for next layer, take out intermediate nodes
        if i == current_stage+1 and (intermediate_flag): # current_stage, only gives intermediate nodes
            pass
        else: # else gives all nodes
            sample_images = sample_images[:, intermediate_index_layer, :, :]
#            print('intermediate_ind_layer:', intermediate_index_layer)
#            print('Sample/cuboid_forNextStage.shape:', sample_images.shape)
        
        if sample_images.shape[1] > 0:
            # Maxpooling
#            if i > 0:
#                sample_images = block_reduce(sample_images, (1, 1, 2, 2), np.max)
#                print('Sample/max_pooling.shape:', sample_images.shape)
                
            [n, num_node, h, w] = sample_images.shape
            feature_layer = []
    
            for channel_idx, bias in zip(np.arange(num_node),bias_layer[intermediate_index_layer]):
#                print('##### intermediate channel %d #####' % channel_idx)
                # Create patches in this slice
                sample_chl = sample_images[:, channel_idx, :, :]
                sample_chl = np.expand_dims(sample_chl, axis=1)
                
                feature_channel = conv_with_bias(i, channel_idx,
                                                 sample_chl, bias,
                                                 pca_params, 
#                                                 [kernel_size[i], stride[i], dilate[i], num_kernels[i], padF],
                                                 train_flag = False)
                # collect all slice feature in this layer
                feature_layer.append(feature_channel)
                
                # end with all valid parent nodes (ALL CHANNELS)
                
            # one layer summary
            sample_images = np.concatenate(feature_layer, axis=1)  # only control features for further growing
            pca_params['Layer_%d/h'% i] = sample_images.shape[-2]
            pca_params['Layer_%d/w'% i] = sample_images.shape[-1]
#            print('Sample/cuboid_toNextStage.shape:', sample_images.shape)
#            print()
            
            # if collectFlag:
            #     feature_all.append(sample_images)
            # end with ONE LAYER conditionally
        # end with ONE LAYER
        
    ### last layer  intermediate_ind is not used
    return sample_images #, sample_images # last stage feature [num, c, h, w]


def inference_chl_wise_simplified_encap(sample_images_init,
                                        transform_setting):

    return inference_chl_wise_simplified(transform_setting['pca_params_init'],
                            sample_images_init, transform_setting['intermediate_flag'],
                            transform_setting['current_stage'], transform_setting['target_stage'])




def reconstruction_chl_wise(pca_params, sample_images_list, current_stage, target_stage):
    '''
    sample_images_list: 
        first several stage: leaf cuboid 
        last element for current stage: intermediate nodes or all nodes (leaf nodes only when current_stage is the last layer)
    '''
    num_samples = sample_images_list[-1].shape[0]
    sample_images = sample_images_list[-1]
    
    # for each hop
    sample_list_idx = 0
    for i in range(current_stage,target_stage,-1):
#        print('--------stage %d --------' % i)
        sample_list_idx -= 1
        
        # data preparation w.r.t. dimension concatenation
        intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % i] # float list
        intermediate_index_layer = np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0]
        h_cur = pca_params['Layer_%d/h' % i]
        w_cur = pca_params['Layer_%d/w' % i]
        
        sample_images_cur = np.zeros((num_samples, len(intermediate_ind_layer), h_cur, w_cur))
        if len(intermediate_index_layer) == sample_images.shape[1]:   #########################################333
            sample_images_cur[:, intermediate_index_layer, :, :] = sample_images
        else:
            sample_images_cur[:, :, :, :] = sample_images
#        print('Sample/cuboid_cur.shape:', sample_images_cur.shape)
        
        if np.abs(sample_list_idx) <= len(sample_images_list) and np.abs(sample_list_idx) > 1:
#            print('multi-input usage')
            # use input sample
            leaf_index_layer = np.where(intermediate_ind_layer <= pca_params['energy_perc_thre'])[0]
            sample_images_cur[:, leaf_index_layer, :, :] = sample_images_list[sample_list_idx]
            
        # go to previous layer
        sample_rec = []
        start = 0
        num_node_list = pca_params['Layer_%d/num_node_list' % i]
        
        # check
        intermd_ind_layer = pca_params['Layer_%d/intermediate_ind' % (i-1)] # float list
        intermd_index_layer = np.where(intermd_ind_layer > pca_params['energy_perc_thre'])[0]
        bias_pre_layer = pca_params['Layer_%d/bias' % (i-1)][intermd_index_layer]
#        print(pca_params['Layer_%d/bias' % (i-1)], bias_pre_layer)
        
#        print(len(bias_pre_layer))
#        print(np.sum(num_node_list), len(num_node_list))
        assert len(bias_pre_layer) == len(num_node_list)
        for channel_idx, channel_dimension in enumerate(num_node_list):
            sample_chl = sample_images_cur[:, start:start+channel_dimension, :, :]
            start += channel_dimension
            rec_chl = decov_with_bias(i, channel_idx, sample_chl, bias_pre_layer[channel_idx], pca_params)
            sample_rec.append(rec_chl)
#            print('sample_chl', sample_chl.shape)
#            print('Sample/rec_chl.shape:', rec_chl.shape)
        
        sample_images = np.concatenate(sample_rec, axis = 1)
#        print('Sample/cuboid_pre.shape:', sample_images.shape)
        
    return sample_images

#%%%%%%%%%%%%%%
# def multi_saab_chl_wise_selected(sample_images_init, interest_sample_idx,
#                                  stride, 
#                                  kernel_size, 
#                                  dilate,
#                                  num_kernels,
#                                  energy_perc_thre, # int(num_nodes) or float(presE, var) for nodes growing down
#                                  padFlag = True,
#                                  recFlag = True,
#                                  collectFlag = False,
#                                  init_bias = 0):
#     '''
#     Do the Saab "training".
#     the length should be equal to kernel_sizes.
#     @Args:
#         sample_images_init(np.array, np.float64),[num_images, channel, height, width]
#         interest_sample_idx(np.array, np.int), [num_nterest_sample, ]
#         stride(list, int): stride for each stage
#         kernel_size(list, int): subspace size for each stage, the length defines how many stages conducted
#         dilate(list, int): dilate for each stage
#         num_kernels(list, float or int): used in pca for number of valid AC components preservation. int w.r.t. number, float w.r.t. cumulative energy
#         energy_perc_thre(float < 1): energy percantage threshold for a single node, if global ene perc < eneray_perc_threshold, then stop splitting
#                                      only work for AC nodes
#         recFlag(boolean): If true, the AC kernel number is limited by the AC feature matrix rank. This is necessary for reconstruction.
#     @Returns: 
#         pca_params(dict): residue saab transform parameters
#         feature_all: a list of feature (all nodes) at each stage
#         sample_images[num_images, channel, height, width]: feature at last stage
#     '''
#     num_layers = len(kernel_size)
#     pca_params = {}
    
# #    pca_params['num_layers'] = num_layers
#     pca_params['stride'] = stride
#     pca_params['kernel_size'] = kernel_size
#     pca_params['dilate'] = dilate
#     pca_params['num_kernels'] = num_kernels
#     pca_params['energy_perc_thre'] = energy_perc_thre
#     pca_params['padFlag'] = padFlag
#     pca_params['recFlag'] = recFlag
    
#     sample_images = sample_images_init.copy()
#     pca_params['Layer_-1/h'] = sample_images.shape[-2]
#     pca_params['Layer_-1/w'] = sample_images.shape[-1]
#     pca_params['Layer_-1/intermediate_ind'] = np.ones(sample_images.shape[1])
# #    pca_params['Layer_-1/Slice_0/bias'] = init_bias
#     pca_params['Layer_-1/bias'] = np.array([init_bias])
#     feature_all = []
    
#     # for each layer
#     i_valid = -1
#     for i in range(num_layers):
#         print('--------stage %d --------' % i)
#         print('Sample/cuboid_previous.shape:', sample_images.shape)
        
#         # prepare for next layer
#         intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % (i-1)]  # float list
#         bias_layer = pca_params['Layer_%d/bias' % (i-1)]
        
#         if np.sum(intermediate_ind_layer) > 0:
#             i_valid += 1
#             intermediate_index_layer = np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0]
#             sample_images = sample_images[:, intermediate_index_layer, :, :]
# #            print('intermediate_ind_layer:', intermediate_index_layer)
#             print('Sample/cuboid_forNextStage.shape:', sample_images.shape)
            
#             # Maxpooling
# #            if i > 0:
# #                sample_images = block_reduce(sample_images, (1, 1, 2, 2), np.max)
# #                print('Sample/max_pooling.shape:', sample_images.shape)
                
#             [n, num_node, h, w] = sample_images.shape
            
#             feature_layer = [] # (unsorteed) features in this new hop from previous sorted feat.
#             intermediate_ind_layer_new = []
#             num_node_list = []
#             bias_layer_new = []
            
# #            print(bias_layer[intermediate_index_layer])
            
#             # for each channel
#             for channel_idx, intermediate_ene, bias in zip(np.arange(num_node), intermediate_ind_layer[intermediate_index_layer], bias_layer[intermediate_index_layer]):  # only partial samples going down
# #                print('##### intermediate channel %d #####' % channel_idx)
#                 # Create patches in this channel
#                 sample_chl = sample_images[:, channel_idx, :, :]
#                 sample_chl = np.expand_dims(sample_chl, axis=1)
#                 # extract filter and store in pca_params
#                 feature_channel, bias_chl = conv_with_bias_selected(i, channel_idx,
#                                                  sample_chl, interest_sample_idx,
#                                                  bias,
#                                                  pca_params, 
#                                                  train_pack=intermediate_ene, train_flag=True)
#                 intermediate_ind_channel = pca_params['Layer_%d/Slice_%d/intermediate_ind' % (i, channel_idx)]
#                 # collect all slice feature in this layer
#                 feature_layer.append(feature_channel)
#                 intermediate_ind_layer_new.append(intermediate_ind_channel)
#                 num_node_list.append(len(intermediate_ind_channel)) # include DC, include small AC
#                 bias_layer_new.append(bias_chl)
#                 # end with all valid parent nodes (ALL CHANNELS)
            
#             # one layer summary
#             sample_images = np.concatenate(feature_layer, axis=1)  # only control features for further growing
#             intermediate_ind_layer_new = np.concatenate(intermediate_ind_layer_new, axis=0)
#             num_node_list = np.array(num_node_list)
#             bias_layer_new = np.concatenate(bias_layer_new)
#             pca_params['Layer_%d/intermediate_ind' % i] = intermediate_ind_layer_new # float list
#             pca_params['Layer_%d/num_node_list' % i] = num_node_list
#             pca_params['Layer_%d/bias' % i] = bias_layer_new
#             pca_params['Layer_%d/h'% i] = sample_images.shape[-2]
#             pca_params['Layer_%d/w'% i] = sample_images.shape[-1]
#             print('Sample/cuboid_toNextStage.shape:', sample_images.shape)
# #            print()
            
#             if collectFlag:
#                 feature_all.append(sample_images)
#             # end with ONE LAYER conditionally
#         # end with ONE LAYER
        
# #    #### adjust last layer intermediate_ind to be all none zero
# #    intermediate_ind_layer_last = pca_params['Layer_%d/intermediate_ind' % i_valid]
# #    intermediate_ind_layer_last[intermediate_ind_layer_last==0] = -1
# #    pca_params['Layer_%d/intermediate_ind' % i_valid] = intermediate_ind_layer_last
#     pca_params['num_layers'] = i_valid+1
#     return pca_params, feature_all, sample_images


# def reconstruction_chl_wise_selected(pca_params, sample_images_list, sample_idx, current_stage, target_stage):
#     '''
#     sample_images_list: 
#         first several stage: leaf cuboid 
#         last element for current stage: intermediate nodes or all nodes (leaf nodes only when current_stage is the last layer)
#     '''
#     num_samples = sample_images_list[-1].shape[0]
#     sample_images = sample_images_list[-1]
    
#     # for each hop
#     sample_list_idx = 0
#     for i in range(current_stage,target_stage,-1):
#         print('--------stage %d --------' % i)
#         sample_list_idx -= 1
        
#         # data preparation w.r.t. dimension concatenation
#         intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % i] # float list
#         intermediate_index_layer = np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0]
#         h_cur = pca_params['Layer_%d/h' % i]
#         w_cur = pca_params['Layer_%d/w' % i]
        
#         sample_images_cur = np.zeros((num_samples, len(intermediate_ind_layer), h_cur, w_cur))
#         if len(intermediate_index_layer) == sample_images.shape[1]:   #########################################333
#             sample_images_cur[:, intermediate_index_layer, :, :] = sample_images
#         else:
#             sample_images_cur[:, :, :, :] = sample_images
#         print('Sample/cuboid_cur.shape:', sample_images_cur.shape)
        
#         if np.abs(sample_list_idx) <= len(sample_images_list) and np.abs(sample_list_idx) > 1:
# #            print('multi-input usage')
#             # use input sample
#             leaf_index_layer = np.where(intermediate_ind_layer <= pca_params['energy_perc_thre'])[0]
#             sample_images_cur[:, leaf_index_layer, :, :] = sample_images_list[sample_list_idx]
            
#         # go to previous layer
#         sample_rec = []
#         start = 0
#         num_node_list = pca_params['Layer_%d/num_node_list' % i]
        
#         # check
#         intermd_ind_layer = pca_params['Layer_%d/intermediate_ind' % (i-1)] # float list
#         intermd_index_layer = np.where(intermd_ind_layer > pca_params['energy_perc_thre'])[0]
#         bias_pre_layer = pca_params['Layer_%d/bias' % (i-1)][intermd_index_layer]
# #        print(pca_params['Layer_%d/bias' % (i-1)], bias_pre_layer)
        
# #        print(len(bias_pre_layer))
# #        print(np.sum(num_node_list), len(num_node_list))
#         assert len(bias_pre_layer) == len(num_node_list)
#         for channel_idx, channel_dimension in enumerate(num_node_list):
#             sample_chl = sample_images_cur[:, start:start+channel_dimension, :, :]
#             start += channel_dimension
#             rec_chl = decov_with_bias_selected(i, channel_idx, sample_chl, sample_idx, bias_pre_layer[channel_idx], pca_params)
#             sample_rec.append(rec_chl)
# #            print('sample_chl', sample_chl.shape)
# #            print('Sample/rec_chl.shape:', rec_chl.shape)
        
#         sample_images = np.concatenate(sample_rec, axis = 1)
#         print('Sample/cuboid_pre.shape:', sample_images.shape)
        
#     return sample_images



# def reconstruction_chl_wise_selected_simplified(pca_params, pca_params_list, sample_images_list, sample_idx_list, current_stage, target_stage):
#     '''
#     sample_images_list: 
#         first several stage: leaf cuboid 
#         last element for current stage: intermediate nodes or all nodes (leaf nodes only when current_stage is the last layer)
#     '''
#     num_samples = sample_images_list[-1].shape[0]
#     sample_images = sample_images_list[-1]
    
#     # for each hop
#     sample_list_idx = 0
#     for i in range(current_stage,target_stage,-1):
#         print('--------stage %d --------' % i)
#         sample_list_idx -= 1
        
#         # data preparation w.r.t. dimension concatenation
#         intermediate_ind_layer = pca_params['Layer_%d/intermediate_ind' % i] # float list
#         intermediate_index_layer = np.where(intermediate_ind_layer > pca_params['energy_perc_thre'])[0]
#         h_cur = pca_params['Layer_%d/h' % i]
#         w_cur = pca_params['Layer_%d/w' % i]
        
#         sample_images_cur = np.zeros((num_samples, len(intermediate_ind_layer), h_cur, w_cur))
#         if len(intermediate_index_layer) == sample_images.shape[1]:   #########################################333
#             sample_images_cur[:, intermediate_index_layer, :, :] = sample_images
#         else:
#             sample_images_cur[:, :, :, :] = sample_images
#         print('Sample/cuboid_cur.shape:', sample_images_cur.shape)
        
#         if np.abs(sample_list_idx) <= len(sample_images_list) and np.abs(sample_list_idx) > 1:
# #            print('multi-input usage')
#             # use input sample
#             leaf_index_layer = np.where(intermediate_ind_layer <= pca_params['energy_perc_thre'])[0]
#             sample_images_cur[:, leaf_index_layer, :, :] = sample_images_list[sample_list_idx]
            
#         # go to previous layer
#         sample_rec = []
#         start = 0
#         num_node_list = pca_params['Layer_%d/num_node_list' % i]
        
#         # check
#         intermd_ind_layer = pca_params['Layer_%d/intermediate_ind' % (i-1)] # float list
#         intermd_index_layer = np.where(intermd_ind_layer > pca_params['energy_perc_thre'])[0]
#         bias_pre_layer = pca_params['Layer_%d/bias' % (i-1)][intermd_index_layer]
# #        print(pca_params['Layer_%d/bias' % (i-1)], bias_pre_layer)
        
# #        print(len(bias_pre_layer))
# #        print(np.sum(num_node_list), len(num_node_list))
#         assert len(bias_pre_layer) == len(num_node_list)
#         for channel_idx, channel_dimension in enumerate(num_node_list):
#             sample_chl = sample_images_cur[:, start:start+channel_dimension, :, :]
#             start += channel_dimension
#             rec_chl = decov_with_bias_selected_simplified(i, channel_idx, sample_chl, sample_idx_list, bias_pre_layer[channel_idx], pca_params_list)
#             sample_rec.append(rec_chl)
# #            print('sample_chl', sample_chl.shape)
# #            print('Sample/rec_chl.shape:', rec_chl.shape)
        
#         sample_images = np.concatenate(sample_rec, axis = 1)
#         print('Sample/cuboid_pre.shape:', sample_images.shape)
        
#     return sample_images





#%%
if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error as MSE
    # init_images = np.random.rand(5, 1, 13, 13) * 1000 # (num_img, num_channel, img_h, img_w)

    from sklearn.datasets import load_digits
    
    digits = load_digits()
    init_images = np.expand_dims(digits.images[1:3], 1)
    print('init_images', init_images.shape)
    print()

    print('filter extraction')
    params, feature_all, feature_last = multi_saab_chl_wise(init_images, 
                                                            [1], # stride,
                                                            [5], # kernel_size,
                                                            [1], #dilate,
                                                            ['full'], #num_kernels,
                                                            0.125, #energy_perc_thre, # int(num_nodes) or float(presE, var) for nodes growing down
                                                            padFlag = [True],
                                                            recFlag = True,
                                                            collectFlag = True)
    print('feature', feature_last.shape)
    print()
    
    print(' check inference_chl_wise')
    __, __, feature1 = inference_chl_wise(params,
                                          init_images, True,
                                          -1, 0,
                                          collectFlag=False)
    print(set((feature_last.reshape(-1) == feature1.reshape(-1)).tolist()))
    print()

    print(' check inference_chl_wise_simplified')
    feature_simp = inference_chl_wise_simplified(params,
                                          init_images, True,
                                          -1, 0)
    print(set((feature_last.reshape(-1) == feature_simp.reshape(-1)).tolist()))
    print()
    
    print(' check reconstruction from Hop1')
    feature0_rec = reconstruction_chl_wise(params, [feature1], 0, -1)
    print(MSE(init_images.reshape(-1),
              feature0_rec.reshape(-1)))
    print()
    


    print('check parallel module')
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=4, verbose=1)(delayed(inference_chl_wise_simplified)(params, init_images[i:i+1], True, -1, 0) for i in range(len(init_images)))
    results = np.concatenate(results, axis=0)
    print(set((results.reshape(-1) == feature1.reshape(-1)).tolist()))
    print(MSE(results.reshape(-1),
              feature1.reshape(-1)))
    print()

    print('manually')
    m_results = []
    for i in range(len(init_images)):
        j = inference_chl_wise_simplified(params, init_images[i:i+1], True, -1, 0)
        m_results.append(j)
    m_results = np.array(m_results)
    print(set((m_results.reshape(-1) == feature1.reshape(-1)).tolist()))
    print(MSE(m_results.reshape(-1),
              feature1.reshape(-1)))
    print()



#%%
    img_idx = 0
    num_chl = len(feature1[img_idx])
    num_c = int(np.sqrt(num_chl))
    num_r = int(np.sqrt(num_chl))

    from matplotlib import pyplot as plt

    max_val = np.max(init_images[img_idx, 0])
    min_val = np.min(init_images[img_idx, 0])

    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(init_images[img_idx, 0], vmin=min_val, vmax=max_val)
    plt.show()

    plt.figure(figsize=(10, 10))
    for idx, i in enumerate(feature1[img_idx]):
        plt.subplot(num_c, num_r, idx+1)
        plt.imshow(i, vmin=min_val, vmax=max_val)
    plt.show()

#%%
    print('partial recontructoin from Hop1')
    feature1[:, 10:, :, :] = 0
    feature0_rec_part = reconstruction_chl_wise(params, [feature1], 0, -1)
    print(MSE(init_images.reshape(-1),
              feature0_rec_part.reshape(-1)))
    print()


#%%
#if __name__ == "__main__":
#    from sklearn.metrics import mean_squared_error as MSE
#    
#    init_images = np.random.rand(10, 1, 300, 400)
#    
#    print('filter extraction')
#    params, feature_all, feature_last = multi_saab_chl_wise(init_images, 
#                                                            [1,1,1], # stride, 
#                                                            [3,3,3], # kernel_size, 
#                                                            [1,1,1], #dilate,
#                                                            ['full', 'full', 'full'], #num_kernels,
#                                                            0.125, #energy_perc_thre, # int(num_nodes) or float(presE, var) for nodes growing down
#                                                            padFlag = [True, True, True],
#                                                            recFlag = True,
#                                                            collectFlag = True)
##    print([i.shape for i in feature_all])
##    print()
#
#
#    print('get node energy')
#    node_energy_all = [params['Layer_%d/intermediate_ind' % i] for i in range(params['num_layers'])]
#    print([i.shape for i in node_energy_all])
#    print()
#    
#    print(' check inference_chl_wise')
#    __, __, feature1 = inference_chl_wise(params,
#                                       init_images, True,
#                                       -1, 1,
#                                       collectFlag=False)
#    print(set((feature_all[1].reshape(-1) == feature1.reshape(-1)).tolist()))
#
#
#    __, __, feature2 = inference_chl_wise(params,
#                                       feature1, False,
#                                       1, 2,
#                                       collectFlag=False)
#    print(set((feature_last.reshape(-1) == feature2.reshape(-1)).tolist()))
#    print()
#
##    # check inference_chl_wise, input control
##    #__, feature_last2 = inference_chl_wise(params,
##    #                                        feature1, True,
##    #                                        1, 2,
##    #                                        collectFlag=False)
##    # supposed to get AssertionError
#
#    print(' MSE benchmark')
#    print(MSE(init_images.reshape(-1), np.zeros(init_images.reshape(-1).shape)))
#    print()
#    
#    print(' check 1 layer, current layer to be final layer # [-1_layer]')
#    feature1_rec = reconstruction_chl_wise(params, [feature_last], 2, 1)
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 1] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer > params['energy_perc_thre'])[0]
#    #print(set((feature_all[1][:,intermediate_index_layer,:,:].reshape(-1) == feature1_rec.reshape(-1)).tolist()))
#    print(MSE(feature_all[1][:,intermediate_index_layer,:,:].reshape(-1),
#              feature1_rec.reshape(-1)))
#    print()
#
#
#    print(' check 1 layer, current layer to be middle layer # [-2_layer], all nodes')
#    feature0_rec = reconstruction_chl_wise(params, [feature_all[1]], 1, 0)
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 0] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer > params['energy_perc_thre'])[0]
#    #print(set((feature_all[1][:,intermediate_index_layer,:,:].reshape(-1) == feature1_rec.reshape(-1)).tolist()))
#    print(MSE(feature_all[0][:,intermediate_index_layer,:,:].reshape(-1),
#              feature0_rec.reshape(-1)))
#    print()
#
#    print(' check 1 layer, current layer to be middle layer # [-2_layer], intermediate nodes')
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 1] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer > params['energy_perc_thre'])[0]
#    feature1_interm = feature_all[1][:,intermediate_index_layer,:,:]
#    feature0_rec = reconstruction_chl_wise(params, [feature1_interm], 1, 0)
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 0] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer > params['energy_perc_thre'])[0]
#    #print(set((feature_all[1][:,intermediate_index_layer,:,:].reshape(-1) == feature1_rec.reshape(-1)).tolist()))
#    print(MSE(feature_all[0][:,intermediate_index_layer,:,:].reshape(-1),
#              feature0_rec.reshape(-1)))
#    print()
#
#    print(' check 2 layer, current layer to be final layer # [-2_layer_leaf, -1_layer]')
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 1] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer < params['energy_perc_thre'])[0]
#    feature1_leaf = feature_all[1][:,intermediate_index_layer,:,:]
#    feature0_rec = reconstruction_chl_wise(params, [feature1_leaf, feature_last], 2, 0)
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 0] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer > params['energy_perc_thre'])[0]
#    #print(set((feature_all[1][:,intermediate_index_layer,:,:].reshape(-1) == feature1_rec.reshape(-1)).tolist()))
#    print(MSE(feature_all[0][:,intermediate_index_layer,:,:].reshape(-1),
#              feature0_rec.reshape(-1)))
#    print()
#    
#    print(' check 2 layers, current layer intermediate nodes # [-3_layer_leaf, -2_layer_intermediate]')
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 0] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer < params['energy_perc_thre'])[0]
#    feature0_leaf = feature_all[0][:,intermediate_index_layer,:,:]
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 1] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer > params['energy_perc_thre'])[0]
#    feature1_interm = feature_all[1][:,intermediate_index_layer,:,:]
#    img_rec = reconstruction_chl_wise(params, [feature0_leaf, feature1_interm], 1, -1)
#    #intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 0] # float list
#    #intermediate_index_layer = np.where(intermediate_ind_layer > params['energy_perc_thre'])[0]
#    #print(set((feature_all[1][:,intermediate_index_layer,:,:].reshape(-1) == feature1_rec.reshape(-1)).tolist()))
#    print(MSE(init_images.reshape(-1),
#              img_rec.reshape(-1)))
#    print()
#    
#    print(' check 2 layers, current layer all nodes # [-3_layer_leaf, -2_layer]')
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 0] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer < params['energy_perc_thre'])[0]
#    feature0_leaf = feature_all[0][:,intermediate_index_layer,:,:]
#    feature1 = feature_all[1]
#    img_rec = reconstruction_chl_wise(params, [feature0_leaf, feature1], 1, -1)
#    #intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 0] # float list
#    #intermediate_index_layer = np.where(intermediate_ind_layer > params['energy_perc_thre'])[0]
#    #print(set((feature_all[1][:,intermediate_index_layer,:,:].reshape(-1) == feature1_rec.reshape(-1)).tolist()))
#    print(MSE(init_images.reshape(-1),
#              img_rec.reshape(-1)))
#    print()
#    
#    print(' check for 3 layers')
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 0] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer < params['energy_perc_thre'])[0]
#    feature0_leaf = feature_all[0][:,intermediate_index_layer,:,:]
#    intermediate_ind_layer = params['Layer_%d/intermediate_ind' % 1] # float list
#    intermediate_index_layer = np.where(intermediate_ind_layer < params['energy_perc_thre'])[0]
#    feature1_leaf = feature_all[1][:,intermediate_index_layer,:,:]
#    img_rec = reconstruction_chl_wise(params, [feature0_leaf, feature1_leaf, feature_all[2]], 2, -1)
#    print(MSE(init_images.reshape(-1),
#              img_rec.reshape(-1)))
#    print()
#
#    print(' check for testing images with different spatial size from training images')
#    test_images = np.random.rand(10, 1, 30, 40)
#    test_params, __, test_feature1 = inference_chl_wise(params,
#                                                         test_images, True,
#                                                         -1, 1,
#                                                         collectFlag=False)
#
#    
##    import lib_patches_images as lib_patches
##    feature_all_samesize = lib_patches.upsampling_set(feature_all, [320, 480])
##    print([i.shape for i in feature_all_samesize])
##    #feature_last2_up = upsampling(feature_last2, [320,480])
##    #print(feature_last2_up.shape)
##    #feature_last2_up = [np.kron(i, np.ones((1,2,2), dtype=i.dtype)) for i in feature_last2]
##    #print(feature_last2_up.shape)

