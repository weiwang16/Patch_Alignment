import numpy as np
import os
import gc

import lib_greensr.util.lib_general_tools as lib_gtools
import lib_greensr.util.lib_data_aug as lib_data_aug
from lib_greensr.general_operator.class_feature_extractor import FEATURE_EXTRACTOR

'''
implementation of "Patch Alignment" in
Wang, Wei, et al. "LSR++: An Efficient and Tiny Model for Image Super-Resolution." 
2023 Asia Pacific Signal and Information Processing Association Annual Summit 
and Conference (APSIPA ASC). IEEE, 2023.
'''


N_GRAD_HIST_BINS = 8



class DATA_MODIFIER():

    def __init__(self):
        pass

    def determine_aug_times(self, ratio_value):
        if ratio_value <= 0.125:
            return 8
        else:
            if ratio_value <= 0.25:
                return 4
            else:
                if ratio_value <= 0.5:
                    return 2
                else:
                    return 1


    def modify_samples_onedatatype(self, obj_arr_list, mod_num):
        '''
        mod_num: e.g. 8, 1, 0.5
        refers to mod_type: e.g. 'aug', 'raw', 'downsampling'
        '''
        print('bfr modify', [i.shape for i in obj_arr_list])
        if mod_num >= 1:
            # mod_type is 'aug' or 'raw'
            obj_arr_aug_list = []
            for obj_arr in obj_arr_list:
                obj_arr_aug = lib_data_aug.augment_images_auto(obj_arr,
                                             aug_times=int(mod_num))
                obj_arr_aug = np.concatenate(obj_arr_aug, axis=0)
                obj_arr_aug_list.append(obj_arr_aug)
            print('aft modify', [i.shape for i in obj_arr_aug_list])
            return obj_arr_aug_list
        else:
            # mode_type is 'downsampling'
            cond_data_idx = \
                np.random.choice(np.arange(len(obj_arr_list[0])),
                                  int(float(mod_num) * len(obj_arr_list[0])),
                                  replace=True)
            obj_arr_aug_list = [i[cond_data_idx] for i in obj_arr_list]
            print('aft modify', [i.shape for i in obj_arr_aug_list])
            return obj_arr_aug_list



    def modify_samples_multidatatype(self,
                               counterparts_arr_list,
                               criterion_arr,
                               data_type_settings_list):
        counterparts_modified_list = [[] for i in counterparts_arr_list]
        for data_type_settings in data_type_settings_list:
            print('data modify', data_type_settings)
            rang_right_edges, range_idx, mod_num = data_type_settings

            sample_idx = \
                lib_gtools.get_bin_qualified_samples(rang_right_edges,
                                                     criterion_arr,
                                                     range_idx)
            print('perc', len(sample_idx)/len(criterion_arr))

            counterparts_selected_modified_list = \
                self.modify_samples_onedatatype([i[sample_idx] \
                                                 for i in counterparts_arr_list],
                                                mod_num)

            for i,j in zip(counterparts_modified_list, counterparts_selected_modified_list):
                i.append(j)

        for idx in range(len(counterparts_modified_list)):
            counterparts_modified_list[idx] = \
                np.concatenate(counterparts_modified_list[idx],
                               axis=0)

        return counterparts_modified_list



    def align_patches(self, patch4d_arr, mode='alignedByHalfMean',
                      get_source_arr_updated_flag=False):
        '''
        This operation aligns the raw patches (patch4d_arr) based on the 
        alignment criteria (mode on sources_arr).
        This operation modifies the input patch4d_arr, and source_arr.
        
        PRE-REQUISITE: 
            1. The patches are single channel images
            2. The patches are square.

        @ Args:
            patch4d_arr, 4-D (num_patches, 1, h, w): 
                Single channel raw patch array, the patches for alignment.
            
            mode, str:
                'raw': no modification
                'alignedByHalfMean': align by half patch var
                'alignedByHalfVar': align by half patch mean
                'alignedByGradHist': align by patch gradient histogram
        
        @ In codes,
            source_arr: 
                The resource on which the alignment criteria works. 
                4-D (num_patches, 1, h, w) 
                    raw patch array if mode == 'alignedByHalfMean' or 'alignedByHalfVar'.
                2-D (num_patches, n_grad_hist_bins)
                    patch gradient histogram if mode == 'alignedByGradHist'.
        
        '''
        num_patches = len(patch4d_arr)
        patch_size = patch4d_arr.shape[-1]
        patch_size_half = patch_size // 2

        if mode == 'raw':
            return patch4d_arr
        else:
            # align by half patch var or mean
            if mode == 'alignedByHalfMean' or mode == 'alignedByHalfVar':
                source_arr = patch4d_arr.copy()

                # left-right
                if mode == 'alignedByHalfMean':
                    left_crit = \
                        np.mean(source_arr[:, :, :, :patch_size_half].reshape((num_patches,
                                                                                -1)),
                                axis=-1)
                    right_crit = \
                        np.mean(source_arr[:, :, :, -1*patch_size_half:].reshape((num_patches,
                                                                                    -1)),
                                axis=-1)
                elif mode == 'alignedByHalfVar':
                    left_crit = \
                        np.var(source_arr[:, :, :, :patch_size_half].reshape((num_patches,
                                                                                -1)),
                                axis=-1)
                    right_crit = \
                        np.var(source_arr[:, :, :, -1*patch_size_half:].reshape((num_patches,
                                                                                    -1)),
                                axis=-1)
                diff = left_crit - right_crit
                patch4d_arr[diff < 0] = \
                    lib_data_aug.transform_images_onetype(patch4d_arr[diff < 0],
                                                          ['fliplr'])
                # up-down
                if mode == 'alignedByHalfMean':
                    up_crit = \
                        np.mean(source_arr[:, :, :patch_size_half, :].reshape((num_patches,
                                                                                -1)),
                                axis=-1)
                    bot_crit = \
                        np.mean(source_arr[:, :, -1*patch_size_half:, :].reshape((num_patches,
                                                                                    -1)),
                                axis=-1)
                elif mode == 'alignedByHalfVar':
                    up_crit = \
                        np.var(source_arr[:, :, :patch_size_half, :].reshape((num_patches,
                                                                                -1)),
                                axis=-1)
                    bot_crit = \
                        np.var(source_arr[:, :, -1*patch_size_half:, :].reshape((num_patches,
                                                                                    -1)),
                                axis=-1)
                diff = up_crit - bot_crit
                patch4d_arr[diff < 0] = \
                    lib_data_aug.transform_images_onetype(patch4d_arr[diff < 0],
                                                          ['flipud'])
                return patch4d_arr

            elif mode == 'alignedByGradHist':
                feat_ext = FEATURE_EXTRACTOR()
                source_arr = feat_ext.get_patch_grad_hist(patch4d_arr, N_GRAD_HIST_BINS)
                return self.align_patches_byGradHist(source_arr, patch4d_arr,
                                                     get_grad_hist_arr_flag=get_source_arr_updated_flag)


    def align_patches_byGradHist(self, grad_hist_arr, patch4d_arr, get_grad_hist_arr_flag=False):
        '''
        grad_hist_arr, 2-D (n_samples, n_grad_hist_bins) numpy array
        This operation modify the input patch4d_arr, and grad_hist_arr
        '''
        grad_hist_arr_updated = grad_hist_arr.copy()
        # print('grad_hist_arr ori')
        # print(grad_hist_arr)

        # Step 1: pre-processing up-down flipping for partial samples
        # find the bin with largest acumulative gradient
        d_max = np.argmax(grad_hist_arr, axis=-1)
        n_grad_hist_bins = grad_hist_arr.shape[-1]
        n_samples = grad_hist_arr.shape[0]

        # determine the samples need up-down flipping pre-processing
        nbr_gap = 2
        d_max_nbr_l = (d_max - nbr_gap) % n_grad_hist_bins
        d_max_nbr_r = (d_max + nbr_gap) % n_grad_hist_bins
        grad_hist_diff = grad_hist_arr[np.arange(n_samples, dtype=int), d_max_nbr_l] \
            - grad_hist_arr[np.arange(n_samples, dtype=int), d_max_nbr_r]

        # up-down flipping pre-processing on partial samples
        patch4d_arr[grad_hist_diff > 0] = \
            lib_data_aug.transform_images_onetype(patch4d_arr[grad_hist_diff > 0],
                                                  ['flipud'])
        # print('top-bottom flipping ind', grad_hist_diff > 0)

        # import matplotlib.pyplot as plt
        # update the grad_hist_arr after the up-down flipping pre-processing
        ori_bin_idx = np.arange(n_grad_hist_bins, dtype=int)
        new_bin_idx = (n_grad_hist_bins - ori_bin_idx) % n_grad_hist_bins
        grad_hist_arr_tomodify = grad_hist_arr[grad_hist_diff > 0]
        grad_hist_arr_updated[grad_hist_diff > 0] = grad_hist_arr_tomodify[:, new_bin_idx]
        # print('done with g_hist modification after top-bottom flipping')

        # Step 2: rotation
        d_max = np.argmax(grad_hist_arr_updated, axis=-1)
        mul_cwninety = d_max // 2
        ali_opr_lookup_table = ['None', 'rot_cw_90', 'rot_cw_180', 'rot_cw_270']
        for multiplier in np.arange(n_grad_hist_bins//2, dtype=int):
            if multiplier > 0:
                cond_sample_idx = np.where(mul_cwninety==multiplier)[0]
                patch4d_arr[cond_sample_idx] = \
                    lib_data_aug.transform_images_onetype(patch4d_arr[cond_sample_idx],
                                                          [ali_opr_lookup_table[multiplier]])

        if get_grad_hist_arr_flag:
            # update grad_hist_updated
            for multiplier in np.arange(n_grad_hist_bins//2, dtype=int):
                if multiplier > 0:
                    ori_bin_idx = np.arange(n_grad_hist_bins, dtype=int)
                    new_bin_idx = (ori_bin_idx - (n_grad_hist_bins - multiplier*2)) % n_grad_hist_bins
                    cond_sample_idx = np.where(mul_cwninety==multiplier)[0]
                    gh_temp = grad_hist_arr_updated[cond_sample_idx]
                    grad_hist_arr_updated[cond_sample_idx] = gh_temp[:, new_bin_idx]

            return patch4d_arr, grad_hist_arr, grad_hist_arr_updated
        else:
            return patch4d_arr



if __name__ == "__main__":

    # demo patch preparation
    from sklearn.datasets import load_sample_images
    import matplotlib.pyplot as plt
    p_s = 15
    sample_info = [(0, 200, 100),
                   (0, 320, 270),
                   (1, 200, 300),
                   (1, 200, 350)]
    dataset = load_sample_images()
    images = np.array(dataset.images)[:, :, :, 0:1]
    images = np.moveaxis(images, -1, 1)
    input_patches_ori = []
    for info in sample_info:
        idx, h, w = info
        input_patches_ori.append(images[idx, 0:1,
                                        h:(h+p_s),
                                        w:(w+p_s)])
    input_patches_ori = np.array(input_patches_ori)

    # input patch checking
    print('=== ORI PATCHES ===')
    print('input_patches_ori', input_patches_ori.shape)
    for i in input_patches_ori:
        plt.figure()
        plt.imshow(np.squeeze(i))
        plt.show()

    # patch alignment
    data_modifier = DATA_MODIFIER()
    input_patches_updated, grad_hist_arr_ori, grad_hist_arr_updated = \
        data_modifier.align_patches(input_patches_ori.copy(),
                        mode='alignedByGradHist',
                        get_source_arr_updated_flag=True)

    # result checking
    print('=== ALIGNED PATCHES ===')
    print('input_patches_updated', input_patches_updated.shape)
    print('grad_hist_arr_ori', grad_hist_arr_ori.shape)
    print('grad_hist_arr_updated', grad_hist_arr_updated.shape)
    for i in input_patches_updated:
        plt.figure()
        plt.imshow(np.squeeze(i))
        plt.show()