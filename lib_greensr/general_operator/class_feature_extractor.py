import numpy as np
from sklearn.decomposition import IncrementalPCA as PCA
import torch
from torch.nn import functional as f

import lib_greensr.util.lib_general_tools as lib_gtools
import lib_greensr.util.lib_saab_channel_wise_v5_realbias_updated_no_bias_generalwindow as lib_saab_cw
from lib_greensr.util.class_handcraft_filters import HANDCRAFT_FILTERS





class FEATURE_EXTRACTOR():

    def __init__(self):
        # self.hc_filters = HANDCRAFT_FILTERS()
        self.hc_filters = None


    #%%
    # normal saab
    # can work on central arbitrary size region

    def normal_saab_fit(self,
                        patch_4d,
                        saab_setting,
                        neighborhood_size=0,
                        return_feat_flag=False):#,
                        # saab_filename_tosave,
                        # retrain_flag=False):
        '''
        - Extract filters from patch_size

        saab_setting = {'patch_size':[[3, 3], [5, 5]], 
                        'patch_stride':[3, 1]}#,
                        # optional
                        #'neighborhood_size':15}
                        #'gaus_sigma':0, 
                        #'align_flag'=True # no align for center patch filter
        '''
        # model_exist_flag = os.path.isfile(saab_filename_tosave)
        # if retrain_flag or (not model_exist_flag):

        patch_stride = saab_setting['patch_stride']
        if neighborhood_size <= 0 or neighborhood_size > patch_4d.shape[-1]:
            n_size = patch_4d.shape[-1]
        else:
            n_size = neighborhood_size
        assert n_size <= patch_4d.shape[-1]
        b_width = (patch_4d.shape[-1] - n_size) // 2

        saab_params, __, feat = \
            lib_saab_cw.multi_saab_chl_wise(patch_4d[:, :,
                                                     b_width:(b_width+n_size),
                                                     b_width:(b_width+n_size)],
                                            patch_stride, # stride,
                                            saab_setting['patch_size'],
                                            [1 for i in patch_stride], #dilate,
                                            ['full' for i in patch_stride], #num_kernels,
                                            0, #energy_perc_thre,
                                            # int(num_nodes) or float(presE, var) \
                                            # for nodes growing down
                                            padFlag = [False for i in saab_setting['patch_stride']],
                                            recFlag = True,
                                            collectFlag = False)
        if return_feat_flag:
            return saab_params, feat
        else:
            return saab_params


    def normal_saab_transform(self,
                              patch_4d,
                              saab_params,
                              neighborhood_size=0):
        '''
        - Transform the samples with pre-saved saab filters
        '''
        # saab_params = lib_gtools.obj_loading(saab_filename_tosave)
        if neighborhood_size == 0:
            n_size = patch_4d.shape[-1]
        else:
            n_size = neighborhood_size
        assert n_size <= patch_4d.shape[-1]
        b_width = (patch_4d.shape[-1] - n_size) // 2

        feat_4d = \
            lib_saab_cw.inference_chl_wise_simplified(saab_params,
                                                      patch_4d[:, :,
                                                               b_width:(b_width+n_size),
                                                               b_width:(b_width+n_size)],
                                                      True,
                                                      -1,
                                                      len(saab_params['kernel_size'])-1)
        return feat_4d


    #%%

    # tbc by local pca
    def chlwise_pca_fit(self, patch_3dor4d):
        '''
        patch_3dor4d: (n, c, h*w) or (n, c, h, w)
        '''
        num_samples = len(patch_3dor4d)
        num_chl = patch_3dor4d.shape[1]
        pca_list = []
        for chl_idx in range(num_chl):
            pca = PCA()
            pca.fit(patch_3dor4d[:, chl_idx].reshape(num_samples, -1))
            pca_list.append(pca)
        return pca_list

    # tbc by local pca
    def chlwise_pca_transform(self, patch_3dor4d, pca_list):
        num_samples = len(patch_3dor4d)
        num_chl = patch_3dor4d.shape[1]
        assert num_chl == len(pca_list)
        patch_tran = []
        for chl_idx in range(num_chl):
            pca = pca_list[chl_idx]
            patch_singchl_tran = pca.transform(patch_3dor4d[:, chl_idx].reshape(num_samples, -1))
            patch_tran.append(patch_singchl_tran)
        return patch_tran # len == c, each one (n, h*w)




    #%%
    def handcraft_feat_transform(self,
                            patch_4d,
                            rep_setting,
                            neighborhood_size=0):

        self.hc_filters = HANDCRAFT_FILTERS()

        if neighborhood_size == 0:
            neighborhood_size = patch_4d.shape[-1]
        b_width = (patch_4d.shape[-1] - neighborhood_size) // 2

        filters = self.hc_filters.get_filter_bank(rep_setting['rep_type'])
        feat_img_arr = \
            f.conv2d(torch.from_numpy(patch_4d[:, :,
                                               b_width:(b_width+neighborhood_size),
                                               b_width:(b_width+neighborhood_size)]).to(torch.float),
                     filters, bias=None,
                     stride=rep_setting['corepatch_stride'],
                     padding=0, dilation=1, groups=1).numpy()
        return feat_img_arr



    #%%
    def hog_transform(self, patch_3dor4d, cell_block_size, base_patch_size=0):
        '''
        ref: 
            https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
            https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
        '''
        from skimage.feature import hog
        from PIL import Image
    
        if base_patch_size > 0 and base_patch_size != patch_3dor4d.shape[-1]:
            # need pre-processing
            patch_resized = \
                [np.array(Image.fromarray(np.squeeze(i).astype(np.uint8)).resize(size=(base_patch_size,
                                                                                       base_patch_size),
                                                    resample=Image.LANCZOS))
                 for i in patch_3dor4d]

            #============================================================
            # print('patch_resized in hog_transform', np.array(patch_resized).shape)
            # lib_gtools.obj_saving('/scratch2/wang890/SR/2022_mar_featureselection/result/rft_reg_res_partition_proghier/model/0_regressors/temp_data/band1_ilr_patches_aug8_15_to_16_lanczos.npy',
            #                       np.array(patch_resized))
            #============================================================


        else:
            patch_resized = patch_3dor4d
        print('patch_resized', np.array(patch_resized).shape)
        cell_size, block_size = cell_block_size
        hog_feat = \
            np.array([hog(np.squeeze(i),
                orientations=8, pixels_per_cell=(cell_size, cell_size),
                cells_per_block=(block_size, block_size),
                feature_vector=True)
            for i in patch_resized])
        return hog_feat

# shape = (2, 1, 7, 7)
# a = np.arange(np.prod(shape)).reshape(shape)
# a_hog = hog_transform(a, (4, 2), base_patch_size=8)


    def get_patch_grad(self, patch_3dor4d):
        sobel_x = self.handcraft_feat_transform(patch_3dor4d,
                                                    {'rep_type':'sobel_x', 'corepatch_stride':1})
        sobel_y = self.handcraft_feat_transform(patch_3dor4d,
                                                    {'rep_type':'sobel_y', 'corepatch_stride':1})
        return sobel_x, sobel_y # same shape as patch_3dor4d

    def get_patch_grad_hist(self, patch_3dor4d, num_hist_bins):
        from lib_greensr.general_operator import gradient_histogram
        grad_x, grad_y = self.get_patch_grad(patch_3dor4d)
        grad_hist =\
            np.array([gradient_histogram.grad_histograms(np.squeeze(grad_x[s_i]), np.squeeze(grad_y[s_i]),
                                               grad_x.shape[-1], grad_x.shape[-2],
                                               grad_x.shape[-1], grad_x.shape[-2],
                                               num_hist_bins).ravel()
                      for s_i in range(len(grad_x))])
        return grad_hist #[num_samples, num_hist_bins]





#%%

if __name__ == "__main__":

    patch_arr = np.random.rand(200, 1, 15, 15)
    saab_setting = {'patch_size':[[3, 3]],
                    'patch_stride':[1],
                    'neighborhood_size':3}

    feat_extractor = FEATURE_EXTRACTOR()

    '''
    saab filter extractor
    can work on arbitrary central neighborhood size out from the given patches
    '''
    if False:
        saab_params = \
            feat_extractor.normal_saab_fit(patch_arr, saab_setting,
                                           neighborhood_size=saab_setting['neighborhood_size'])
    
        feat = \
        feat_extractor.normal_saab_transform(patch_arr, saab_params,
                                             neighborhood_size=saab_setting['neighborhood_size'])
        print('feat', feat.shape)
#%%
    '''
    hand-craft filters
    '''
    if True:
        # inputs = torch.randn(2, 4, 5, 5)
        # filters_demo = torch.randn(8, 4, 3, 3)
        # feat = f.conv2d(inputs, filters_demo, padding=1)
        # print('feat', feat.shape)
    
        rep_setting = {'rep_type':'haar',
                        'corepatch_stride':1}
        feat = \
            feat_extractor.local_des_transform(patch_arr,
                                                      rep_setting,
                                                      neighborhood_size=0)
        print('feat', feat.shape)

        # rep_setting_list = [{'rep_type':'haar',
        #                 'corepatch_stride':1},
        #                {'rep_type':'laws',
        #                 'corepatch_stride':1}]
        # # feat_list = \
        # #     feat_extractor.local_des_multiple_transform(patch_arr,
        # #                                               rep_setting_list,
        # #                                               neighborhood_size=0)
        # # print('feat', [i.shape for i in feat_list])
        # feat_list = \
        #     [feat_extractor.local_des_transform(patch_arr, rep_setting,
        #                                         neighborhood_size=0)
        #      for rep_setting in rep_setting_list]
        # print('feat', [i.shape for i in feat_list])

    #%%
    '''
    chl-wise pca
    '''
    if True:
        pca_list = feat_extractor.chlwise_pca_fit(feat)
        feat_chlpca_list = feat_extractor.chlwise_pca_transform(feat, pca_list)

        # pca_set_list = [feat_extractor.chlwise_pca_fit(i) for i in feat_list]
        # feat_chlpca_list = [feat_extractor.chlwise_pca_transform(i, j)
        #                     for i,j in zip(feat_list, pca_set_list)]
        # print('feat_chlpca_list', [j.shape for i in feat_chlpca_list for j in i])

