#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


#%%
def augment_images(sample_images, aug_times=8, aug_type_list=[[None], ['rot_anti_cw_90'], ['rot_anti_cw_180'], ['rot_anti_cw_270'],
                                                              ['flipud'], ['flipud', 'rot_anti_cw_90'], ['flipud', 'rot_anti_cw_180'], ['flipud', 'rot_anti_cw_270']]):
    assert aug_times == len(aug_type_list)
    sample_augmented = [transform_images_onetype(sample_images, aug_type=aug_type_sing) for aug_type_sing in aug_type_list]
#    sample_augmented = np.concatenate(sample_augmented, axis=0)
#    assert len(sample_augmented) == aug_times * len(sample_images)
    return sample_augmented

def augment_images_4d(sample_images_4d, aug_times=8, aug_type_list=[[None], ['rot_anti_cw_90'], ['rot_anti_cw_180'], ['rot_anti_cw_270'],
                                                              ['flipud'], ['flipud', 'rot_anti_cw_90'], ['flipud', 'rot_anti_cw_180'], ['flipud', 'rot_anti_cw_270']]):
    assert aug_times == len(aug_type_list)
    sample_augmented = [transform_images_onetype(np.squeeze(sample_images_4d), aug_type=aug_type_sing) for aug_type_sing in aug_type_list]
#    sample_augmented = np.concatenate(sample_augmented, axis=0)
#    assert len(sample_augmented) == aug_times * len(sample_images)
    sample_augmented = [np.expand_dims(i, 1) for i in sample_augmented]
    return sample_augmented



def augment_images_4d_auto(sample_images_4dor3d, aug_times=8):
    '''
    return sample_augmented (list), each element is 4-D np.ndarray
    '''

    # assert aug_times == len(aug_type_list)
    
    if aug_times == 8:
        aug_type_list=[[None], ['rot_anti_cw_90'], 
                       ['rot_anti_cw_180'], ['rot_anti_cw_270'],
                       ['flipud'], ['flipud', 'rot_anti_cw_90'], 
                       ['flipud', 'rot_anti_cw_180'], ['flipud', 'rot_anti_cw_270']]
    elif aug_times == 4:
        aug_type_list=[[None], ['rot_anti_cw_90'], 
                       ['flipud'], ['flipud', 'rot_anti_cw_90']]
    elif aug_times == 2:
        aug_type_list=[[None], ['rot_anti_cw_90']]
    elif aug_times == 1:
        aug_type_list = [[None]]

    s = sample_images_4dor3d.shape

    sample_augmented = [transform_images_onetype(sample_images_4dor3d.reshape((s[0], 1, s[-2], s[-1])), aug_type=aug_type_sing) for aug_type_sing in aug_type_list]
#    sample_augmented = np.concatenate(sample_augmented, axis=0)
#    assert len(sample_augmented) == aug_times * len(sample_images)
    # sample_augmented = [np.expand_dims(i, 1) for i in sample_augmented]
    return sample_augmented


def augment_images_auto(sample_images_4dor3d, aug_times=8):
    '''
    return sample_augmented (list), each element has the same dim as input sample_images_4dor3d
    '''
    # assert aug_times == len(aug_type_list)

    if len(sample_images_4dor3d.shape) >=3:
        if aug_times == 8:
            aug_type_list=[[None], ['rot_anti_cw_90'], 
                           ['rot_anti_cw_180'], ['rot_anti_cw_270'],
                           ['flipud'], ['flipud', 'rot_anti_cw_90'], 
                           ['flipud', 'rot_anti_cw_180'], ['flipud', 'rot_anti_cw_270']]
        elif aug_times == 4:
            aug_type_list=[[None], ['rot_anti_cw_90'], 
                           ['flipud'], ['flipud', 'rot_anti_cw_90']]
        elif aug_times == 2:
            aug_type_list=[[None], ['rot_anti_cw_90']]
        elif aug_times == 1:
            aug_type_list = [[None]]
    
        # s = sample_images_4dor3d.shape
        sample_augmented = [transform_images_onetype(sample_images_4dor3d, aug_type=aug_type_sing) for aug_type_sing in aug_type_list]

    else:
        sample_augmented = [sample_images_4dor3d for i in range(aug_times)]

    return sample_augmented



def transform_images_onetype(samples, aug_type=['rot_anti_cw_90']):
    '''
    samples, [n, h, w]
    aug_type_list: e.g. ['flipud', 'rot_anti_cw_270'] for successive operations
        optional list element: 'rot_anti_cw_90', # i.e. 'rot_cw_270'
                               'rot_anti_cw_180', # i.e. 'rot_cw_180'
                               'rot_anti_cw_270', # i.e. 'rot_cw_90'
                               'flipud',
                               'fliplr',
                               'None'
                              anything else generates original input samples as output
    return samples_transformed, np.ndarray, the same shape as the input samples
    '''
    # samples_shape = samples.shape
    # assert len(samples_shape) == 3 or len(samples_shape) == 4
    
    samples_transformed = samples.copy()
    for aug in aug_type:
        # print('aug type', aug)
        if aug == 'rot_anti_cw_90' or aug == 'rot_cw_270':
            samples_transformed = np.rot90(samples_transformed,
                                           k=1, axes=(-2, -1))
        elif aug == 'rot_anti_cw_180' or aug == 'rot_cw_180':
            samples_transformed = np.rot90(samples_transformed,
                                           k=2, axes=(-2, -1))
        elif aug == 'rot_anti_cw_270' or aug == 'rot_cw_90':
            samples_transformed = np.rot90(samples_transformed,
                                           k=1, axes=(-1, -2))
        elif aug == 'flipud':
            samples_transformed = np.flip(samples_transformed, axis=-2)
        elif aug == 'fliplr':
            samples_transformed = np.flip(samples_transformed, axis=-1)
        elif aug == 'None':
            samples_transformed = samples_transformed
    return samples_transformed



if __name__ == '__main__':
    from sklearn.datasets import load_digits
    
    # transform_images_onetype
    print(" > This is a demo for transform_images_onetype, SELF")
    digits = load_digits()
    samples_collect = digits.images[1:3]
    # samples_collect = np.expand_dims(samples_collect, 1)
    # samples_aug = transform_images_onetype(samples_collect, aug_type=['fliplr'])
    # print(samples_collect.shape, samples_aug.shape)
    # for img1, img2 in zip(samples_collect, samples_aug):
    #     plt.imshow(np.squeeze(img1))
    #     plt.show()
    #     plt.imshow(np.squeeze(img2))
    #     plt.show()
    # print()

    for img in samples_collect:
        plt.imshow(np.squeeze(img))
        plt.show()
        img2 = transform_images_onetype(img, aug_type=['fliplr'])
        plt.imshow(np.squeeze(img2))
        plt.show()
        print()

    #%%
    # # augment_images
    # print(" > This is a demo for augment_images_4d_auto, None")
    # digits = load_digits()
    # samples_collect = digits.images[1:3]
    # # samples_aug = augment_images(samples_collect, aug_times=4, 
    # #                              aug_type_list=[[None], 
    # #                                             ['rot_anti_cw_90'], 
    # #                                             ['rot_anti_cw_180'], 
    # #                                             ['rot_anti_cw_270']])
    # samples_aug = augment_images_4d_auto(samples_collect, aug_times=1)
    # print(samples_collect.shape, [i.shape for i in samples_aug])
    # for img_set in samples_aug:
    #     for img in img_set:
    #         plt.imshow(np.squeeze(img))
    #         plt.show()
    

#%%
# == OLD VERSION1 ====
#import random
#def upbalancing(samples, sample_idx_, total_num, c_lists=[]):
#    '''
#    augment samples by 4 types of implementations:
#        up-down flip,
#        left-right flip,
#        clockwise rotation by 90 degrees,
#        and anti-clockwise rotation by 90 degrees
#    @ Args:
#        samples (3D np.array): [n,h,w], the whole sample array
#        sample_idx_ (1D np.array): [m,], indices of samples for augmentation
#        total_num (int):
#            the total number the selected samples will be augmented to
#            (including initial selected samples)
#        c_lists (list):
#            A list of other acompany np.array associated with samples
#            Each accompany np.array's element index refers to image index,
#            i.e. len(one_company_list) == len(samples)
#    '''
#    samples = samples[sample_idx_]
#    sample_num = len(sample_idx_)
#    aug_num = total_num - sample_num
#
#    fliplr_num = int(0.2 * aug_num)
#    flipud_num = int(0.2 * aug_num)
#    row_cw_num = int(0.2 * aug_num)
#    rot_anti_cw_num = aug_num - fliplr_num - flipud_num - row_cw_num
##    print('sample number for each augmentatio type')
##    print('original_sample_num, fliplr_num, flipud_num, row_cw_num, rot_anti_cw_num')
##    print(n, fliplr_num, flipud_num, row_cw_num, rot_anti_cw_num)
#
#    samples_augmented_ = []
#    c_lists_augmented_ = [[] for i in c_lists]
#    c_lists_valid = [one_list[sample_idx_] for one_list in c_lists]
#
#    # original samples
#    samples_augmented_.extend(samples)
#    for one_list_valid, one_list_augmented in zip(c_lists_valid, c_lists_augmented_):
#        one_list_augmented.extend(one_list_valid)
#
#    # generate aug
#    aug_type_list = ['fliplr', 'flipud', 'rot_cw', 'rot_anti_cw']
#    aug_num_list = [fliplr_num, flipud_num, row_cw_num, rot_anti_cw_num]
#    for one_type_aug, aug_num_sing in zip(aug_type_list, aug_num_list):
#        samp, c_lists_aug = upsampling_one_type(samples, aug_num_sing, c_lists_=c_lists_valid,
#                                                aug_type=one_type_aug)
#        samples_augmented_.extend(samp)
##        print(len(samp))
##        print(len(c_lists_aug[0]))
#        for one_list_aug, one_list_augmented in zip(c_lists_aug, c_lists_augmented_):
#            one_list_augmented.extend(one_list_aug)
#
#    samples_augmented_ = np.array(samples_augmented_)
##    print('samples_augmented_.shape', samples_augmented_.shape)
#    return samples_augmented_, c_lists_augmented_
#
#
#def upsampling_one_type(samples, aug_num, c_lists=[], aug_type=None):
#    '''
#    augment selected samples to a certain number (aug_num) by one augmentation implementation
#    @ Args:
#        samples (3D np.array): [m,h,w] selected sample array
#    '''
#    samples_collect = []
#    c_lists_aug = [[] for one_list in c_lists]
#
#    sample_num = samples.shape[0]
##    print('samples_.shape', samples_.shape)
#    times = aug_num // sample_num
#    if times >= 1:
#        for i in range(times):
#            samples_collect.extend(samples)
#            if len(c_lists) > 0:
#                for one_list_aug, one_list in zip(c_lists_aug, c_lists):
#                    one_list_aug.extend(one_list)
#    rest = aug_num % sample_num
#    if rest > 0:
#        idx = random.sample(range(sample_num), rest)
#        samples_collect.extend(samples[idx])
#        if len(c_lists) > 0:
#            for one_list_aug, one_list in zip(c_lists_aug, c_lists):
#                one_list_aug.extend(one_list[idx])
#
#    samples_collect = np.array(samples_collect)
##    print(samples_collect.shape)
#    if len(samples_collect) > 0:
#        if aug_type == 'fliplr':
#            samples_aug = np.flip(samples_collect, axis=2)
#        elif aug_type == 'rot_cw':
#            samples_aug = np.rot90(samples_collect, k=3, axes=(1, 2))
#        elif aug_type == 'rot_anti_cw':
#            samples_aug = np.rot90(samples_collect, k=1, axes=(1, 2))
#        elif aug_type == 'flipud':
#            samples_aug = np.flip(samples_collect, axis=1)
#    else:
#        samples_aug = []
#
#    return samples_aug, c_lists_aug


#%%
# == OLD VERSIOIN2 ====
#def data_augmentation(imgs, codes, names, w, h, area, flag,verbose = False):
##    imgs = imgs.reshape(imgs.shape[0], imgs.shape[2], imgs.shape[3])
#    imgs = np.squeeze(imgs)
#    imgs_output = list()
#    codes_output = list()
#    names_output = list()
#    w_output = list()
#    h_output = list()
#    area_output = list()
#
#    for s in range(imgs.shape[0]):
#        img = imgs[s]
#        imgs_output.append(img)
#        codes_output.append(codes[s])
#        names_output.append(names[s])
#        w_output.append(w[s])
#        h_output.append(h[s])
#        area_output.append(area[s])
#        HEIGHT = img.shape[1]
#        WIDTH = img.shape[0]
#
#        if 'flipping' in flag:
#            # Flipping images with Numpy
#            flipped_img = np.fliplr(img)
#            imgs_output.append(flipped_img)
#            codes_output.append(codes[s])
#            names_output.append(names[s]+'.flipping')
#            w_output.append(w[s])
#            h_output.append(h[s])
#            area_output.append(area[s])
#
#        # get mean
#        mean = 0
#        for j in range(WIDTH):
#            for i in range(HEIGHT):
#                mean += img[j][i]
#        mean /= (WIDTH * HEIGHT)
#
#        if 'shift_right' in flag:
#            # Shifting right
#            img_right = np.copy(img)
#            for i in range(HEIGHT):
#                for j in range(WIDTH):
#                    if (i > 5):
#                        img_right[j][i] = img[j][i - 5]
#                    else:
#                        img_right[j][i] = mean
#            imgs_output.append(img_right)
#            codes_output.append(codes[s])
#            names_output.append(names[s]+'.shift_right')
#            w_output.append(w[s])
#            h_output.append(h[s])
#            area_output.append(area[s])
#
#        if 'shift_left' in flag:
#            # Shifting left
#            img_left = np.copy(img)
#            for j in range(WIDTH):
#                for i in range(HEIGHT):
#                    if (i < HEIGHT - 5):
#                        img_left[j][i] = img[j][i + 5]
#                    else:
#                        img_left[j][i] = mean
#            imgs_output.append(img_left)
#            codes_output.append(codes[s])
#            names_output.append(names[s]+'.shift_left')
#            w_output.append(w[s])
#            h_output.append(h[s])
#            area_output.append(area[s])
#
#        if 'shift_up' in flag:
#            # Shifting Up
#            img_up = np.copy(img)
#            for j in range(WIDTH):
#                for i in range(HEIGHT):
#                    if (j < WIDTH - 5):
#                        img_up[j][i] = img[j + 5][i]
#                    else:
#                        img_up[j][i] = mean
#            imgs_output.append(img_up)
#            codes_output.append(codes[s])
#            names_output.append(names[s]+'.shift_up')
#            w_output.append(w[s])
#            h_output.append(h[s])
#            area_output.append(area[s])
#
#        if 'shift_down' in flag:
#            # Shifting Down
#            img_down = np.copy(img)
#            for j in range(WIDTH):
#                for i in range(HEIGHT):
#                    if (j > 5):
#                        img_down[j][i] = img[j - 5][i]
#                    else:
#                        img_down[j][i] = mean
#            imgs_output.append(img_down)
#            codes_output.append(codes[s])
#            names_output.append(names[s]+'.shift_down')
#            w_output.append(w[s])
#            h_output.append(h[s])
#            area_output.append(area[s])
#
#        if  'add_noise' in flag:
#            # ADDING NOISE
#            img_noise = np.copy(img)
#            noise = np.random.randint(5, size=(WIDTH, HEIGHT), dtype='uint8')
#            for i in range(WIDTH):
#                for j in range(HEIGHT):
#                    if (img[i][j] != 255):
#                        img_noise[i][j] += noise[i][j]
#            imgs_output.append(img_noise)
#            codes_output.append(codes[s])
#            names_output.append(names[s]+'.add_noise')
#            w_output.append(w[s])
#            h_output.append(h[s])
#            area_output.append(area[s])
#
#        if verbose:
#            plt.imshow(img, cmap = 'gray')
#            plt.show()
#            plt.imshow(flipped_img, cmap = 'gray')
#            plt.show()
#            plt.imshow(img_right, cmap = 'gray')
#            plt.show()
#            plt.imshow(img_left, cmap = 'gray')
#            plt.show()
#            plt.imshow(img_up, cmap = 'gray')
#            plt.show()
#            plt.imshow(img_down, cmap = 'gray')
#            plt.show()
#            plt.imshow(img_noise, cmap = 'gray')
#            plt.show()
#
#    imgs_output = np.array(imgs_output)
#    imgs_output = imgs_output.reshape(imgs_output.shape[0],-1,imgs_output.shape[1],imgs_output.shape[2])
#    codes_output = np.array(codes_output)
#    names_output = np.array(names_output)
#    w_output = np.array(w_output)
#    h_output = np.array(h_output)
#    area_output = np.array(area_output)
#    return imgs_output, codes_output, names_output, w_output, h_output, area_output


