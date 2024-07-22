import numpy as np
# import cv2
import torch
from torch.nn import functional as f



class HANDCRAFT_FILTERS():
    def __init__(self):
        pass

    def gene_laws_kernels(self):
        c1 = 1.0/np.sqrt(3.0) * np.array([1, 1, 1]).reshape((-1, 1))
        c2 = 1.0/np.sqrt(2.0) * np.array([-1, 0, 1]).reshape((-1, 1))
        c3 = 1.0/np.sqrt(6.0) * np.array([-1, 2, -1]).reshape((-1, 1))
        col_bank = [c1, c2, c3]
        r1 = 1.0/np.sqrt(3.0) * np.array([1, 1, 1]).reshape((1, -1))
        r2 = 1.0/np.sqrt(2.0) * np.array([-1, 0, 1]).reshape((1, -1))
        r3 = 1.0/np.sqrt(6.0) * np.array([-1, 2, -1]).reshape((1, -1))
        row_bank = [r1, r2, r3]
        return col_bank, row_bank

    def gene_haar_kernels(self):
        c1 = 1.0/np.sqrt(2.0) * np.array([1, 1]).reshape((-1, 1))
        c2 = 1.0/np.sqrt(2.0) * np.array([1, -1]).reshape((-1, 1))
        col_bank = [c1, c2]
        r1 = 1.0/np.sqrt(2.0) * np.array([1, 1]).reshape((1, -1))
        r2 = 1.0/np.sqrt(2.0) * np.array([-1, 1]).reshape((1, -1))
        row_bank = [r1, r2]
        return col_bank, row_bank

    def gene_sobel_gradientx_kernels(self):
        c = np.array([1, 2, 1]).reshape((-1, 1))
        col_bank = [c]
        r = np.array([-1, 0, 1]).reshape((1, -1))
        row_bank = [r]
        return col_bank, row_bank


    def gene_sobel_gradienty_kernels(self):
        c = np.array([1, 0, -1]).reshape((-1, 1))
        col_bank = [c]
        r = np.array([1, 2, 1]).reshape((1, -1))
        row_bank = [r]
        return col_bank, row_bank




    def get_filter_bank(self, rep_type):
        if rep_type == 'haar':
            c_bank, r_bank = self.gene_haar_kernels()
        if rep_type == 'laws':
            c_bank, r_bank = self.gene_laws_kernels()
        if rep_type == 'sobel_x':
            c_bank, r_bank = self.gene_sobel_gradientx_kernels()
        if rep_type == 'sobel_y':
            c_bank, r_bank = self.gene_sobel_gradienty_kernels()
        filter_bank = np.array([np.dot(c, r) for c in c_bank for r in r_bank])
        filter_bank = torch.from_numpy(np.expand_dims(filter_bank, 1)).to(torch.float)
        # print('filter_bank', filter_bank.shape)
        return filter_bank


    def get_filter_bank_2d(self, rep_type):
        '''
        col-wise kernels
        '''
        filters = self.get_filter_bank(rep_type)
        print(filters.reshape((len(filters), -1)).shape)
        filters = torch.transpose(filters.reshape((len(filters), -1)), -2, -1)
        return filters



    def get_inverse_filter_bank_2d(self, rep_type):
        '''
        row-wise kernels
        '''
        filters_2d = self.get_filter_bank_2d(rep_type)
        filters_inverse_2d = torch.inverse(filters_2d)
        return filters_inverse_2d






if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error as MSE

    hc_filters = HANDCRAFT_FILTERS()

    # laws_filters = hc_filters.get_filter_bank('laws')
    # print('laws_filters', laws_filters.shape) # torch.Size([9, 1, 3, 3])

    # laws_filters = hc_filters.get_filter_bank_2d('laws')
    # print('laws_filters', laws_filters.shape) # torch.Size([9, 9])

    sobel_x = hc_filters.get_filter_bank('sobel_x')
    print('sobel_x', sobel_x.shape) # torch.Size([9, 9])

    sobel_y = hc_filters.get_filter_bank('sobel_y')
    print('sobel_y', sobel_y.shape) # torch.Size([9, 9])



