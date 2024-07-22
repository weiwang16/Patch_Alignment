# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
# cimport numpy as cnp
cimport numpy as np
cimport cython

# from .._shared.fused_numerics cimport np_floats
# cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float32_t cell_hog(np.ndarray[np.float32_t, ndim=2]magnitude,
             np.ndarray[np.float32_t, ndim=2] orientation,
             np.float32_t orientation_start, np.float32_t orientation_end,
             int cell_columns, int cell_rows,
             int column_index, int row_index,
             int size_columns, int size_rows,
             int range_rows_start, int range_rows_stop,
             int range_columns_start, int range_columns_stop):
    """Calculation of the cell's HOG value
    Parameters
    ----------
    magnitude : ndarray
        The gradient magnitudes of the pixels.
    orientation : ndarray
        Lookup table for orientations.
    orientation_start : float
        Orientation range start.
    orientation_end : float
        Orientation range end.
    cell_columns : int
        Pixels per cell (rows). i.e. cell size
    cell_rows : int
        Pixels per cell (columns). i.e. cell size
    column_index : int
        Block column index. the idx of central pixel
    row_index : int
        Block row index. the idx of central pixel
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    range_rows_start : int
        Start row of cell. #the local idx of each pixel in the cell w.r.t. the central pixel
    range_rows_stop : int
        Stop row of cell.
    range_columns_start : int
        Start column of cell.
    range_columns_stop : int
        Stop column of cell
    Returns
    -------
    total : float
        The total HOG value.
    """
    cdef int cell_column, cell_row, cell_row_index, cell_column_index
    cdef np.float32_t total = 0.
    # total = 0.0

    for cell_row in range(range_rows_start, range_rows_stop):
        cell_row_index = row_index + cell_row
        if (cell_row_index < 0 or cell_row_index >= size_rows):
            continue

        for cell_column in range(range_columns_start, range_columns_stop):
            cell_column_index = column_index + cell_column
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[cell_row_index, cell_column_index]
                    >= orientation_start
                    or orientation[cell_row_index, cell_column_index]
                    < orientation_end):
                continue

            total += magnitude[cell_row_index, cell_column_index]

    return total / (cell_rows * cell_columns)


def grad_histograms(gradient_x,
                   gradient_y,
                   cell_columns, cell_rows,
                   size_columns, size_rows,
                   # int number_of_cells_columns, int number_of_cells_rows,
                   number_of_orientations):#,
                   # orientation_histogram):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.
       Cells are non-overlapping in the given image.

    Parameters
    ----------
    gradient_x : ndarray
        First order image gradients along x (along row, across col).
    gradient_y : ndarray
        First order image gradients along y (along col, across rows).
    cell_columns : int
        Pixels per cell (rows).
    cell_rows : int
        Pixels per cell (columns).
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    # number_of_cells_columns : int
    #     Number of cells (rows).
    # number_of_cells_rows : int
    #     Number of cells (columns).
    number_of_orientations : int
        Number of orientation bins.
    # orientation_histogram : ndarray
    #     The histogram array which is modified in place.
    """

    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = np.rad2deg(np.arctan2(gradient_y, gradient_x)) #% 180, use signed angle in [-pi, pi]
    # orientation = orientation % 180 + (-180) * (orientation // 180)
    # print((orientation <0).shape)
    orientation[orientation <0] = 360.0 + orientation[orientation <0]
    # print(np.min(orientation.ravel()), np.max(orientation.ravel()))

    # cdef int i, c, r, r_i, c_i, cc, cr, c_0, r_0, \
    #     range_rows_start, range_rows_stop, \
    #     range_columns_start, range_columns_stop
    # cdef np_floats orientation_start, orientation_end, \
    #     number_of_orientations_per_180
    # cdef int number_of_cells_columns, number_of_cells_rows

    number_of_cells_columns = int(size_columns // cell_columns)
    number_of_cells_rows = int(size_rows // cell_rows)

    r_0 = int(cell_rows / 2)
    c_0 = int(cell_columns / 2)
    cc = int(cell_rows * number_of_cells_rows)
    cr = int(cell_columns * number_of_cells_columns)

    range_rows_stop = int((cell_rows + 1) / 2) #the local idx of each pixel in the cell w.r.t. the central pixel
    range_rows_start = -1 * int((cell_rows / 2))
    range_columns_stop = int((cell_columns + 1) / 2)
    range_columns_start = -1 * int((cell_columns / 2))
    number_of_orientations_per_180 = 360. / number_of_orientations #180. / number_of_orientations
    orientation_histogram = np.zeros((number_of_cells_rows, number_of_cells_columns, number_of_orientations))

    # init_angle = -180 - number_of_orientations_per_180 * 1.0/2
    init_angle = 0 - number_of_orientations_per_180 * 1.0/2
    # with nogil:
        # compute orientations integral images
    for i in range(number_of_orientations+1):
        # isolate orientations in this range
        orientation_start = number_of_orientations_per_180 * (i + 1) + init_angle
        orientation_end = number_of_orientations_per_180 * i + init_angle
        c = c_0 #the idx of central pixel, init as cell_rows / 2
        r = r_0 #the idx of central pixel, init as cell_rows / 2
        r_i = 0 #the local idx of each pixel in the cell w.r.t. the upper left corner
        c_i = 0 #the local idx of each pixel in the cell  w.r.t. the upper left corner

        while r < cc:
            c_i = 0
            c = c_0

            while c < cr:
                temp = cell_hog(magnitude, orientation,
                             orientation_start, orientation_end,
                             cell_columns, cell_rows, c, r,
                             size_columns, size_rows,
                             range_rows_start, range_rows_stop,
                             range_columns_start, range_columns_stop)
                orientation_histogram[r_i, c_i, i%number_of_orientations] += temp
                c_i += 1
                c += cell_columns

            r_i += 1
            r += cell_rows
    return orientation_histogram #[num_call_in_h, num_cell_in_w, num_hist_bins]



'''
https://github.com/scikit-image/scikit-image/blob/5e74a4a3a5149a8a14566b81a32bb15499aa3857/skimage/feature/_hoghistogram.pyx

cython
https://zhuanlan.zhihu.com/p/24311879
'''

