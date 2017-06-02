from __future__ import division, print_function
import numpy as np
from scipy import linalg
import sympy


def find_roots(coef_row, coef_col):
    """
    Find the common roots of two bivariate polynomials with coefficients specified by
    two 2D arrays.
    the variation along the first dimension (i.e., columns) is in the incresing order of y.
    the variation along the second dimension (i.e., rows) is in the incresing order of x.
    :param coef_row: polynomial coefficients the first polynomial for the annihilation along rows
    :param coef_col: polynomial coefficients the second polynomial for the annihilation along cols
    :return:
    """
    # assert coef_col.shape[0] >= coef_row.shape[0] and coef_row.shape[1] >= coef_col.shape[1]
    if coef_row.shape[1] < coef_col.shape[1]:
        # swap input coefficients
        coef_row, coef_col = coef_col, coef_row
    x, y = sympy.symbols('x, y')  # build symbols
    # collect both polynomials as a function of x; y will be included in the coefficients
    poly1 = 0
    poly2 = 0

    max_row_degree_x = coef_row.shape[1] - 1
    max_row_degree_y = coef_row.shape[0] - 1
    for x_count in range(max_row_degree_x + 1):
        for y_count in range(max_row_degree_y + 1):
            poly1 += coef_row[y_count, x_count] * x ** (max_row_degree_x - x_count) * \
                     y ** (max_row_degree_y - y_count)

    max_col_degree_x = coef_col.shape[1] - 1
    max_col_degree_y = coef_col.shape[0] - 1
    for x_count in range(max_col_degree_x + 1):
        for y_count in range(max_col_degree_y + 1):
            poly2 += coef_col[y_count, x_count] * x ** (max_col_degree_x - x_count) * \
                     y ** (max_col_degree_y - y_count)

    poly1_x = sympy.Poly(poly1, x)
    poly2_x = sympy.Poly(poly2, x)

    K = max_row_degree_x  # highest power of the first polynomial (in x)
    L = max_col_degree_x  # highest power of the second polynomial (in x)

    if coef_row.shape[0] == 1:  # i.e., independent of variable y
        x_roots_all = np.roots(coef_row.squeeze())
        eval_poly2 = sympy.lambdify(x, poly2)
        x_roots = []
        y_roots = []
        for x_loop in x_roots_all:
            y_roots_loop = np.roots(np.array(sympy.Poly(eval_poly2(x_loop), y).all_coeffs(), dtype=complex))
            y_roots.append(y_roots_loop)
            x_roots.append(np.tile(x_loop, y_roots_loop.size))
    elif coef_col.shape[1] == 1:  # i.e., independent of variable x
        y_roots_all = np.roots(coef_col.squeeze())
        eval_poly1 = sympy.lambdify(y, poly1)
        x_roots = []
        y_roots = []
        for y_loop in y_roots_all:
            x_roots_loop = np.roots(np.array(sympy.Poly(eval_poly1(y_loop), x).all_coeffs(), dtype=complex))
            x_roots.append(x_roots_loop)
            y_roots.append(np.tile(y_loop, x_roots_loop.size))
    else:
        if L>=1:
            toep1_r = np.hstack((poly1_x.all_coeffs()[::-1], np.zeros(L - 1)))
            toep1_r = np.concatenate((toep1_r, np.zeros(L + K - toep1_r.size)))
            toep1_c = np.concatenate(([poly1_x.all_coeffs()[-1]], np.zeros(L - 1)))
        else:  # for the case with L == 0
            toep1_r = np.zeros((0, L + K))
            toep1_c = np.zeros((0, 0))

        if K >= 1:
            toep2_r = np.hstack((poly2_x.all_coeffs()[::-1], np.zeros(K - 1)))
            toep2_r = np.concatenate((toep2_r, np.zeros(L + K - toep2_r.size)))
            toep2_c = np.concatenate(([poly2_x.all_coeffs()[-1]], np.zeros(K - 1)))
        else:  # for the case with K == 0
            toep2_r = np.zeros((0, L + K))
            toep2_c = np.zeros((0, 0))

        blk_mtx1 = linalg.toeplitz(toep1_c, toep1_r)
        blk_mtx2 = linalg.toeplitz(toep2_c, toep2_r)
        if blk_mtx1.size != 0 and blk_mtx2.size != 0:
            # for debugging only
            # print('blk_mtx1 size: {0}, blk_mtx2_size: {1}'.format(blk_mtx1.shape, blk_mtx2.shape))
            mtx = np.vstack((blk_mtx1, blk_mtx2))
        elif blk_mtx1.size == 0 and blk_mtx2.size != 0:
            mtx = blk_mtx2
        elif blk_mtx1.size !=0 and blk_mtx2.size == 0:
            mtx = blk_mtx1
        else:
            mtx = np.zeros((0, 0))

        max_poly_degree = np.int(max_row_degree_y * L + max_col_degree_y * K)
        num_samples = (max_poly_degree + 1)

        # Construct the system matrix to compute the resultant
        # (determinant of the polynomial matrix)
        # We use the Fourier basis to sample the polynomial
        y_vals = np.exp(2j * np.pi * np.arange(num_samples) / num_samples)[:,None]
        y_powers = np.reshape(np.arange(max_poly_degree + 1)[::-1], (1, -1), order='F')
        Y = y_vals ** y_powers

        # compute resultant, which is the determinant of mtx.
        # it is a polynomial in terms of variable y
        func_resultant = sympy.lambdify(y, sympy.Matrix(mtx))
        det_As = np.array([linalg.det(np.array(func_resultant(y_roots_loop), dtype=complex))
                           for y_roots_loop in y_vals.squeeze()], dtype=complex)

        coef_resultant = linalg.lstsq(Y, det_As)[0]
        
        # coef_resultant = linalg.solve(Y, det_As)
        y_roots_all = np.roots(coef_resultant)

        # use the root values for y to find the root values for x
        func_loop = sympy.lambdify(y, poly2_x.all_coeffs())
        # x_roots = np.zeros(y_roots.size)
        x_roots = []
        y_roots = []
        for loop in range(y_roots_all.size):
            y_roots_loop = y_roots_all[loop]
            x_roots_loop = np.roots(func_loop(y_roots_loop))
            # poly_val = np.abs(func_eval(x_roots_loop, y_roots_loop))
            x_roots.append(x_roots_loop)
            y_roots.append(np.tile(y_roots_loop, x_roots_loop.size))

    return np.array(x_roots), np.array(y_roots)


def check_error(coef, x, y):
    val = 0
    max_degree_x = coef.shape[1] - 1
    max_degree_y = coef.shape[0] - 1
    for x_count in range(max_degree_x + 1):
        for y_count in range(max_degree_y + 1):
            val += coef[y_count, x_count] * x ** (max_degree_x - x_count) * \
                   y ** (max_degree_y - y_count)
    return val


if __name__ == '__main__':
    coef_row = np.random.randn(3, 4)
    coef_col = np.random.randn(4, 3)
    x_roots, y_roots = find_roots(coef_row, coef_col)
    # print x_roots, y_roots
    print(np.abs(check_error(coef_row, x_roots, y_roots)))
