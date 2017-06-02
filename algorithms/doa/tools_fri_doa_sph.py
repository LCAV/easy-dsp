from __future__ import division, print_function
# import datetime
# import time
import numpy as np
import scipy as sp
from scipy import linalg
import scipy.special
import scipy.misc
import scipy.optimize
from functools import partial
# from joblib import Parallel, delayed, cpu_count
# import os
# import scipy.sparse as sps

import matplotlib.pyplot as plt

# from tools import sph_distance

from .poly_commn_roots import find_roots

def sph2cart(r, colatitude, azimuth):
    """
    spherical to cartesian coordinates
    :param r: radius
    :param colatitude: co-latitude
    :param azimuth: azimuth
    :return:
    """
    r_sin_colatitude = r * np.sin(colatitude)
    x = r_sin_colatitude * np.cos(azimuth)
    y = r_sin_colatitude * np.sin(azimuth)
    z = r * np.cos(colatitude)
    return x, y, z


def sph_cov_mtx_est(y_mic):
    """
    estimate covariance matrix based on the received signals at microphones
    :param y_mic: received signal (complex base-band representation) at microphones
    :return:
    """
    # Q: total number of microphones
    # num_snapshot: number of snapshots used to estimate the covariance matrix
    # num_bands: number of sub-bands considered
    Q, num_snapshot, num_bands = y_mic.shape
    cov_mtx = np.zeros((Q, Q, num_bands), dtype=complex)
    for band in range(num_bands):
        for q in range(Q):
            y_mic_outer = y_mic[q, :, band]
            for qp in range(Q):
                y_mic_inner = y_mic[qp, :, band]
                cov_mtx[qp, q, band] = np.dot(y_mic_outer, y_mic_inner.T.conj())
    return cov_mtx / num_snapshot


def sph_extract_off_diag(mtx):
    """
    extract off-diagonal entries in mtx
    The output vector is order in a column major manner
    :param mtx: input matrix to extract the off-diagonal entries
    :return:
    """
    # we transpose the matrix because the function np.extract will first flatten the matrix
    # withe ordering convention 'C' instead of 'F'!!
    Q = mtx.shape[0]
    num_bands = mtx.shape[2]
    extract_cond = np.reshape((1 - np.eye(Q)).T.astype(bool), (-1, 1), order='F')
    return np.column_stack([np.reshape(np.extract(extract_cond, mtx[:, :, band].T),
                                       (-1, 1), order='F')
                            for band in range(num_bands)])


# ======= functions for the joint estimation of colatiudes and azimuth =======
def sph_gen_dirty_img(a, p_mic_x, p_mic_y, p_mic_z,
                      omega_bands, sound_speed, L, num_pixel_plt=5e3):
    """
    Compute the dirty image associated with the given visibility measurements.
    :param a: the visibility measurements (i.e., cross-correlation between received
            signals at different microphones)
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param p_mic_z: a vector that contains microphones' z-coordinates
    :param omega_bands: mid-band (ANGULAR) frequencies [radian/sec]
    :param sound_speed: speed of sound
    :param L: maximum degree of spherical harmonics
    :return:
    """
    # define the plotting grid where the least square solution is evaluated
    bound = np.ceil((1 + np.sqrt(1 + 8 * num_pixel_plt)) / 4.)
    azimuth_plt, colatitude_plt = np.meshgrid(-np.pi + 2 * np.pi / (2. * bound - 1) *
                                              np.arange(2 * bound),
                                              np.pi / (2. * bound - 1) *
                                              np.append(2 * np.arange(bound),
                                                        2 * bound - 1)[::-1])

    norm_factor = np.reshape(sound_speed / omega_bands, (1, -1), order='F')
    # normalised antenna coordinates in a matrix form
    # each column corresponds to the location in one subband,
    # i.e., the second dimension corresponds to different subbands
    # NOTE: The NEGATIVE sign here converts DOA to the propagation vector
    p_mic_x_normalised = np.reshape(-p_mic_x, (-1, 1), order='F') / norm_factor
    p_mic_y_normalised = np.reshape(-p_mic_y, (-1, 1), order='F') / norm_factor
    p_mic_z_normalised = np.reshape(-p_mic_z, (-1, 1), order='F') / norm_factor

    # linear transformation matrix that maps uniform samples of
    # sinusoids to visibilities
    mtx_freq2visibility = \
        sph_mtx_freq2visibility(L, p_mic_x_normalised,
                                p_mic_y_normalised, p_mic_z_normalised)

    # min. least square estimate
    f_hat_lsq = linalg.lstsq(np.row_stack(mtx_freq2visibility),
                             np.reshape(a, (-1, 1), order='F'))[0]
    # reshape the vector in a lower triangle form
    '''
    0 0 x 0 0
    0 x x x 0
    x x x x x
    '''
    f_hat_lsq = low_tri_from_vec(f_hat_lsq.squeeze())

    # evaluate the least square solution on the plotting grid defined by
    # colatitude_plt and azimuth_plt
    img = np.zeros(colatitude_plt.shape, dtype=complex)

    # loop over the spherical harmonics indices
    for l in range(L + 1):
        for m in range(-l, l + 1):
            img += f_hat_lsq[l, m + L] * sph_harm_ufnc(l, m, colatitude_plt, azimuth_plt)

    return img, colatitude_plt, azimuth_plt


def sph_recon_2d_dirac_joint(a, p_mic_x, p_mic_y, p_mic_z, omega_bands,
                             sound_speed, K, L, noise_level, max_ini=20,
                             stop_cri='max_iter', max_iter=20, num_rotation=1, G_iter=1,
                             signal_type='visibility', thresh_reliable=0.1,
                             use_lu=True, verbose=False, symb=False,
                             use_GtGinv=False, **kwargs):
    """
    INTERFACE for joint estimation of the Dirac deltas azimuths and co-latitudes
    :param a: the measured visibilities in a matrix form, where the second dimension
              corresponds to different subbands
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param p_mic_z: a vector that contains microphones' z-coordinates
    :param omega_bands: mid-band (ANGULAR) frequencies [radian/sec]
    :param sound_speed: speed of sound
    :param K: number of point sources
    :param L: maximum degree of spherical harmonics
    :param noise_level: noise level in the measured visibilities
    :param max_ini: maximum number of random initialisation used
    :param stop_cri: either 'mse' or 'max_iter'
    :param num_rotation: number of random rotation applied. if num_rotation == 1, then
            no rotation is applied.
    :param G_iter: number of iterations used to update the linear mapping from the FRI
            sequence to the visibility measurements based on the reconstructed Dirac
            locations. If G_iter == 1, then no update.
    :param signal_type: The type of the signal a, possible values are 'visibility' 
            for covariance matrix
    :param use_lu: whether to use LU decomposition to improved efficiency or not
    :param verbose: whether output intermediate results for debugging or not
    :return:
    """
    if 'colatitudek_doa_ref' in kwargs and 'azimuthk_doa_ref' in kwargs:
        ref_sol_available = True
        colatitudek_doa_ref = kwargs['colatitudek_doa_ref']
        azimuthk_doa_ref = kwargs['azimuthk_doa_ref']
        K_ref = colatitudek_doa_ref.size
        # convert DOA to source locations
        azimuthk_ref = np.mod(azimuthk_doa_ref - np.pi, 2 * np.pi)
        colatitudek_ref = np.pi - colatitudek_doa_ref
        xyz_ref = np.row_stack(sph2cart(1, colatitudek_ref, azimuthk_ref))
    else:
        ref_sol_available = False
        K_ref = 0
    # whether update the linear mapping or not
    update_G = (G_iter != 1)

    # p_mic_x = np.squeeze(-p_mic_x.copy())
    # p_mic_y = np.squeeze(-p_mic_y.copy())
    # p_mic_z = np.squeeze(-p_mic_z.copy())

    num_bands = np.array(omega_bands).size  # number of bands considered

    if len(a.shape) == 2:
        assert a.shape[1] == num_bands

    norm_factor = np.reshape(sound_speed / omega_bands, (1, -1), order='F')
    # normalised antenna coordinates in a matrix form
    # each column corresponds to the location in one subband,
    # i.e., the second dimension corresponds to different subbands
    p_mic_x_normalised = np.reshape(p_mic_x, (-1, 1), order='F') / norm_factor
    p_mic_y_normalised = np.reshape(p_mic_y, (-1, 1), order='F') / norm_factor
    p_mic_z_normalised = np.reshape(p_mic_z, (-1, 1), order='F') / norm_factor

    # we use the real-valued representation of the measurements
    a_ri = np.row_stack((a.real, a.imag))

    min_error_all = float('inf')
    for count_rotate in range(num_rotation):
        # apply a random rotation
        # SO(3) is defined by three angles
        # if count_rotate == 0:  # <= i.e., no rotation
        #     rotate_angle1 = rotate_angle2 = rotate_angle3 = 0
        # else:
        rotate_angle1 = np.random.rand() * np.pi * 2
        rotate_angle2 = np.random.rand() * np.pi
        rotate_angle3 = np.random.rand() * np.pi * 2

        # build rotation matrix
        rotate_mtx1 = np.array([[np.cos(rotate_angle1), -np.sin(rotate_angle1), 0],
                                [np.sin(rotate_angle1), np.cos(rotate_angle1), 0],
                                [0, 0, 1]])
        rotate_mtx2 = np.array([[np.cos(rotate_angle2), 0, np.sin(rotate_angle2)],
                                [0, 1, 0],
                                [-np.sin(rotate_angle2), 0, np.cos(rotate_angle2)]])
        rotate_mtx3 = np.array([[np.cos(rotate_angle3), -np.sin(rotate_angle3), 0],
                                [np.sin(rotate_angle3), np.cos(rotate_angle3), 0],
                                [0, 0, 1]])
        rotate_mtx = np.dot(rotate_mtx1, np.dot(rotate_mtx2, rotate_mtx3))

        # rotate antenna steering vector
        p_mic_rotated = np.dot(rotate_mtx,
                               np.vstack((p_mic_x_normalised.flatten('F'),
                                          p_mic_y_normalised.flatten('F'),
                                          p_mic_z_normalised.flatten('F')))
                               )
        p_mic_x_rotated = np.reshape(p_mic_rotated[0, :], p_mic_x_normalised.shape, order='F')
        p_mic_y_rotated = np.reshape(p_mic_rotated[1, :], p_mic_y_normalised.shape, order='F')
        p_mic_z_rotated = np.reshape(p_mic_rotated[2, :], p_mic_z_normalised.shape, order='F')

        # linear transformation matrix that maps uniform samples of
        # sinusoids to visibilities
        mtx_freq2visibility = \
            sph_mtx_freq2visibility(L, p_mic_x_rotated,
                                    p_mic_y_rotated, p_mic_z_rotated)
        G_ri_lst = sph_mtx_fri2visibility_row_major(L, mtx_freq2visibility, aslist=True, symb=symb)

        for count_G in range(G_iter):
            # +2 here so that we first obtain the denoised data with a LARGER annihilating filter
            sz_coef = 2 + np.int(np.ceil(0.5 * (-1 + np.sqrt(1 + 8 * (K + 1)))))
            # sz_coef = sph_determin_max_coef_sz(L)
            K_alg = np.int((sz_coef + 1) * sz_coef * 0.5) - 2

            if count_G == 0:
                if ref_sol_available:
                    xyz_ref_rotated = np.dot(rotate_mtx, xyz_ref)
                    colatitudek_ref_rotated = np.arccos(xyz_ref_rotated[2, :])
                    azimuthk_ref_rotated = np.arctan2(xyz_ref_rotated[1, :],
                                                      xyz_ref_rotated[0, :])
                    G_amp_ref_ri_lst = [
                        sph_build_mtx_amp_ri(
                            p_mic_x_rotated[:, band_count],
                            p_mic_y_rotated[:, band_count],
                            p_mic_z_rotated[:, band_count],
                            azimuthk_ref_rotated,
                            colatitudek_ref_rotated
                        )
                        for band_count in range(num_bands)
                    ]

                    # b_opt_ri_lst = \
                    #     sph_dirac_recon_alg_joint(G_ri_lst, a_ri, K_alg, L, L, noise_level,
                    #                               max_ini, stop_cri, max_iter, use_lu=use_lu,
                    #                               symb=symb, use_GtGinv=use_GtGinv,
                    #                               G_amp_ref_ri_lst=G_amp_ref_ri_lst)[3]

                    c_row_opt, c_col_opt, min_error, b_opt_ri_lst, ini, \
                    sz_coef_row0, sz_coef_row1, sz_coef_col0, sz_coef_col1 = \
                        sph_dirac_recon_alg_joint(G_ri_lst, a_ri, K_alg, L, L, noise_level,
                                                  max_ini, stop_cri, max_iter,
                                                  use_lu=use_lu, symb=symb,
                                                  G_amp_ref_ri_lst=G_amp_ref_ri_lst)
                    azimuthk_recon, colatitudek_recon = \
                        sph_extract_innovation(
                            a_ri, K,
                            sph_reshape_coef(c_row_opt, sz_coef_row0, sz_coef_row1),
                            sph_reshape_coef(c_col_opt, sz_coef_col0, sz_coef_col1),
                            p_mic_x_rotated, p_mic_y_rotated, p_mic_z_rotated,
                            G_amp_ref_ri_lst=G_amp_ref_ri_lst,
                            colatitude_ref_rotated=colatitudek_ref_rotated,
                            azimuth_ref_rotated=azimuthk_ref_rotated
                        )
                else:
                    # b_opt_ri_lst = \
                    #     sph_dirac_recon_alg_joint(G_ri_lst, a_ri, K_alg, L, L, noise_level,
                    #                               max_ini, stop_cri, max_iter, use_lu=use_lu,
                    #                               symb=symb, use_GtGinv=use_GtGinv)[3]

                    c_row_opt, c_col_opt, min_error, b_opt_ri_lst, ini, \
                    sz_coef_row0, sz_coef_row1, sz_coef_col0, sz_coef_col1 = \
                        sph_dirac_recon_alg_joint(G_ri_lst, a_ri, K_alg, L, L, noise_level,
                                                  max_ini, stop_cri, max_iter,
                                                  use_lu=use_lu, symb=symb)
                    azimuthk_recon, colatitudek_recon = \
                        sph_extract_innovation(
                            a_ri, K,
                            sph_reshape_coef(c_row_opt, sz_coef_row0, sz_coef_row1),
                            sph_reshape_coef(c_col_opt, sz_coef_col0, sz_coef_col1),
                            p_mic_x_rotated, p_mic_y_rotated, p_mic_z_rotated)

                G_ri_lst = sph_update_G_ri(
                    colatitudek_recon[K_ref:],
                    azimuthk_recon[K_ref:],
                    L, p_mic_x_rotated,
                    p_mic_y_rotated,
                    p_mic_z_rotated,
                    num_bands, G_ri_lst, symb=symb)

            # use the denoised data b to solve the problem once more but with the
            # desired (smaller) filter size
            if ref_sol_available:
                c_row_opt, c_col_opt, min_error, b_opt_ri_lst, ini, \
                sz_coef_row0, sz_coef_row1, sz_coef_col0, sz_coef_col1 = \
                    sph_dirac_recon_alg_joint(G_ri_lst, a_ri, K, L, L, noise_level,
                                              max_ini, stop_cri, max_iter, use_lu=use_lu,
                                              # beta=b_opt_ri_lst,
                                              symb=symb,
                                              G_amp_ref_ri_lst=G_amp_ref_ri_lst)

                try:
                    azimuthk_recon, colatitudek_recon = \
                        sph_extract_innovation(
                            a_ri, K,
                            sph_reshape_coef(c_row_opt, sz_coef_row0, sz_coef_row1),
                            sph_reshape_coef(c_col_opt, sz_coef_col0, sz_coef_col1),
                            p_mic_x_rotated, p_mic_y_rotated, p_mic_z_rotated,
                            G_amp_ref_ri_lst=G_amp_ref_ri_lst,
                            colatitude_ref_rotated=colatitudek_ref_rotated,
                            azimuth_ref_rotated=azimuthk_ref_rotated
                        )
                except RuntimeError:
                    continue
            else:
                c_row_opt, c_col_opt, min_error, b_opt_ri_lst, ini, \
                sz_coef_row0, sz_coef_row1, sz_coef_col0, sz_coef_col1 = \
                    sph_dirac_recon_alg_joint(G_ri_lst, a_ri, K, L, L, noise_level,
                                              max_ini, stop_cri, max_iter, use_lu=use_lu,
                                              # beta=b_opt_ri_lst,
                                              symb=symb)

                try:
                    azimuthk_recon, colatitudek_recon = \
                        sph_extract_innovation(
                            a_ri, K,
                            sph_reshape_coef(c_row_opt, sz_coef_row0, sz_coef_row1),
                            sph_reshape_coef(c_col_opt, sz_coef_col0, sz_coef_col1),
                            p_mic_x_rotated, p_mic_y_rotated, p_mic_z_rotated
                        )
                except RuntimeError:
                    continue

            xk_recon, yk_recon, zk_recon = sph2cart(1, colatitudek_recon, azimuthk_recon)
            xyz_rotate_back = linalg.solve(rotate_mtx,
                                           np.vstack((xk_recon.flatten('F'),
                                                      yk_recon.flatten('F'),
                                                      zk_recon.flatten('F')))
                                           )
            colatitudek_recon = np.arccos(xyz_rotate_back[2, :])
            azimuthk_recon = np.mod(np.arctan2(xyz_rotate_back[1, :],
                                               xyz_rotate_back[0, :]),
                                    2 * np.pi)

            # use the correctly identified colatitude and azimuth to reconstruct the correct amplitudes
            # implementation with list
            alphak_recon = []
            error_loop = 0
            for band_count in range(num_bands):
                a_ri_band = a_ri[:, band_count]

                amp_mtx_ri_sorted_band = \
                    sph_build_mtx_amp_ri(
                        p_mic_x_normalised[:, band_count],
                        p_mic_y_normalised[:, band_count],
                        p_mic_z_normalised[:, band_count],
                        azimuthk_recon, colatitudek_recon
                    )

                alphak_recon_band = sp.optimize.nnls(
                    np.dot(amp_mtx_ri_sorted_band.T, amp_mtx_ri_sorted_band),
                    np.dot(amp_mtx_ri_sorted_band.T, a_ri_band)
                )[0]

                alphak_recon.append(alphak_recon_band)
                error_loop += linalg.norm(a_ri_band -
                                          np.dot(amp_mtx_ri_sorted_band,
                                                 alphak_recon_band)
                                          )

            if verbose:
                print('objective function value: {0:.3e}'.format(error_loop))

            if error_loop < min_error_all:
                min_error_all = error_loop
                colatitudek_opt = colatitudek_recon
                azimuthk_opt = azimuthk_recon
                alphak_opt = np.reshape(np.concatenate(alphak_recon),
                                        (-1, num_bands), order='F')

            xyz_opt_rotated = np.dot(rotate_mtx,
                                     np.row_stack(sph2cart(1, colatitudek_opt, azimuthk_opt))
                                     )
            colatitude_opt_rotated = np.arccos(xyz_opt_rotated[2, :])
            azimuth_opt_rotated = np.arctan2(xyz_opt_rotated[1, :], xyz_opt_rotated[0, :])

            b_opt_ri_lst = [sph_build_beta(alphak_opt[:, band_count],
                                           colatitude_opt_rotated,
                                           azimuth_opt_rotated, L, symb=symb)[1]
                            for band_count in range(num_bands)]

            if update_G:
                G_ri_lst = sph_update_G_ri(
                    colatitude_opt_rotated[K_ref:],
                    azimuth_opt_rotated[K_ref:],
                    L, p_mic_x_rotated,
                    p_mic_y_rotated,
                    p_mic_z_rotated,
                    num_bands, G_ri_lst, symb=symb)

    # convert to DOA
    azimuthk_doa = np.mod(azimuthk_opt + np.pi, 2 * np.pi)
    # TODO: confirm with Robin about this conversion!
    # TODO: update planar_select_reliable_recon accordingly
    # should be in [0, pi]
    colatitudek_doa = np.mod(np.pi - colatitudek_opt, 2 * np.pi)
    return colatitudek_doa, azimuthk_doa, alphak_opt
    # return colatitudek_opt, azimuthk_opt, np.reshape(alphak_opt, (-1, num_bands), order='F')


def sph_determin_max_coef_sz(L):
    """
    determine the maximum size of the annihilating filter coefficients, which are assumed
    to be an isosceles right triangle.
    :param L: the spherical harmonics is from -L <= m <= L
    :return: 
    """
    return int((4 * L + 9 - np.sqrt((4 * L + 9) ** 2 - 8 * (L + 2) ** 2)) // 2)


def sph_extract_innovation(a_ri, K, c_row, c_col,
                           p_mic_x, p_mic_y, p_mic_z,
                           **kwargs):
    """
    retrieve Dirac parameters, i.e., Dirac locations and amplitudes from the 
    reconstructed annihilating filter coefficients.
    :return: 
    """
    if 'G_amp_ref_ri_lst' in kwargs:
        G_amp_ref_ri_lst = kwargs['G_amp_ref_ri_lst']
    else:
        G_amp_ref_ri_lst = None

    if 'colatitude_ref_rotated' in kwargs and 'azimuth_ref_rotated' in kwargs:
        colatitude_ref = kwargs['colatitude_ref_rotated']
        azimuth_ref = kwargs['azimuth_ref_rotated']
    else:
        colatitude_ref = None
        azimuth_ref = None

    if G_amp_ref_ri_lst is None or azimuth_ref is None or colatitude_ref is None:
        ref_sol_available = False
        K_ref = 0
    else:
        ref_sol_available = True
        K_ref = G_amp_ref_ri_lst[0].shape[1]

    num_bands = a_ri.shape[1]
    # root finding
    z1, z2 = find_roots(c_row, c_col)
    z1 = z1.flatten('F')
    z2 = z2.flatten('F')

    if z1.size == 0 or z2.size == 0:
        raise RuntimeError('No roots found. Continue.')

    azimuth_ks = np.mod(-np.angle(z1), 2 * np.pi)
    colatitude_ks = np.mod(np.real(np.arccos(z2)), 2 * np.pi)

    # find the amplitudes of the Dirac deltas
    colatitude_k_grid = colatitude_ks.flatten('F')
    azimuth_k_grid = azimuth_ks.flatten('F')

    if colatitude_ks.shape[0] == 0:
        raise RuntimeError('No source found. Continue.')

    # implementation with list comprehension
    '''strategy I'''
    alphak_recon_lst = []
    if ref_sol_available:
        for band_count in range(num_bands):
            amp_mtx_loop = np.column_stack((
                G_amp_ref_ri_lst[band_count],
                sph_build_mtx_amp_ri(p_mic_x[:, band_count],
                                     p_mic_y[:, band_count],
                                     p_mic_z[:, band_count],
                                     azimuth_k_grid, colatitude_k_grid)
            ))

            alphak_recon_lst.append(
                sp.optimize.nnls(
                    np.dot(amp_mtx_loop.T, amp_mtx_loop),
                    np.dot(amp_mtx_loop.T, a_ri[:, band_count])
                )[0]
            )

    else:
        for band_count in range(num_bands):
            amp_mtx_loop = \
                sph_build_mtx_amp_ri(p_mic_x[:, band_count],
                                     p_mic_y[:, band_count],
                                     p_mic_z[:, band_count],
                                     azimuth_k_grid, colatitude_k_grid)
            alphak_recon_lst.append(
                sp.optimize.nnls(
                    np.dot(amp_mtx_loop.T, amp_mtx_loop),
                    np.dot(amp_mtx_loop.T, a_ri[:, band_count])
                )[0]
            )

    alphak_recon = np.reshape(np.concatenate(alphak_recon_lst),
                              (-1, num_bands), order='F')

    # extract the locations that have the largest amplitudes
    alphak_sort_idx = np.argsort(np.abs(alphak_recon[K_ref:, :]), axis=0)[:-K - 1:-1]

    # now use majority vote across all sub-bands
    idx_all = np.zeros((alphak_recon.shape[0] - K_ref,
                        alphak_recon.shape[1]),
                       dtype=int)
    for loop in range(num_bands):
        idx_all[alphak_sort_idx[:, loop], loop] = 1
    idx_sel = np.argsort(np.sum(idx_all, axis=1))[:-K - 1:-1]

    '''strategy II'''
    # num_removal = colatitude_k_grid.size - K
    # mask_all = (np.column_stack((
    #     np.ones((colatitude_k_grid.size, colatitude_k_grid.size), dtype=int) -
    #     np.eye(colatitude_k_grid.size, dtype=int),
    #     np.ones((colatitude_k_grid.size, K_ref), dtype=int)
    # ))).astype(bool)
    # if ref_sol_available:
    #     amp_mtx_full = [
    #         np.column_stack((
    #             G_amp_ref_ri_lst[band_count],
    #             sph_build_mtx_amp_ri(
    #                 p_mic_x[:, band_count],
    #                 p_mic_y[:, band_count],
    #                 p_mic_z[:, band_count],
    #                 azimuth_k=azimuth_k_grid,
    #                 colatitude_k=colatitude_k_grid
    #             )
    #         ))
    #         for band_count in range(num_bands)]
    # else:
    #     amp_mtx_full = [
    #         sph_build_mtx_amp_ri(
    #             p_mic_x[:, band_count],
    #             p_mic_y[:, band_count],
    #             p_mic_z[:, band_count],
    #             azimuth_k=azimuth_k_grid,
    #             colatitude_k=colatitude_k_grid
    #         )
    #         for band_count in range(num_bands)]
    #
    # leave_one_out_error = [
    #     sph_compute_fitting_error_amp(
    #         a_ri,
    #         amp_mtx_ri_lst=[
    #             amp_mtx_full[band_count][:, mask_all[removal_ind, :]]
    #             for band_count in range(num_bands)],
    #         num_bands=num_bands)[0]
    #     for removal_ind in range(mask_all.shape[0])]
    #
    # idx_sel = np.argsort(np.asarray(leave_one_out_error))[num_removal:]
    # ========================================================

    if ref_sol_available:
        colatitudek_recon = np.concatenate((colatitude_ref, colatitude_k_grid[idx_sel]))
        azimuthk_recon = np.concatenate((azimuth_ref, azimuth_k_grid[idx_sel]))
    else:
        colatitudek_recon = colatitude_k_grid[idx_sel]
        azimuthk_recon = azimuth_k_grid[idx_sel]

    return azimuthk_recon, colatitudek_recon


def sph_build_beta(alpha, theta, phi, L, symb=False):
    alpha = np.reshape(alpha, (-1, 1), order='F')
    theta = np.reshape(theta, (1, -1), order='F')
    phi = np.reshape(phi, (1, -1), order='F')
    beta_pos = np.zeros(int((L + 1) * (L + 2) * 0.5), dtype=complex)
    beta_neg = np.zeros(int(L * (L + 1) * 0.5), dtype=complex)
    # the positive m part
    count = 0
    for m in range(L + 1):
        for n in range(L + 1 - m):
            beta_pos[count] = np.dot(np.cos(theta) ** n *
                                     np.sin(theta) ** m *
                                     np.exp(-1j * m * phi),
                                     alpha).squeeze()
            count += 1

    # the negative m part
    count = 0
    for m in range(1, L + 1):
        for n in range(L + 1 - m):
            beta_neg[count] = np.dot(np.cos(theta) ** n *
                                     np.sin(theta) ** m *
                                     np.exp(-1j * m * phi),
                                     np.conj(alpha)).squeeze()
            count += 1

    if symb:
        return beta_pos, np.concatenate((beta_pos.real, beta_pos.imag))
    else:
        beta_cpx = np.concatenate((beta_pos, beta_neg))
        beta_ri = np.concatenate((beta_cpx.real, beta_cpx.imag))
        return beta_cpx, beta_ri


def sph_update_G_ri(colatitudek, azimuthk, L, p_mic_x, p_mic_y, p_mic_z, num_bands,
                    mtx_fri2visibility_lst, symb=False):
    """
    update the linear transformation matrix that links the uniformly sampled sinusoids
    to the visibility measurements (i.e., the cross-correlation of microphone signals)
    :param colatitudek: estimated co-latitudes
    :param azimuthk: estimated azimuths
    :param L: maximum degree of the spherical harmonics
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param p_mic_z: a vector that contains microphones' z-coordinates
    :param mtx_fri2visibility_lst: a list that contains the linear transformation
                matrix that links the FRI sequence with the measured visibility
    :return:
    """
    K = colatitudek.size
    # mtx_freq2visibility_ri = cpx_mtx2real(mtx_freq2visibility)
    mtx_fri2freq = sph_mtx_fri2freq_row_major(L, symb=symb)
    # ---------------------------------------------------------------
    # the spherical harmonics are related to the amplitudes linearly
    # reshape colatitudek, azimuthk in order to use broadcasting
    colatitudek = np.reshape(colatitudek, (1, -1), order='F')
    azimuthk = np.reshape(azimuthk, (1, -1), order='F')

    if symb:
        m_grid, l_grid = np.meshgrid(np.arange(0, L + 1, step=1, dtype=int),
                                     np.arange(0, L + 1, step=1, dtype=int))
        m_grid = vec_from_low_tri_col_by_col_pos_ms(m_grid)[:, np.newaxis]
        l_grid = vec_from_low_tri_col_by_col_pos_ms(l_grid)[:, np.newaxis]
    else:
        m_grid, l_grid = np.meshgrid(np.arange(-L, L + 1, step=1, dtype=int),
                                     np.arange(0, L + 1, step=1, dtype=int))
        m_grid = vec_from_low_tri_col_by_col(m_grid)[:, np.newaxis]
        l_grid = vec_from_low_tri_col_by_col(l_grid)[:, np.newaxis]

    # mtx_Ylm_cpx = np.conj(sph_harm_ufnc(l_grid, m_grid, colatitudek, azimuthk))
    # mtx_Ylm = np.vstack((mtx_Ylm_cpx.real, mtx_Ylm_cpx.imag))
    mtx_Ylm = cpx_mtx2real(np.conj(sph_harm_ufnc(l_grid, m_grid, colatitudek, azimuthk)))
    # mtx_amp2fri = linalg.solve(mtx_fri2freq, mtx_Ylm)
    if symb:
        mtx_amp2fri_half = \
            linalg.solve(np.row_stack((
                mtx_fri2freq[np.int(0.5 * L * (L + 1)):(L + 1) ** 2, :],
                mtx_fri2freq[(L + 1) ** 2 + np.int(0.5 * L * (L + 1)):, :]
            )),
                mtx_Ylm)[:, :K]
    else:
        mtx_amp2fri_half = linalg.solve(mtx_fri2freq, mtx_Ylm)[:, :K]
    # the pseudo-inverse gives the mapping from amplitude to fri sequence G1
    mtx_fri2amp_half = linalg.lstsq(mtx_amp2fri_half, np.eye(mtx_amp2fri_half.shape[0]))[0]
    # ---------------------------------------------------------------
    # the mapping from Dirac amplitudes to visibilities G2
    mtx_amp2visibility_half_lst = [
        sph_build_mtx_amp_ri(
            p_mic_x[:, band_count], p_mic_y[:, band_count],
            p_mic_z[:, band_count], azimuthk, colatitudek)
        for band_count in range(num_bands)]
    # ---------------------------------------------------------------
    # project the mapping from FRI sequence to visibilities to the null space of mtx_fri2amp
    mtx_null_proj_half = np.eye(mtx_fri2amp_half.shape[1]) - \
                         np.dot(mtx_fri2amp_half.T,
                                linalg.lstsq(mtx_fri2amp_half.T,
                                             np.eye(mtx_fri2amp_half.shape[1]))[0]
                                )

    return [np.dot(mtx_amp2visibility_half_lst[band_count],
                   mtx_fri2amp_half) +
            np.dot(mtx_fri2visibility_lst[band_count], mtx_null_proj_half)
            for band_count in range(num_bands)]


def sph_build_mtx_amp_ri(p_mic_x, p_mic_y, p_mic_z, azimuth_k, colatitude_k):
    mtx = sph_build_mtx_amp_cpx(p_mic_x, p_mic_y, p_mic_z, azimuth_k, colatitude_k)
    # because we know that the amplitude sigma_k^2 is real-valued
    # so the number of columns is halved compared with the full-complex case
    return np.vstack((mtx.real, mtx.imag))


def sph_build_mtx_amp_cpx(p_mic_x, p_mic_y, p_mic_z, azimuth_k, colatitude_k):
    """
    the matrix that maps Diracs' amplitudes to the visibilities
    :param p_mic_x: a vector that contains microphones' x-coordinates (normalised by
                the mid-band frequency and the speed of sound)
    :param p_mic_y: a vector that contains microphones' y-coordinates (normalised by
                the mid-band frequency and the speed of sound)
    :param p_mic_z: a vector that contains microphones' z-coordinates (normalised by
                the mid-band frequency and the speed of sound)
    :param azimuth_k: Diracs' azimuth
    :param colatitude_k: Diracs' co-latitudes
    :return:
    """
    xk, yk, zk = sph2cart(1, colatitude_k, azimuth_k)
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    zk = np.reshape(zk, (1, -1), order='F')

    num_mic = p_mic_x.shape[0]
    num_mic_1 = num_mic - 1
    K = azimuth_k.size

    mtx = np.zeros((num_mic * num_mic_1, K), dtype=complex, order='C')

    mtx_bg_idx0 = 0
    for q in range(num_mic):
        p_x_qqp = np.reshape(p_mic_x[q] -
                             np.concatenate((p_mic_x[:q], p_mic_x[q + 1:])),
                             (num_mic_1, 1), order='F')
        p_y_qqp = np.reshape(p_mic_y[q] -
                             np.concatenate((p_mic_y[:q], p_mic_y[q + 1:])),
                             (num_mic_1, 1), order='F')
        p_z_qqp = np.reshape(p_mic_z[q] -
                             np.concatenate((p_mic_z[:q], p_mic_z[q + 1:])),
                             (num_mic_1, 1), order='F')
        mtx[mtx_bg_idx0:mtx_bg_idx0 + num_mic_1, :] = \
            np.exp(-1j * (xk * p_x_qqp + yk * p_y_qqp + zk * p_z_qqp))
        mtx_bg_idx0 += num_mic_1

    return mtx


def sph_dirac_recon_alg_joint(G_ri_lst0, a_ri, K, L, M, noise_level, max_ini,
                              stop_cri, max_iter, use_lu=True, symb=False,
                              use_GtGinv=False, **kwargs):
    """
    ALGORITHM for joint estimation of Diracs on the sphere by enforcing the
    annihilation along the azimuth and the latitudes jointly.
    :param G_lst: a list that constains the linear transformation matrix that
                links the given visibility measurements to the uniform sinusoidal
                samples for each sub-band.
    :param a_ri: the given visibility measurements
    :param K: number of Diracs
    :param L: maximum degree of the spherical harmonics
    :param M: maximum order of the spherical harmonics (M <= L)
    :param noise_level: noise level present in the visibility measurements
    :param max_ini: maximum number of initialisations allowed for the algorithm
    :param stop_cri: either 'mse' or 'max_iter'
    :param max_iter: maximum number of iterations for when 'max_iter' creteria is used
    :param verbose: output intermediate results or not (for debugging)
    :param kwargs: optional input argument, include 'beta'. If it is not given,
                then we use the least square solution for it.
    :return:
    """
    compute_mse = (stop_cri == 'mse')
    num_bands = a_ri.shape[1]  # number of sub-bands
    # determine the size of 2D annihilating filter size (as an isosceles right triangle)
    sz_coef_row0 = sz_coef_row1 = np.int(np.ceil(0.5 * (-1 + np.sqrt(1 + 8 * (K + 1)))))

    if sph_compute_size(sz_coef_row0, sz_coef_row1) == K + 1:
        sz_coef_col0 = sz_coef_col1 = sz_coef_row0 + 1
    else:
        sz_coef_col0 = sz_coef_col1 = sz_coef_row0

    if symb:
        expansion_mtx = linalg.block_diag(*([sph_build_exp_mtx_symb(L)] * 2))
    else:
        expansion_mtx = None

    # number of coefficients for the annihilating filter
    num_coef_row = sph_compute_size(sz_coef_row0, sz_coef_row1)
    num_coef_col = sph_compute_size(sz_coef_col0, sz_coef_col1)

    # size of various matrices / vectors
    sz_coef = 2 * (num_coef_row + num_coef_col)
    sz_S0 = 2 * (num_coef_row + num_coef_col - 2 * (K + 1))

    sz_b_pos = np.int(0.5 * (L + 1) * (L + 2))

    if symb:
        sz_G1 = (L + 1) * (L + 2)
    else:
        sz_G1 = 2 * (L + 1) ** 2

    if 'G_amp_ref_ri_lst' in kwargs:
        G_amp_ref_ri_lst = kwargs['G_amp_ref_ri_lst']
        K_ref = G_amp_ref_ri_lst[0].shape[1]
        sz_G1 += K_ref
        mtx_extract_b = np.eye(sz_G1)[K_ref:, :]
        G_ri_lst = [
            np.column_stack((G_amp_ref_ri_lst[band_count], G_ri_lst0[band_count]))
            for band_count in range(num_bands)
        ]
        ref_sol_available = True
    else:
        mtx_extract_b = None
        G_ri_lst = G_ri_lst0
        ref_sol_available = False

    # both G and a are real-valued
    for G_subband in G_ri_lst:
        assert not np.iscomplexobj(G_subband)

    assert not np.iscomplexobj(a_ri)

    # precompute a few things
    Gt_a_lst = []
    lu_GtG_lst = []
    Tbeta_ri0_lst = []

    if 'beta' in kwargs:
        beta_ri_lst = kwargs['beta']
        compute_beta = False
    else:
        beta_ri_lst = []
        compute_beta = True

    for loop in range(num_bands):
        G_loop = G_ri_lst[loop]
        a_loop = a_ri[:, loop]

        Gt_a_loop = np.dot(G_loop.T, a_loop)
        Gt_a_lst.append(Gt_a_loop)

        if use_GtGinv:
            GtGinv_loop = linalg.solve(np.dot(G_loop.T, G_loop),
                                       np.eye(sz_G1), check_finite=False)
            if compute_beta:
                beta_ri_loop = np.dot(GtGinv_loop, Gt_a_loop)
                beta_ri_lst.append(beta_ri_loop)
            else:
                beta_ri_loop = beta_ri_lst[loop]

            # regardless of use_lu, we store GtGinv here
            lu_GtG_lst.append(GtGinv_loop)
        else:
            GtG_loop = np.dot(G_loop.T, G_loop)

            if compute_beta:
                # Here lstsq was sometimes not converging
                # so we switched to 'gelsy' solver that doesn't use SVD
                # beta_ri_loop = linalg.lstsq(G_loop, a_loop, lapack_driver='gelsy')[0]
                beta_ri_loop = linalg.solve(GtG_loop, Gt_a_loop)
                beta_ri_lst.append(beta_ri_loop)
            else:
                beta_ri_loop = beta_ri_lst[loop]

            if use_lu:
                lu_GtG_lst.append(linalg.lu_factor(GtG_loop, check_finite=False))
            else:
                lu_GtG_lst.append(GtG_loop)

        if symb:
            if ref_sol_available:
                beta_ri_loop_full = np.dot(expansion_mtx,
                                           np.dot(mtx_extract_b, beta_ri_loop))
            else:
                beta_ri_loop_full = np.dot(expansion_mtx, beta_ri_loop)

            beta_cpx_loop = beta_ri_loop_full[:(L + 1) ** 2] + \
                            1j * beta_ri_loop_full[(L + 1) ** 2:]
        else:
            if ref_sol_available:
                beta_ri_loop = np.dot(mtx_extract_b, beta_ri_loop)

            beta_cpx_loop = beta_ri_loop[:(L + 1) ** 2] + \
                            1j * beta_ri_loop[(L + 1) ** 2:]

        beta_pos_cpx_loop = beta_cpx_loop[:sz_b_pos]
        beta_neg_cpx_loop = beta_cpx_loop[sz_b_pos:]
        Tbeta_ri0_lst.append(
            sph_T_mtx_joint_ri(beta_pos_cpx_loop, beta_neg_cpx_loop,
                               sz_coef_row0, sz_coef_row1,
                               sz_coef_col0, sz_coef_col1))

    # initialize error to something very large
    min_error = float('inf')

    # the effective number of equations in the annihilation constraints
    c_ri = np.random.randn(sz_coef)
    c_rwo_col_cpx = c_ri[:num_coef_row + num_coef_col] + \
                    1j * c_ri[num_coef_row + num_coef_col:]
    c_row = c_rwo_col_cpx[:num_coef_row]
    c_col = c_rwo_col_cpx[num_coef_row:]

    R_test = sph_R_mtx_joint_ri(
        c_row, c_col, L, M, expansion_mtx=expansion_mtx,
        mtx_extract_b=mtx_extract_b,
        sz_coef_row0=sz_coef_row0, sz_coef_row1=sz_coef_row1,
        sz_coef_col0=sz_coef_col0, sz_coef_col1=sz_coef_col1)
    s_test = linalg.svd(R_test, compute_uv=False, full_matrices=True)

    sz_Rc0_effective = min(R_test.shape[0], R_test.shape[1]) - \
                       np.where(np.abs(s_test) < 1e-10)[0].size

    rhs = np.concatenate((np.zeros(sz_coef + sz_S0, dtype=float), np.array([1, 1, 0, 0])))

    for ini in range(max_ini):
        # select a subset of size (K + 1) of these coefficients
        # here the indices corresponds to the part of coefficients that are ZERO
        S_ri = sph_sel_coef_subset_ri(num_coef_row, num_coef_col, K)
        S_ri_T = S_ri.T

        # randomly initialise the annihilating fitler coefficients
        c_ri = np.random.randn(sz_coef)
        c0_pos_r = c_ri[:num_coef_row][:, np.newaxis]
        c0_neg_r = c_ri[num_coef_row:num_coef_row + num_coef_col][:, np.newaxis]
        c0_pos_i = c_ri[num_coef_row + num_coef_col:2 * num_coef_row + num_coef_col][:, np.newaxis]
        c0_neg_i = c_ri[2 * num_coef_row + num_coef_col:][:, np.newaxis]
        c0_r_T = linalg.block_diag(c0_pos_r.T, c0_neg_r.T)
        c0_i_T = linalg.block_diag(c0_pos_i.T, c0_neg_i.T)
        c0_T = np.vstack((np.hstack((c0_r_T, c0_i_T)),
                          np.hstack((-c0_i_T, c0_r_T))))
        c0 = c0_T.T

        c_row_col_cpx = c_ri[:num_coef_row + num_coef_col] + \
                        1j * c_ri[num_coef_row + num_coef_col:]
        c_row = c_row_col_cpx[:num_coef_row]
        c_col = c_row_col_cpx[num_coef_row:]

        Rmtx_band_ri = sph_R_mtx_joint_ri(c_row, c_col, L, M,
                                          expansion_mtx=expansion_mtx,
                                          mtx_extract_b=mtx_extract_b,
                                          sz_coef_row0=sz_coef_row0,
                                          sz_coef_row1=sz_coef_row1,
                                          sz_coef_col0=sz_coef_col0,
                                          sz_coef_col1=sz_coef_col1)

        Q_H = linalg.qr(Rmtx_band_ri, mode='economic',
                        check_finite=False)[0][:, :sz_Rc0_effective].T

        Rmtx_band_ri = np.dot(Q_H, Rmtx_band_ri)

        # last row in mtx_loop
        mtx_loop_last_row = np.hstack((np.vstack((S_ri, c0_T)),
                                       np.zeros((sz_S0 + 4, sz_S0 + 4), dtype=float)
                                       ))

        for inner in range(max_iter):
            if inner == 0:
                mtx_loop = \
                    np.vstack((np.hstack((sph_compute_mtx_obj_multiband(lu_GtG_lst, Tbeta_ri0_lst,
                                                                        Rmtx_band_ri, Q_H,
                                                                        num_bands, sz_coef,
                                                                        use_lu=use_lu,
                                                                        use_GtGinv=use_GtGinv),
                                          S_ri_T, c0)),
                               mtx_loop_last_row
                               ))
            else:
                mtx_loop[:sz_coef, :sz_coef] = \
                    sph_compute_mtx_obj_multiband(lu_GtG_lst, Tbeta_ri0_lst,
                                                  Rmtx_band_ri, Q_H, num_bands,
                                                  sz_coef, lu_lst, use_lu=use_lu,
                                                  use_GtGinv=use_GtGinv)

            try:
                c_ri = linalg.solve(mtx_loop, rhs, overwrite_a=False,
                                    check_finite=False)[:sz_coef]
            except linalg.LinAlgError:
                break

            c_row_col_cpx = c_ri[:num_coef_row + num_coef_col] + \
                            1j * c_ri[num_coef_row + num_coef_col:]
            c_row = c_row_col_cpx[:num_coef_row]
            c_col = c_row_col_cpx[num_coef_row:]

            # build new R matrix
            Rmtx_band_ri = sph_R_mtx_joint_ri(c_row, c_col, L, M,
                                              expansion_mtx=expansion_mtx,
                                              mtx_extract_b=mtx_extract_b,
                                              sz_coef_row0=sz_coef_row0,
                                              sz_coef_row1=sz_coef_row1,
                                              sz_coef_col0=sz_coef_col0,
                                              sz_coef_col1=sz_coef_col1)

            Q_H = linalg.qr(Rmtx_band_ri, mode='economic',
                            check_finite=False)[0][:, :sz_Rc0_effective].T

            Rmtx_band_ri = np.dot(Q_H, Rmtx_band_ri)

            # reconstruct b-s for all the sub-bands
            error_loop, lu_lst, b_recon_ri_lst = \
                sph_compute_b(G_ri_lst, lu_GtG_lst, beta_ri_lst,
                              Rmtx_band_ri, num_bands, a_ri,
                              use_lu=use_lu, use_GtGinv=use_GtGinv)

            if error_loop < min_error:
                min_error = error_loop
                b_opt_ri_lst = b_recon_ri_lst
                c_row_opt = c_row
                c_col_opt = c_col

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

    return c_row_opt, c_col_opt, min_error, b_opt_ri_lst, ini, \
           sz_coef_row0, sz_coef_row1, sz_coef_col0, sz_coef_col1


def planar_select_reliable_recon(a_cpx, p_mic_x, p_mic_y, p_mic_z, omega_bands,
                                 colatitudek_doa, azimuthk_doa, sound_speed,
                                 num_removal):
    """
    select reliable reconstruction the reconstructed Diracs
    :param a_cpx: complex-valued visibility measurements
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param p_mic_z: a vector that contains microphones' z-coordinates
    :param omega_bands: mid-band (ANGULAR) frequencies [radian/sec]
    :param colatitudek_doa: colatitude of the source (DOA) -> a conversion is needed
    :param azimuthk_doa: azimuth of the source (DOA) -> a conversion is needed
    :param sound_speed: speed of sound
    :param num_removal: number of Diracs to be removed
    :return: 
    """
    a_ri = np.row_stack((a_cpx.real, a_cpx.imag))
    # convert DOA to the source locations
    azimuthk_recon = np.mod(azimuthk_doa - np.pi, 2 * np.pi)
    colatitudek_recon = np.pi - colatitudek_doa
    num_bands = np.array(omega_bands).size  # number of subbands
    K = colatitudek_doa.size  # total number of Diracs
    norm_factor = np.reshape(sound_speed / omega_bands, (1, -1), order='F')
    # normalised antenna coordinates in a matrix form
    # each column corresponds to the location in one subband,
    # i.e., the second dimension corresponds to different subbands
    p_mic_x_normalised = np.reshape(p_mic_x, (-1, 1), order='F') / norm_factor
    p_mic_y_normalised = np.reshape(p_mic_y, (-1, 1), order='F') / norm_factor
    p_mic_z_normalised = np.reshape(p_mic_z, (-1, 1), order='F') / norm_factor

    mask_all = (np.ones((K, K), dtype=int) - np.eye(K, dtype=int)).astype(bool)
    # compute the amp_mtx with all Dirac locations
    amp_mtx_full = [
        sph_build_mtx_amp_ri(
            p_mic_x_normalised[:, band_count],
            p_mic_y_normalised[:, band_count],
            p_mic_z_normalised[:, band_count],
            azimuthk_recon, colatitudek_recon
        )
        for band_count in range(num_bands)]

    leave_one_out_error = []
    for removal_ind in range(K):
        amp_mtx_loop = [amp_mtx_full[band_count][:, mask_all[removal_ind, :]]
                        for band_count in range(num_bands)]

        leave_one_out_error.append(
            sph_compute_fitting_error_amp(a_ri, amp_mtx_loop, num_bands)[0]
        )

    idx_opt = np.argsort(np.asarray(leave_one_out_error))[num_removal:]
    azimuthk_reliable = azimuthk_recon[idx_opt]
    colatitudek_reliable = colatitudek_recon[idx_opt]

    amp_mtx_reliable = [
        sph_build_mtx_amp_ri(
            p_mic_x_normalised[:, band_count],
            p_mic_y_normalised[:, band_count],
            p_mic_z_normalised[:, band_count],
            azimuthk_reliable, colatitudek_reliable
        )
        for band_count in range(num_bands)]
    amplitudek_reliable = \
        sph_compute_fitting_error_amp(a_ri, amp_mtx_reliable, num_bands)[1]

    # convert to DOA
    azimuthk_reliable_doa = np.mod(azimuthk_reliable + np.pi, 2 * np.pi)
    colatitudek_reliable_doa = np.pi - colatitudek_reliable
    return colatitudek_reliable_doa, azimuthk_reliable_doa, amplitudek_reliable


def sph_compute_fitting_error_amp(a_ri, amp_mtx_ri_lst, num_bands):
    amplitude_recon = []
    error_all = 0
    for band_count in range(num_bands):
        amp_mtx_loop = amp_mtx_ri_lst[band_count]
        amplitude_band = sp.optimize.nnls(
            np.dot(amp_mtx_loop.T, amp_mtx_loop),
            np.dot(amp_mtx_loop.T, a_ri[:, band_count])
        )[0]

        amplitude_recon.append(amplitude_band)
        error_all += linalg.norm(a_ri[:, band_count] -
                                 np.dot(amp_mtx_loop, amplitude_band))

    return error_all, \
           np.reshape(np.asarray(amplitude_recon),
                      (-1, num_bands), order='F')


def sph_eval_fitting_error(G_lst, lu_GtG_lst, beta_ri_lst, Rmtx_band_ri, Q_H,
                           Tbeta_lst, num_bands, a_ri, c_ri, eval_b=False, use_lu=True):
    """
    Evaluate the fitting error at each iteration. Depending on cases, we may not always
    need to compute b explicitly.
    :param G_lst: list of G for different subbands
    :param lu_GtG_lst: list of G^H G for different subbands
    :param beta_ri_lst: list of beta-s for different subbands
    :param Rmtx_band_ri: right dual matrix for the annihilating filter
            (same for each block -> not a list)
    :param Q_H: a rectangular matrix that extracts the effective lines of equations
    :param Tbeta_lst: list of Teoplitz matrices for beta-s
    :param num_bands: number of bands
    :param a_ri: a 2D numpy array. each column corresponds to the measurements
            within a subband
    :param c_ri: reconstructed annihilating filters (for row and columns)
    :param eval_b: whether b is evaluated or not
    :return: fitting error, R_GtGinv_Rt_lst, and (optional) b for each sub-band
    """
    if eval_b:
        return sph_compute_b(G_lst, lu_GtG_lst, beta_ri_lst, Rmtx_band_ri, num_bands, a_ri, use_lu)
    else:
        lu_lst = []
        mtx_obj = np.zeros((c_ri.size, c_ri.size), dtype=float)
        if use_lu:
            for loop in range(num_bands):
                Tbeta_loop = np.dot(Q_H, Tbeta_lst[loop])

                lu_loop = \
                    linalg.lu_factor(np.dot(Rmtx_band_ri,
                                            linalg.lu_solve(lu_GtG_lst[loop],
                                                            Rmtx_band_ri.T,
                                                            check_finite=False)),
                                     overwrite_a=True, check_finite=False)

                mtx_obj += np.dot(Tbeta_loop.T,
                                  linalg.lu_solve(lu_loop,
                                                  Tbeta_loop,
                                                  check_finite=False)
                                  )
                lu_lst.append(lu_loop)
        else:
            for loop in range(num_bands):
                Tbeta_loop = np.dot(Q_H, Tbeta_lst[loop])

                lu_loop = \
                    np.dot(Rmtx_band_ri,
                           linalg.solve(lu_GtG_lst[loop],
                                        Rmtx_band_ri.T,
                                        check_finite=False))
                mtx_obj += np.dot(Tbeta_loop.T,
                                  linalg.solve(lu_loop,
                                               Tbeta_loop,
                                               check_finite=False)
                                  )
                lu_lst.append(lu_loop)

        return np.dot(c_ri.T, np.dot(mtx_obj, c_ri)), lu_lst


def sph_compute_b(G_lst, lu_GtG_lst, beta_ri_lst, Rmtx_band_ri,
                  num_bands, a_ri, use_lu=True, use_GtGinv=False):
    """
    compute the uniform sinusoidal samples b from the updated annihilating
    filter coeffiients.
    :param G_lst: list of G for different subbands
    :param lu_GtG_lst: list of G^H G for different subbands
    :param beta_ri_lst: list of beta-s for different subbands
    :param Rmtx_band_ri: right dual matrix for the annihilating filter
            (same for each block -> not a list)
    :param num_bands: number of bands
    :param a_ri: a 2D numpy array. each column corresponds to the measurements
            within a subband
    :return:
    """
    if use_GtGinv:  # lu_GtG_lst actually stores GtGinv in this case
        if use_lu:
            lu_lst = [
                linalg.lu_factor(np.dot(Rmtx_band_ri,
                                        np.dot(lu_GtG_lst[loop],
                                               Rmtx_band_ri.T)),
                                 overwrite_a=True, check_finite=False)
                for loop in range(num_bands)
            ]
            b_lst = [
                beta_ri_lst[loop] - \
                np.dot(lu_GtG_lst[loop],
                       np.dot(Rmtx_band_ri.T,
                              linalg.lu_solve(lu_lst[loop],
                                              np.dot(Rmtx_band_ri, beta_ri_lst[loop]),
                                              check_finite=False)
                              )
                       )
                for loop in range(num_bands)
            ]
            a_Gb_lst = [
                a_ri[:, loop] - np.dot(G_lst[loop], b_lst[loop])
                for loop in range(num_bands)
            ]
        else:
            lu_lst = [
                np.dot(Rmtx_band_ri,
                       np.dot(lu_GtG_lst[loop],
                              Rmtx_band_ri.T))
                for loop in range(num_bands)
            ]
            b_lst = [
                beta_ri_lst[loop] - \
                np.dot(lu_GtG_lst[loop],
                       np.dot(Rmtx_band_ri.T,
                              linalg.solve(lu_lst[loop],
                                           np.dot(Rmtx_band_ri, beta_ri_lst[loop]),
                                           check_finite=False)
                              )
                       )
                for loop in range(num_bands)
            ]
            a_Gb_lst = [
                a_ri[:, loop] - np.dot(G_lst[loop], b_lst[loop])
                for loop in range(num_bands)
            ]
    else:
        if use_lu:
            lu_lst = [
                linalg.lu_factor(np.dot(Rmtx_band_ri,
                                        linalg.lu_solve(lu_GtG_lst[loop],
                                                        Rmtx_band_ri.T,
                                                        check_finite=False)),
                                 overwrite_a=True, check_finite=False)
                for loop in range(num_bands)
            ]
            b_lst = [
                beta_ri_lst[loop] - \
                linalg.lu_solve(lu_GtG_lst[loop],
                                np.dot(Rmtx_band_ri.T,
                                       linalg.lu_solve(lu_lst[loop],
                                                       np.dot(Rmtx_band_ri, beta_ri_lst[loop]),
                                                       check_finite=False)
                                       ),
                                check_finite=False
                                )
                for loop in range(num_bands)
            ]
            a_Gb_lst = [
                a_ri[:, loop] - np.dot(G_lst[loop], b_lst[loop])
                for loop in range(num_bands)
            ]
        else:
            lu_lst = [
                np.dot(Rmtx_band_ri,
                       linalg.solve(lu_GtG_lst[loop],
                                    Rmtx_band_ri.T,
                                    check_finite=False))
                for loop in range(num_bands)
            ]
            b_lst = [
                beta_ri_lst[loop] - \
                linalg.solve(lu_GtG_lst[loop],
                             np.dot(Rmtx_band_ri.T,
                                    linalg.solve(lu_lst[loop],
                                                 np.dot(Rmtx_band_ri, beta_ri_lst[loop]),
                                                 check_finite=False)
                                    ),
                             check_finite=False)
                for loop in range(num_bands)
            ]
            a_Gb_lst = [
                a_ri[:, loop] - np.dot(G_lst[loop], b_lst[loop])
                for loop in range(num_bands)
            ]

    return linalg.norm(np.concatenate(a_Gb_lst)), lu_lst, b_lst


def sph_compute_mtx_obj_multiband(lu_GtG_lst, Tbeta_lst, Rmtx_band_ri,
                                  Q_H, num_bands, sz_coef, lu_lst=None,
                                  use_lu=True, use_GtGinv=False):
    """
    compute the matrix (M) in the objective function:
        min   c^H M c
        s.t.  c0^H c = 1

    :param lu_GtG_lst: list of G^H * G
    :param Tbeta_lst: list of Teoplitz matrices for beta-s
    :param Rmtx_band_ri: right dual matrix for the annihilating filter (same for each block -> not a list)
    :param Q_H: a rectangular matrix that extracts the effective lines of equations
    :param num_bands: number of sub-bands
    :param K: number of Dirac
    :return:
    """
    mtx = np.zeros((sz_coef, sz_coef), dtype=float)  # <= G, Tbeta, Rc0 are real-valued
    if lu_lst is None:
        if use_GtGinv:  # lu_GtG_lst actually stores GtGinv in this case
            for loop in range(num_bands):
                Tbeta_loop = np.dot(Q_H, Tbeta_lst[loop])
                mtx += np.dot(Tbeta_loop.T,
                              linalg.solve(np.dot(Rmtx_band_ri,
                                                  np.dot(lu_GtG_lst[loop],
                                                         Rmtx_band_ri.T)),
                                           Tbeta_loop, overwrite_a=True, check_finite=False)
                              )
        else:
            if use_lu:
                for loop in range(num_bands):
                    Tbeta_loop = np.dot(Q_H, Tbeta_lst[loop])
                    mtx += np.dot(Tbeta_loop.T,
                                  linalg.solve(np.dot(Rmtx_band_ri,
                                                      linalg.lu_solve(lu_GtG_lst[loop],
                                                                      Rmtx_band_ri.T,
                                                                      check_finite=False)),
                                               Tbeta_loop, overwrite_a=True, check_finite=False)
                                  )
            else:
                for loop in range(num_bands):
                    Tbeta_loop = np.dot(Q_H, Tbeta_lst[loop])
                    mtx += np.dot(Tbeta_loop.T,
                                  linalg.solve(np.dot(Rmtx_band_ri,
                                                      linalg.solve(lu_GtG_lst[loop],
                                                                   Rmtx_band_ri.T,
                                                                   check_finite=False)),
                                               Tbeta_loop, overwrite_a=True, check_finite=False)
                                  )
    else:
        if use_GtGinv:  # lu_GtG_lst actually stores GtGinv in this case
            for loop in range(num_bands):
                Tbeta_loop = np.dot(Q_H, Tbeta_lst[loop])
                mtx += np.dot(Tbeta_loop.T,
                              np.dot(lu_lst[loop], Tbeta_loop)
                              )
        else:
            if use_lu:
                for loop in range(num_bands):
                    Tbeta_loop = np.dot(Q_H, Tbeta_lst[loop])
                    mtx += np.dot(Tbeta_loop.T,
                                  linalg.lu_solve(lu_lst[loop], Tbeta_loop,
                                                  check_finite=False)
                                  )
            else:
                for loop in range(num_bands):
                    Tbeta_loop = np.dot(Q_H, Tbeta_lst[loop])
                    mtx += np.dot(Tbeta_loop.T,
                                  linalg.solve(lu_lst[loop], Tbeta_loop,
                                               check_finite=False)
                                  )
    return mtx


def sph_sel_coef_subset_ri(num_coef_row, num_coef_col, K):
    """
    matrix that indicates where the cooefficients are zero
    :param num_coef_row: total number of coefficients (for the annihilation along rows)
    :param num_coef_col: total number of coefficients (for the annihilation along columns)
    :param K: number of Dirac. The number of non-zero coefficients is K + 1
    :return:
    """
    sub_set_idx_row = np.sort(np.random.permutation(num_coef_row)[K + 1:])
    sub_set_idx_col = np.sort(np.random.permutation(num_coef_col)[K + 1:])

    S_row = np.eye(num_coef_row)[sub_set_idx_row, :]
    S_col = np.eye(num_coef_col)[sub_set_idx_col, :]

    # deal with specific cases when sub_set_idx has length 0
    if S_row.shape[0] == 0 and S_col.shape[0] == 0:
        S = np.zeros((0, num_coef_row + num_coef_col))
    elif S_row.shape[0] == 0 and S_col.shape[0] != 0:
        S = np.column_stack((np.zeros((S_col.shape[0], num_coef_row)), S_col))
    elif S_row.shape[0] != 0 and S_col.shape[0] == 0:
        S = np.column_stack((S_row, np.zeros((S_row.shape[0], num_coef_col))))
    else:
        S = linalg.block_diag(S_row, S_col)

    # we express all matrices and vectors in their real-valued extended form
    if S_row.shape[0] == 0 and S_col.shape[0] == 0:
        S_ri = np.zeros((0, 2 * (num_coef_row + num_coef_col)))
    else:
        S_ri = np.vstack((np.hstack((S.real, -S.imag)),
                          np.hstack((S.imag, S.real))))

    return S_ri


def sph_T_mtx_joint_ri(b_pos, b_neg, sz_coef_row0, sz_coef_row1,
                       sz_coef_col0, sz_coef_col1):
    """
    Convolution matrix associated with the uniformly sampled sinusoids
    :param b_pos: b for the case with m>=0 (complex valued)
    :param b_neg: b for the case with m<=0 (complex valued)
    :param sz_coef_row0: size0 of the annihilating filter along the row
    :param sz_coef_row1: size1 of the annihilating filter along the row
    :param sz_coef_col0: size0 of the annihilating filter along the column
    :param sz_coef_col1: size1 of the annihilating filter along the column
    :return:
    """
    b_pos_reshaped = up_tri_from_vec_pos_ms(b_pos)
    b_neg_reshaped = up_tri_from_vec_pos_ms(b_neg)
    T_pos_row_cpx = sph_convmtx2_valid(b_pos_reshaped,
                                       sz_coef_row0, sz_coef_row1)
    T_neg_row_cpx = sph_convmtx2_valid(b_neg_reshaped,
                                       sz_coef_row0, sz_coef_row1)
    T_pos_col_cpx = sph_convmtx2_valid(b_pos_reshaped,
                                       sz_coef_col0, sz_coef_col1)
    T_neg_col_cpx = sph_convmtx2_valid(b_neg_reshaped,
                                       sz_coef_col0, sz_coef_col1)
    T_cpx = linalg.block_diag(np.vstack((T_pos_row_cpx, T_neg_row_cpx)),
                              np.vstack((T_pos_col_cpx, T_neg_col_cpx)))
    return np.vstack((np.hstack((T_cpx.real, -T_cpx.imag)),
                      np.hstack((T_cpx.imag, T_cpx.real))
                      ))


def sph_R_mtx_joint_ri(coef_row, coef_col, L, M, expansion_mtx=None,
                       mtx_extract_b=None, **kwargs):
    """
    Right dual matrix associated with the annihilating filter coefficients.
    Here we express everything in real values
    :param coef_row: the annihilating filter coefficients (complex valued) along the row.
                convert to its complex-valued representation if needed.
    :param coef_col: the annihilating filter coefficients (complex valued) along the column.
                convert to its complex-valued representation if needed.
    :param L: max degree of spherical harmonics
    :param M: max order of spherical harmonics
    :return:
    """
    if 'sz_coef_row0' in kwargs and 'sz_coef_row1' in kwargs:
        sz_coef_row0 = kwargs['sz_coef_row0']
        sz_coef_row1 = kwargs['sz_coef_row1']

        coef_row_mtx = sph_reshape_coef(coef_row, sz_coef_row0, sz_coef_row1)
    else:
        coef_row_mtx = lower_tri_from_vec_left_half(coef_row)

    if 'sz_coef_col0' in kwargs and 'sz_coef_col1' in kwargs:
        sz_coef_col0 = kwargs['sz_coef_col0']
        sz_coef_col1 = kwargs['sz_coef_col1']

        coef_col_mtx = sph_reshape_coef(coef_col, sz_coef_col0, sz_coef_col1)
    else:
        coef_col_mtx = lower_tri_from_vec_left_half(coef_col)

    R_row_pos_cpx = sph_convmtx2_valid(coef_row_mtx, L + 1, M + 1)
    R_row_neg_cpx = sph_convmtx2_valid(coef_row_mtx, L, M)
    R_col_pos_cpx = sph_convmtx2_valid(coef_col_mtx, L + 1, M + 1)
    R_col_neg_cpx = sph_convmtx2_valid(coef_col_mtx, L, M)

    R_cpx = np.vstack((linalg.block_diag(R_row_pos_cpx, R_row_neg_cpx),
                       linalg.block_diag(R_col_pos_cpx, R_col_neg_cpx),
                       ))
    if expansion_mtx is None and mtx_extract_b is None:
        return np.vstack((np.hstack((R_cpx.real, -R_cpx.imag)),
                          np.hstack((R_cpx.imag, R_cpx.real))
                          ))
    elif expansion_mtx is None and mtx_extract_b is not None:
        return np.dot(np.vstack((np.hstack((R_cpx.real, -R_cpx.imag)),
                                 np.hstack((R_cpx.imag, R_cpx.real)))),
                      mtx_extract_b)
    elif expansion_mtx is not None and mtx_extract_b is not None:
        return np.dot(np.vstack((np.hstack((R_cpx.real, -R_cpx.imag)),
                                 np.hstack((R_cpx.imag, R_cpx.real))
                                 )),
                      np.dot(expansion_mtx, mtx_extract_b))
    else:
        return np.dot(np.vstack((np.hstack((R_cpx.real, -R_cpx.imag)),
                                 np.hstack((R_cpx.imag, R_cpx.real)))),
                      expansion_mtx)


def sph_reshape_coef(coef, sz0, sz1):
    if sz0 <= sz1:
        assert 2 * coef.size == (2 * sz1 - sz0 + 1) * sz0
        # number of elements in the triangle shaped part
        sz_tri_region = np.int(0.5 * (sz0 - 1) * sz0)
        coef_row_mtx = \
            np.column_stack((np.vstack((np.zeros((1, sz0 - 1), dtype=coef.dtype),
                                        lower_tri_from_vec_left_half(coef[:sz_tri_region]))),
                             np.reshape(coef[sz_tri_region:],
                                        (sz0, -1), order='F')
                             ))
    else:
        assert 2 * coef.size == (2 * sz0 - sz1 + 1) * sz1
        coef_row_mtx = np.zeros((sz0, sz1),
                                dtype=coef.dtype, order='F')
        vec_bg_idx = 0
        mtx_bg_idx = sz1 - 1
        for loop in range(sz1):
            vec_len_loop = sz0 - sz1 + 1 + loop
            coef_row_mtx[mtx_bg_idx:mtx_bg_idx + vec_len_loop, loop] = \
                coef[vec_bg_idx:vec_bg_idx + vec_len_loop]
            vec_bg_idx += vec_len_loop
            mtx_bg_idx -= 1
    return coef_row_mtx


def lower_tri_from_vec_left_half(vec):
    """
    build a matrix of the form
        0 0 0 x
        0 0 x x
        0 x x x
        x x x x
    from the given data vector.
    :param vec: the data vector
    :return:
    """
    L = np.int((-3 + np.sqrt(1 + 8 * vec.size)) * 0.5)
    assert (L + 1) * (L + 2) == 2 * vec.size
    mtx = np.zeros((L + 1, L + 1), dtype=vec.dtype)
    mtx_bg_idx = L
    vec_bg_idx = 0
    for loop in range(0, L + 1):
        vec_len_loop = loop + 1
        mtx[mtx_bg_idx:mtx_bg_idx + vec_len_loop, loop] = \
            vec[vec_bg_idx:vec_bg_idx + vec_len_loop]
        vec_bg_idx += vec_len_loop
        mtx_bg_idx -= 1
    return mtx


def low_tri_from_vec(vec):
    """
    build a matrix of the form

        0 0 0 x 0 0 0
        0 0 x x x 0 0       (*)
        0 x x x x x 0
        x x x x x x x

    from the given data matrix. This function is the inverse operator
    of the function vec_from_low_tri_col_by_col(mtx).
    :param vec: the data vector
    :return: a matrix of the form (*)
    """
    L = np.sqrt(vec.size).astype(int) - 1
    assert (L + 1) ** 2 == vec.size
    mtx = np.zeros((L + 1, 2 * L + 1), dtype=vec.dtype)
    bg_idx = 0
    for loop in range(2 * L + 1):
        if loop <= L:
            vec_len_loop = loop + 1
        else:
            vec_len_loop = 2 * L + 1 - loop
        mtx[-vec_len_loop::, loop] = vec[bg_idx:bg_idx + vec_len_loop]
        bg_idx += vec_len_loop
    return mtx


def sph_compute_size(sz0, sz1):
    if sz0 <= sz1:
        num_coef = np.int(0.5 * (2 * sz1 - sz0 + 1) * sz0)
    else:
        num_coef = np.int(0.5 * (2 * sz0 - sz1 + 1) * sz1)
    return num_coef


# ==== for 2D convolution matrices ====
def convmtx2(H, M, N):
    """
    build 2d convolution matrix
    :param H: 2d filter
    :param M: input signal dimension is M x N
    :param N: input signal dimension is M x N
    :return:
    """
    P, Q = H.shape
    blockHeight = int(M + P - 1)
    blockWidth = int(M)
    blockNonZeros = int(P * M)
    N_blockNonZeros = int(N * blockNonZeros)
    totalNonZeros = Q * N_blockNonZeros

    THeight = int((N + Q - 1) * blockHeight)
    TWidth = int(N * blockWidth)

    Tvals = np.zeros(totalNonZeros, dtype=H.dtype)
    Trows = np.zeros(totalNonZeros, dtype=int)
    Tcols = np.zeros(totalNonZeros, dtype=int)

    c = np.repeat(np.arange(1, M + 1)[:, np.newaxis], P, axis=1)
    r = np.repeat(np.reshape(c + np.arange(0, P)[np.newaxis], (-1, 1), order='F'), N, axis=1)
    c = np.repeat(c.flatten('F')[:, np.newaxis], N, axis=1)

    colOffsets = np.arange(N, dtype=int) * M
    colOffsets = (np.repeat(colOffsets[np.newaxis], M * P, axis=0) + c).flatten('F') - 1

    rowOffsets = np.arange(N, dtype=int) * blockHeight
    rowOffsets = (np.repeat(rowOffsets[np.newaxis], M * P, axis=0) + r).flatten('F') - 1

    for k in range(Q):
        val = (np.tile((H[:, k]).flatten(), (M, 1))).flatten('F')
        first = int(k * N_blockNonZeros)
        last = int(first + N_blockNonZeros)
        Trows[first:last] = rowOffsets
        Tcols[first:last] = colOffsets
        Tvals[first:last] = np.tile(val, N)
        rowOffsets += blockHeight

    # T = sps.coo_matrix((Tvals, (Trows, Tcols)),
    #                    shape=(THeight, TWidth)).toarray()
    T = np.zeros((THeight, TWidth), dtype=H.dtype)
    T[Trows, Tcols] = Tvals
    return T


def convmtx2_valid(H, M, N):
    """
    2d convolution matrix with the boundary condition 'valid', i.e., only filter
    within the given data block.
    :param H: 2d filter
    :param M: input signal dimension is M x N
    :param N: input signal dimension is M x N
    :return:
    """
    T = convmtx2(H, M, N)
    s_H0, s_H1 = H.shape
    S = np.zeros((s_H0 + M - 1, s_H1 + N - 1), dtype=bool)
    if M >= s_H0:
        S[s_H0 - 1: M, s_H1 - 1: N] = True
    else:
        S[M - 1: s_H0, N - 1: s_H1] = True
    T = T[S.flatten('F'), :]
    return T


def sph_convmtx2_valid(H, sz_data0, sz_data1):
    """
    2d convolution matrix for spherical harmonics, which are arranged as a triangle,
    with the boundary condition 'valid', i.e., only filter within the given data block.

    Here we assume the data block is of the shape:
        * * * * *
        * * * * 0
        * * * 0 0
        * * 0 0 0
        * 0 0 0 0
    and the filter is of the shape:
        0 0 *
        0 * *
        * * *
    which will be flipped around the center and move on top of the data block.

    :param H: 2d filter
    :param sz_data0: input signal dimension is M x sz_data1
    :param sz_data1: input signal dimension is M x sz_data1
    :return:
    """
    T = convmtx2_valid(H, sz_data0, sz_data1)
    sz_h0, sz_h1 = H.shape
    # now select the output that is non-trivially zero (due to the triangluar shape)
    sz_output0 = np.abs(sz_data0 - sz_h0) + 1
    sz_output1 = np.abs(sz_data1 - sz_h1) + 1

    row_sel_idx = np.where(~(np.tri(sz_output0, sz_output1,
                                    -1 + min(sz_output1 - sz_output0, 0),
                                    dtype=bool))[:, ::-1].flatten('F'))[0]
    if sz_data0 >= sz_h0:
        # R(c)
        col_sel_idx = np.where(~(np.tri(sz_data0, sz_data1,
                                        -1 + min(sz_data1 - sz_data0, 0),
                                        dtype=bool))[:, ::-1].flatten('F'))[0]
    else:
        # T(b)
        col_sel_idx = np.where(~(np.tri(sz_data0, sz_data1,
                                        -1 + min(sz_data1 - sz_data0, 0),
                                        dtype=bool))[::-1, :].flatten('F'))[0]
    return T[row_sel_idx, :][:, col_sel_idx]


# ============================================================================

def sph_recon_2d_dirac(a, p_mic_x, p_mic_y, p_mic_z, K, L, noise_level,
                       max_ini=50, stop_cri='mse', num_rotation=5,
                       verbose=False, update_G=False, **kwargs):
    """
    reconstruct point sources on the sphere from the visibility measurements
    :param a: the measured visibilities
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' y coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' z coordinates
    :param K: number of point sources
    :param L: maximum degree of spherical harmonics
    :param noise_level: noise level in the measured visiblities
    :param max_ini: maximum number of random initialisations used
    :param stop_cri: either 'mse' or 'max_iter'
    :param num_rotation: number of random initialisations
    :param verbose: print intermediate results to console or not
    :param update_G: update the linear mapping matrix in the objective function or not
    :return:
    """
    assert L >= K + np.sqrt(K) - 1
    num_mic = p_mic_x.shape[0]
    assert num_mic * (num_mic - 1) >= (L + 1) ** 2
    num_bands = p_mic_x.shape[1]
    # data used when the annihilation is enforced along each column
    a_ri = np.vstack((np.real(a), np.imag(a)))
    a_col_ri = a_ri
    # data used when the annihilation is enforced along each row
    a_row = a_ri

    if update_G:
        if 'G_iter' in kwargs:
            max_loop_G = kwargs['G_iter']
        else:
            max_loop_G = 2
    else:
        max_loop_G = 1

    # initialisation
    min_error_rotate = float('inf')
    colatitudek_opt = np.zeros(K)
    azimuthk_opt = np.zeros(K)
    alphak_opt = np.zeros(K)

    # SO(3) is defined by three angles
    rotate_angle_all_1 = np.random.rand(num_rotation) * np.pi * 2.
    rotate_angle_all_2 = np.random.rand(num_rotation) * np.pi
    rotate_angle_all_3 = np.random.rand(num_rotation) * np.pi * 2.
    # build rotation matrices for all angles
    for rand_rotate in range(num_rotation):
        rotate_angle_1 = rotate_angle_all_1[rand_rotate]
        rotate_angle_2 = rotate_angle_all_2[rand_rotate]
        rotate_angle_3 = rotate_angle_all_3[rand_rotate]
        # build rotation matrices
        rotate_mtx1 = np.array([[np.cos(rotate_angle_1), -np.sin(rotate_angle_1), 0],
                                [np.sin(rotate_angle_1), np.cos(rotate_angle_1), 0],
                                [0, 0, 1]])
        rotate_mtx2 = np.array([[np.cos(rotate_angle_2), 0, np.sin(rotate_angle_2)],
                                [0, 1, 0],
                                [-np.sin(rotate_angle_2), 0, np.cos(rotate_angle_2)]])
        rotate_mtx3 = np.array([[np.cos(rotate_angle_3), -np.sin(rotate_angle_3), 0],
                                [np.sin(rotate_angle_3), np.cos(rotate_angle_3), 0],
                                [0, 0, 1]])
        rotate_mtx = np.dot(rotate_mtx1, np.dot(rotate_mtx2, rotate_mtx3))

        # rotate microphone steering vector (due to change of coordinate w.r.t. the random rotation)
        p_mic_xyz_rotated = np.dot(rotate_mtx,
                                   np.vstack((np.reshape(p_mic_x, (1, -1), order='F'),
                                              np.reshape(p_mic_y, (1, -1), order='F'),
                                              np.reshape(p_mic_z, (1, -1), order='F')
                                              ))
                                   )
        p_mic_x_rotated = np.reshape(p_mic_xyz_rotated[0, :], p_mic_x.shape, order='F')
        p_mic_y_rotated = np.reshape(p_mic_xyz_rotated[1, :], p_mic_x.shape, order='F')
        p_mic_z_rotated = np.reshape(p_mic_xyz_rotated[2, :], p_mic_x.shape, order='F')
        mtx_freq2visibility = \
            sph_mtx_freq2visibility(L, p_mic_x_rotated, p_mic_y_rotated, p_mic_z_rotated)

        # linear transformation matrix that maps uniform samples of sinusoids to visibilities
        G_row = sph_mtx_fri2visibility_row_major(L, mtx_freq2visibility)
        G_col_ri = sph_mtx_fri2visibility_col_major_ri(L, mtx_freq2visibility)

        # ext_r, ext_i = sph_Hsym_ext(L)
        # mtx_hermi = linalg.block_diag(ext_r, ext_i)
        # mtx_repe = np.eye(num_bands, dtype=float)
        # G_col_ri_symb = np.dot(G_col_ri, np.kron(mtx_repe, mtx_hermi))

        for loop_G in range(max_loop_G):
            # parallel implentation
            partial_sph_dirac_recon = partial(sph_2d_dirac_v_and_h,
                                              G_row=G_row, a_row=a_row,
                                              G_col=G_col_ri, a_col=a_col_ri,
                                              K=K, L=L, noise_level=noise_level,
                                              max_ini=max_ini, stop_cri=stop_cri)
            '''
            coef_error_b_all = Parallel(n_jobs=2)(delayed(partial_sph_dirac_recon)(ins)
                                                  for ins in range(2))
            '''

            coef_error_b_all = [partial_sph_dirac_recon(ins) for ins in range(2)]

            c_row, error_row, b_opt_row = coef_error_b_all[0]
            c_col, error_col, b_opt_col = coef_error_b_all[1]

            # recover the signal parameters from the reconstructed annihilating filters
            tk = np.roots(np.squeeze(c_col))  # tk: cos(colatitude_k)
            tk[tk > 1] = 1
            tk[tk < -1] = -1
            colatitude_ks = np.real(np.arccos(tk))
            uk = np.roots(np.squeeze(c_row))
            azimuth_ks = np.mod(-np.angle(uk), 2. * np.pi)
            if verbose:
                # print(c_col)
                print('noise level: {0:.3e}'.format(noise_level))
                print('objective function value (column): {0:.3e}'.format(error_col))
                print('objective function value (row)   : {0:.3e}'.format(error_row))

            # find the correct associations of colatitude_ks and azimuth_ks
            # assume we have K^2 Diracs and find the amplitudes
            colatitude_k_grid, azimuth_k_grid = np.meshgrid(colatitude_ks, azimuth_ks)
            colatitude_k_grid = colatitude_k_grid.flatten('F')
            azimuth_k_grid = azimuth_k_grid.flatten('F')

            mtx_amp = sph_mtx_amp(colatitude_k_grid, azimuth_k_grid, p_mic_x_rotated,
                                  p_mic_y_rotated, p_mic_z_rotated)
            # the amplitudes are positive (sigma^2)
            alphak_recon = np.reshape(sp.optimize.nnls(np.vstack((mtx_amp.real, mtx_amp.imag)),
                                                       np.concatenate((a.real.flatten('F'),
                                                                       a.imag.flatten('F')))
                                                       )[0],
                                      (-1, p_mic_x.shape[1]), order='F')
            # extract the indices of the largest amplitudes for each sub-band
            alphak_sort_idx = np.argsort(np.abs(alphak_recon), axis=0)[:-K - 1:-1, :]
            # now use majority vote across all sub-bands
            idx_all = np.zeros(alphak_recon.shape, dtype=int)
            for loop in range(num_bands):
                idx_all[alphak_sort_idx[:, loop], loop] = 1
            idx_sel = np.argsort(np.sum(idx_all, axis=1))[:-K - 1:-1]
            colatitudek_recon_rotated = colatitude_k_grid[idx_sel]
            azimuthk_recon_rotated = azimuth_k_grid[idx_sel]
            # rotate back
            # transform to cartesian coordinate
            xk_recon, yk_recon, zk_recon = sph2cart(1, colatitudek_recon_rotated, azimuthk_recon_rotated)
            xyz_rotate_back = linalg.solve(rotate_mtx,
                                           np.vstack((xk_recon.flatten('F'),
                                                      yk_recon.flatten('F'),
                                                      zk_recon.flatten('F')
                                                      ))
                                           )
            xk_recon = np.reshape(xyz_rotate_back[0, :], xk_recon.shape, 'F')
            yk_recon = np.reshape(xyz_rotate_back[1, :], yk_recon.shape, 'F')
            zk_recon = np.reshape(xyz_rotate_back[2, :], zk_recon.shape, 'F')

            # transform back to spherical coordinate
            colatitudek_recon = np.arccos(zk_recon)
            azimuthk_recon = np.mod(np.arctan2(yk_recon, xk_recon), 2. * np.pi)
            # now use the correctly identified colatitude and azimuth to reconstruct the correct amplitudes
            mtx_amp_sorted = sph_mtx_amp(colatitudek_recon, azimuthk_recon, p_mic_x, p_mic_y, p_mic_z)
            alphak_recon = sp.optimize.nnls(np.vstack((mtx_amp_sorted.real, mtx_amp_sorted.imag)),
                                            np.concatenate((a.real.flatten('F'),
                                                            a.imag.flatten('F')))
                                            )[0]

            error_loop = linalg.norm(a.flatten('F') - np.dot(mtx_amp_sorted, alphak_recon))

            if verbose:
                # for debugging only
                print('colatitude_ks  : {0}'.format(np.degrees(colatitudek_recon)))
                print('azimuth_ks    : {0}'.format(np.degrees(np.mod(azimuthk_recon, 2 * np.pi))))
                print('error_loop: {0}\n'.format(error_loop))

            if error_loop < min_error_rotate:
                min_error_rotate = error_loop
                colatitudek_opt, azimuthk_opt, alphak_opt = colatitudek_recon, azimuthk_recon, alphak_recon

            # update the linear transformation matrix
            if update_G:
                G_row = sph_mtx_updated_G_row_major(colatitudek_recon_rotated, azimuthk_recon_rotated,
                                                    L, p_mic_x_rotated, p_mic_y_rotated,
                                                    p_mic_z_rotated, mtx_freq2visibility)
                G_col_ri = sph_mtx_updated_G_col_major_ri(colatitudek_recon_rotated,
                                                          azimuthk_recon_rotated, L,
                                                          p_mic_x_rotated,
                                                          p_mic_y_rotated,
                                                          p_mic_z_rotated,
                                                          mtx_freq2visibility)
                # G_col_ri_symb = np.dot(G_col_ri, np.kron(mtx_repe, mtx_hermi))
    return colatitudek_opt, azimuthk_opt, np.reshape(alphak_opt, (-1, num_bands), order='F')


# ======== functions for the reconstruction of colaitudes and azimuth ========
def add_noise_inner(visi_noiseless, var_noise, num_mic, Ns=1000):
    """
    add noise to the Fourier measurements
    :param visi_noiseless: the noiseless Fourier data
    :param var_noise: variance of the noise
    :param num_mic: number of stations
    :param Ns: number of observation samples used to estimate the covariance matrix
    :return:
    """
    SigmaMtx = visi_noiseless + var_noise * np.eye(*visi_noiseless.shape)
    wischart_mtx = np.kron(SigmaMtx.conj(), SigmaMtx) / Ns
    # the noise vairance matrix is given by the Cholesky decomposition
    noise_conv_mtx_sqrt = np.linalg.cholesky(wischart_mtx)
    # the noiseless visibility
    visi_noiseless_vec = np.reshape(visi_noiseless, (-1, 1), order='F')
    # TODO: find a way to add Hermitian symmetric noise here.
    noise = np.dot(noise_conv_mtx_sqrt,
                   np.random.randn(*visi_noiseless_vec.shape) +
                   1j * np.random.randn(*visi_noiseless_vec.shape)) / np.sqrt(2)
    visi_noisy = np.reshape(visi_noiseless_vec + noise, visi_noiseless.shape, order='F')
    # extract the off-diagonal entries
    extract_cond = np.reshape((1 - np.eye(num_mic)).T.astype(bool), (-1, 1), order='F')
    visi_noisy = np.reshape(np.extract(extract_cond, visi_noisy.T), (-1, 1), order='F')
    visi_noiseless_off_diag = \
        np.reshape(np.extract(extract_cond, visi_noiseless.T), (-1, 1), order='F')
    # calculate the equivalent SNR
    noise = visi_noisy - visi_noiseless_off_diag
    P = 20 * np.log10(linalg.norm(visi_noiseless_off_diag) / linalg.norm(noise))
    return visi_noisy, P, noise, visi_noiseless_off_diag


def sph_2d_dirac_v_and_h(direction, G_row, a_row, G_col, a_col, K, L, noise_level, max_ini, stop_cri):
    """
    used to run the reconstructions along horizontal and vertical directions in parallel.
    """
    if direction == 0:
        c_recon, error_recon, b_opt = \
            sph_2d_dirac_horizontal_ri(G_row, a_row, K, L, noise_level, max_ini, stop_cri)[:3]
    else:
        c_recon, error_recon, b_opt = \
            sph_2d_dirac_vertical_real_coef(G_col, a_col, K, L, noise_level, max_ini, stop_cri)[:3]
        # c_recon, error_recon, b_opt = \
        #     sph_2d_dirac_vertical_symb(G_col, a_col, K, L, noise_level, max_ini, stop_cri)[:3]
    return c_recon, error_recon, b_opt


def sph_2d_dirac_horizontal_ri(G_row, a_ri, K, L, noise_level, max_ini=100, stop_cri='mse'):
    """
    Dirac reconstruction on the sphere along the horizontal direction.
    :param G_row: the linear transformation matrix that links the given spherical harmonics
                    to the annihilable sequence.
    :param a_ri: the given visibility measurements
            CAUTION: here we assume that the spherical harmonics is rearranged (based on the
            sign of the order 'm'): the spherical harmonics with negative m-s are complex-conjuaged
            and flipped around m = 0, i.e., m varies from 0 to -L. The column that corresponds to
            m = 0 is stored twice.
            The total number of elements in 'a' is: (L + 1) * (L + 2)
    :param K: number of Diracs
    :param L: maximum degree of the spherical harmonics
    :param noise_level: noise level present in the visibility measurements
    :param max_ini: maximum number of initialisations allowed for the algorithm
    :param stop_cri: either 'mse' or 'max_iter'
    :return:
    """
    num_bands = a_ri.shape[1]
    a_ri = a_ri.flatten('F')
    compute_mse = (stop_cri == 'mse')
    # size of G_row: (L + 1)(L + 2) x (L + 1)^2
    GtG = np.dot(G_row.T, G_row)  # size: 2(L + 1)^2 x 2(L + 1)^2
    Gt_a = np.dot(G_row.T, a_ri)

    # maximum number of iterations with each initialisation
    max_iter = 50
    min_error = float('inf')
    # the least square solution incase G is rank deficient
    # size: 2(L + 1)^2 x num_bands
    beta = np.reshape(linalg.lstsq(G_row, a_ri)[0], (-1, num_bands), order='F')
    # print linalg.norm(np.dot(G_row, beta) - a_ri)
    # size of Tbeta: (L + 1 - K)^2 x (K + 1)
    Tbeta = sph_Tmtx_row_major_ri(beta, K, L)

    # repetition matrix for Rmtx (the same annihilating filter for all sub-bands)
    mtx_repe = np.eye(num_bands, dtype=float)

    # size of various matrices / vectors
    sz_G1 = 2 * (L + 1) ** 2 * num_bands

    sz_Tb0 = 2 * (L + 1 - K) * (L - K + 2) * num_bands
    sz_Tb1 = 2 * (K + 1)

    sz_Rc0 = 2 * (L + 1 - K) * (L - K + 2) * num_bands
    sz_Rc1 = 2 * (L + 1) ** 2 * num_bands

    sz_coef = 2 * (K + 1)
    sz_b = 2 * (L + 1) ** 2 * num_bands  # here the data that corresponds to m = 0 is only stored once.

    rhs = np.concatenate((np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float),
                          np.array(1, dtype=float)[np.newaxis]))
    rhs_bl = np.squeeze(np.vstack((Gt_a[:, np.newaxis],
                                   np.zeros((sz_Rc0, 1), dtype=Gt_a.dtype)
                                   ))
                        )

    # the main iteration with different random initialisations
    for ini in range(max_ini):
        c = np.random.randn(sz_coef, 1)  # the real and imaginary parts of the annihilating filter coefficients
        c0 = c.copy()
        error_seq = np.zeros(max_iter)
        R_loop = np.kron(mtx_repe, sph_Rmtx_row_major_ri(c, K, L))
        for inner in range(max_iter):
            Mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta.T,
                                             np.zeros((sz_coef, sz_Rc1)), c0
                                             )),
                                  np.hstack((Tbeta, np.zeros((sz_Tb0, sz_Tb0)),
                                             -R_loop, np.zeros((sz_Rc0, 1))
                                             )),
                                  np.hstack((np.zeros((sz_Rc1, sz_Tb1)), -R_loop.T,
                                             GtG, np.zeros((sz_Rc1, 1))
                                             )),
                                  np.hstack((c0.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
                                             ))
                                  ))
            # matrix should be Hermitian symmetric
            Mtx_loop = (Mtx_loop + Mtx_loop.T) * 0.5
            c = linalg.solve(Mtx_loop, rhs)[:sz_coef]

            R_loop = np.kron(mtx_repe, sph_Rmtx_row_major_ri(c, K, L))
            Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T
                                               )),
                                    np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))
                                               ))
                                    ))
            b_recon = linalg.solve(Mtx_brecon, rhs_bl)[:sz_b]

            error_seq[inner] = linalg.norm(a_ri - np.dot(G_row, b_recon))
            if error_seq[inner] < min_error:
                min_error = error_seq[inner]
                b_opt_row = b_recon
                c_opt_row = c[:K + 1] + 1j * c[K + 1:]  # real and imaginary parts

            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    return c_opt_row, min_error, b_opt_row, ini


def sph_2d_dirac_vertical_real_coef(G_col_ri, a_ri, K, L, noise_level, max_ini=100, stop_cri='mse'):
    """
    Dirac reconstruction on the sphere along the vertical direction.
    Here we use explicitly the fact that the annihilating filter is real-valuded.
    :param G_col: the linear transformation matrix that links the given spherical harmonics
                    to the annihilable sequence.
    :param a: the given spherical harmonics
    :param K: number of Diracs
    :param L: maximum degree of the spherical harmonics
    :param M: maximum order of the spherical harmonics (M <= L)
    :param noise_level: noise level present in the spherical harmonics
    :param max_ini: maximum number of initialisations allowed for the algorithm
    :param stop_cri: either 'mse' or 'maxiter'
    :return:
    """
    num_visi_ri, num_bands = a_ri.shape
    a_ri = a_ri.flatten('F')
    compute_mse = (stop_cri == 'mse')
    # size of G_col: (L + 1 + (2 * L + 1 - M) * M) x (L + 1 + (2 * L + 1 - M) * M)
    GtG = np.dot(G_col_ri.T, G_col_ri)
    Gt_a = np.dot(G_col_ri.T, a_ri)

    # maximum number of iterations with each initialisation
    max_iter = 50
    min_error = float('inf')
    beta_ri = np.reshape(linalg.lstsq(G_col_ri, a_ri)[0], (-1, num_bands), order='F')

    # size of Tbeta: num_bands * 2(L + 1 - K)^2 x (K + 1)
    Tbeta = sph_Tmtx_col_major_real_coef(beta_ri, K, L)

    # repetition matrix for Rmtx (the same annihilating filter for all sub-bands)
    mtx_repe = np.eye(num_bands, dtype=float)

    # size of various matrices / vectors
    sz_G1 = 2 * (L + 1) ** 2 * num_bands

    sz_Tb0 = 2 * (L + 1 - K) ** 2 * num_bands
    sz_Tb1 = K + 1

    sz_Rc0 = 2 * (L + 1 - K) ** 2 * num_bands
    sz_Rc1 = 2 * (L + 1) ** 2 * num_bands

    sz_coef = K + 1
    sz_b = 2 * (L + 1) ** 2 * num_bands

    rhs = np.concatenate((np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float),
                          np.array(1, dtype=float)[np.newaxis]))
    rhs_bl = np.squeeze(np.vstack((Gt_a[:, np.newaxis],
                                   np.zeros((sz_Rc0, 1), dtype=Gt_a.dtype)
                                   ))
                        )

    # the main iteration with different random initialisations
    for ini in range(max_ini):
        # c = np.random.randn(sz_coef, 1)  # we know that the coefficients are real-valuded
        # favor the polynomial to have roots inside unit circle
        c = np.sort(np.abs(np.random.randn(sz_coef, 1)))
        c0 = c.copy()
        error_seq = np.zeros(max_iter)
        R_loop = np.kron(mtx_repe, sph_Rmtx_col_major_real_coef(c, K, L))
        for inner in range(max_iter):
            Mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta.T,
                                             np.zeros((sz_coef, sz_Rc1)), c0
                                             )),
                                  np.hstack((Tbeta, np.zeros((sz_Tb0, sz_Tb0)),
                                             -R_loop, np.zeros((sz_Rc0, 1))
                                             )),
                                  np.hstack((np.zeros((sz_Rc1, sz_Tb1)), -R_loop.T,
                                             GtG, np.zeros((sz_Rc1, 1))
                                             )),
                                  np.hstack((c0.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
                                             ))
                                  ))
            # matrix should be Hermitian symmetric
            Mtx_loop = (Mtx_loop + Mtx_loop.T) * 0.5
            c = linalg.solve(Mtx_loop, rhs)[:sz_coef]

            R_loop = np.kron(mtx_repe, sph_Rmtx_col_major_real_coef(c, K, L))
            Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                    np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                    ))
            b_recon = linalg.solve(Mtx_brecon, rhs_bl)[:sz_b]

            error_seq[inner] = linalg.norm(a_ri - np.dot(G_col_ri, b_recon))
            if error_seq[inner] < min_error:
                min_error = error_seq[inner]
                b_opt_col = b_recon
                c_opt_col = c

            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    return c_opt_col, min_error, b_opt_col, ini


# def sph_Hsym_ext(L):
#     """
#     Expansion matrix to impose the Hermitian symmetry in the uniformly
#     sampled sinusoids \tilde{b}_nm.
#     We assume the input to the extension matrix is the tilde{b}_nm with m <= 0.
#     The first half of the data is the real-part of tilde{b}_nm for m <= 0;
#     while the second half of the data is the imaginary-part of tilde{b}_nm for m < 0.
#     The reason is that tilde{b}_n0 is real-valuded.
#     :param L: maximum degree of the spherical harmonics
#     :return:
#     """
#     # mtx = np.zeros((2 * (L + 1) ** 2, (L + 1) ** 2), dtype=float)
#     sz_non_pos = np.int((L + 1) * (L + 2) * 0.5)
#     sz_pos = np.int((L + 1) * L * 0.5)
#     mtx_real_neg = np.eye(sz_non_pos)
#     mtx_real_pos = np.zeros((sz_pos, sz_pos), dtype=float)
#     v_idx0 = 0
#     h_idx0 = sz_pos - L
#     for m in range(1, L + 1, 1):
#         vec_len = L + 1 - m
#         mtx_real_pos[v_idx0:v_idx0 + vec_len, h_idx0:h_idx0+vec_len] = np.eye(vec_len)
#         v_idx0 += vec_len
#         h_idx0 -= (vec_len - 1)
#     mtx_real = np.vstack((mtx_real_neg, np.hstack((mtx_real_pos, np.zeros((sz_pos, L + 1))))))
#     mtx_imag = np.vstack((mtx_real_neg[:, :sz_pos], -mtx_real_pos))
#     return mtx_real, mtx_imag


# def sph_2d_dirac_vertical_symb(G_col_ri, a_ri, K, L, noise_level,
#                                max_ini=100, stop_cri='mse'):
#     """
#     Dirac reconstruction on the sphere along the vertical direction.
#     Here we exploit the fact that the Diracs have real-valuded amplitudes. Hence b_nm is
#     Hermitian symmetric w.r.t. m.
#     Also we enforce the constraint that tk = cos theta_k is real-valued. Hence, the
#     annihilating filter coefficients (for co-latitudes) are real-valued.
#     Here we use explicitly the fact that the annihilating filter is real-valuded.
#     :param G_col: the linear transformation matrix that links the given spherical harmonics
#                     to the annihilable sequence.
#     :param a: the given spherical harmonics
#     :param K: number of Diracs
#     :param L: maximum degree of the spherical harmonics
#     :param M: maximum order of the spherical harmonics (M <= L)
#     :param noise_level: noise level present in the spherical harmonics
#     :param max_ini: maximum number of initialisations allowed for the algorithm
#     :param stop_cri: either 'mse' or 'maxiter'
#     :return:
#     """
#     num_visi_ri, num_bands = a_ri.shape
#     a_ri = a_ri.flatten('F')
#     compute_mse = (stop_cri == 'mse')
#
#     # size of G_col: (L + 1 + (2 * L + 1 - M) * M) x (L + 1 + (2 * L + 1 - M) * M)
#     GtG = np.dot(G_col_ri.T, G_col_ri)
#     Gt_a = np.dot(G_col_ri.T, a_ri)
#
#     # maximum number of iterations with each initialisation
#     max_iter = 50
#     min_error = float('inf')
#     beta_ri = np.reshape(linalg.lstsq(G_col_ri, a_ri)[0], (-1, num_bands), order='F')
#
#     # size of Tbeta: num_bands * (L + 1 - K)^2 x (K + 1)
#     Tbeta = sph_Tmtx_col_real_coef_symb(beta_ri, K, L)
#
#     # repetition matrix for Rmtx (the same annihilating filter for all sub-bands)
#     mtx_repe = np.eye(num_bands, dtype=float)
#
#     # size of various matrices / vectors
#     sz_G1 = (L + 1) ** 2 * num_bands
#
#     sz_Tb0 = (L + 1 - K) ** 2 * num_bands
#     sz_Tb1 = K + 1
#
#     sz_Rc0 = (L + 1 - K) ** 2 * num_bands
#     sz_Rc1 = (L + 1) ** 2 * num_bands
#
#     sz_coef = K + 1
#     sz_b = (L + 1) ** 2 * num_bands
#
#     rhs = np.concatenate((np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float),
#                           np.array(1, dtype=float)[np.newaxis]))
#     rhs_bl = np.squeeze(np.vstack((Gt_a[:, np.newaxis],
#                                    np.zeros((sz_Rc0, 1), dtype=Gt_a.dtype)
#                                    ))
#                         )
#
#     # the main iteration with different random initialisations
#     for ini in range(max_ini):
#         c = np.random.randn(sz_coef, 1)  # we know that the coefficients are real-valuded
#         c0 = c.copy()
#         error_seq = np.zeros(max_iter)
#         R_loop = np.kron(mtx_repe, sph_Rmtx_col_real_coef_symb(c, K, L))
#         for inner in range(max_iter):
#             Mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta.T,
#                                              np.zeros((sz_coef, sz_Rc1)), c0
#                                              )),
#                                   np.hstack((Tbeta, np.zeros((sz_Tb0, sz_Tb0)),
#                                              -R_loop, np.zeros((sz_Rc0, 1))
#                                              )),
#                                   np.hstack((np.zeros((sz_Rc1, sz_Tb1)), -R_loop.T,
#                                              GtG, np.zeros((sz_Rc1, 1))
#                                              )),
#                                   np.hstack((c0.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
#                                              ))
#                                   ))
#             # matrix should be Hermitian symmetric
#             Mtx_loop = (Mtx_loop + Mtx_loop.T) * 0.5
#             # c = linalg.solve(Mtx_loop, rhs)[:sz_coef]
#             c = linalg.lstsq(Mtx_loop, rhs)[0][:sz_coef]
#
#             R_loop = np.kron(mtx_repe, sph_Rmtx_col_real_coef_symb(c, K, L))
#             Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
#                                     np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
#                                     ))
#             # b_recon = linalg.solve(Mtx_brecon, rhs_bl)[:sz_b]
#             b_recon = linalg.lstsq(Mtx_brecon, rhs_bl)[0][:sz_b]
#
#             error_seq[inner] = linalg.norm(a_ri - np.dot(G_col_ri, b_recon))
#             if error_seq[inner] < min_error:
#                 min_error = error_seq[inner]
#                 b_opt_col = b_recon
#                 c_opt_col = c
#
#             if min_error < noise_level and compute_mse:
#                 break
#
#         if min_error < noise_level and compute_mse:
#             break
#     return c_opt_col, min_error, b_opt_col, ini


# -----------------------------------------------
def sph_mtx_amp_inner(p_mic_x_loop, p_mic_y_loop, p_mic_z_loop, xk, yk, zk):
    """
    inner loop (for each sub-band) used for building the matrix for the least
    square estimation of Diracs' amplitudes
    :param p_mic_x_loop: a vector that contains the microphone x-coordinates (multiplied by mid-band frequency)
    :param p_mic_y_loop: a vector that contains the microphone y-coordinates (multiplied by mid-band frequency)
    :param p_mic_z_loop: a vector that contains the microphone z-coordinates (multiplied by mid-band frequency)
    :param xk: source location (x-coordinate)
    :param yk: source location (y-coordinate)
    :param zk: source location (z-coordinate)
    :return:
    """
    num_mic_per_band = p_mic_x_loop.size
    K = xk.size

    mtx_amp_blk = np.zeros(((num_mic_per_band - 1) * num_mic_per_band, K),
                           dtype=complex, order='C')
    count_inner = 0
    for q in range(num_mic_per_band):
        p_x_outer = p_mic_x_loop[q]
        p_y_outer = p_mic_y_loop[q]
        p_z_outer = p_mic_z_loop[q]
        for qp in range(num_mic_per_band):
            if not q == qp:
                p_x_qqp = p_x_outer - p_mic_x_loop[qp]
                p_y_qqp = p_y_outer - p_mic_y_loop[qp]
                p_z_qqp = p_z_outer - p_mic_z_loop[qp]
                mtx_amp_blk[count_inner, :] = np.exp(-1j * (xk * p_x_qqp +
                                                            yk * p_y_qqp +
                                                            zk * p_z_qqp))
                count_inner += 1
    return mtx_amp_blk


def sph_mtx_amp_inner_ri(p_mic_x_loop, p_mic_y_loop, p_mic_z_loop, xk, yk, zk):
    """
    inner loop (for each sub-band) used for building the matrix for the least
    square estimation of Diracs' amplitudes
    :param p_mic_x_loop: a vector that contains the microphone x-coordinates (multiplied by mid-band frequency)
    :param p_mic_y_loop: a vector that contains the microphone y-coordinates (multiplied by mid-band frequency)
    :param p_mic_z_loop: a vector that contains the microphone z-coordinates (multiplied by mid-band frequency)
    :param xk: source location (x-coordinate)
    :param yk: source location (y-coordinate)
    :param zk: source location (z-coordinate)
    :return:
    """
    num_mic_per_band = p_mic_x_loop.size
    K = xk.size

    mtx_amp_blk = np.zeros(((num_mic_per_band - 1) * num_mic_per_band, K),
                           dtype=complex, order='C')
    count_inner = 0
    for q in range(num_mic_per_band):
        p_x_outer = p_mic_x_loop[q]
        p_y_outer = p_mic_y_loop[q]
        p_z_outer = p_mic_z_loop[q]
        for qp in range(num_mic_per_band):
            if not q == qp:
                p_x_qqp = p_x_outer - p_mic_x_loop[qp]
                p_y_qqp = p_y_outer - p_mic_y_loop[qp]
                p_z_qqp = p_z_outer - p_mic_z_loop[qp]
                mtx_amp_blk[count_inner, :] = np.exp(-1j * (xk * p_x_qqp +
                                                            yk * p_y_qqp +
                                                            zk * p_z_qqp))
                count_inner += 1
    return cpx_mtx2real(mtx_amp_blk)


def sph_mtx_amp(colatitude_k, azimuth_k, p_mic_x, p_mic_y, p_mic_z):
    num_bands = p_mic_x.shape[1]

    xk, yk, zk = sph2cart(1, colatitude_k, azimuth_k)  # on S^2, -> r = 1
    # reshape xk, yk, zk to use broadcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    zk = np.reshape(zk, (1, -1), order='F')

    # parallel implementation
    partial_mtx_amp_mtx_inner = partial(sph_mtx_amp_inner, xk=xk, yk=yk, zk=zk)
    '''
    mtx_amp_lst = \
        Parallel(n_jobs=-1,
                 backend='threading')(delayed(partial_mtx_amp_mtx_inner)(p_mic_x[:, band_loop],
                                                                         p_mic_y[:, band_loop],
                                                                         p_mic_z[:, band_loop])
                                      for band_loop in range(num_bands))
    '''
    mtx_amp_lst = [partial_mtx_amp_mtx_inner(
        p_mic_x[:, band_loop],
        p_mic_y[:, band_loop],
        p_mic_z[:, band_loop]
    ) for band_loop in range(num_bands)]

    return linalg.block_diag(*mtx_amp_lst)


def sph_mtx_amp_ri(colatitude_k, azimuth_k, p_mic_x, p_mic_y, p_mic_z):
    num_bands = p_mic_x.shape[1]

    xk, yk, zk = sph2cart(1, colatitude_k, azimuth_k)  # on S^2, -> r = 1
    # reshape xk, yk, zk to use broadcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    zk = np.reshape(zk, (1, -1), order='F')

    # parallel implementation
    partial_mtx_amp_mtx_inner = partial(sph_mtx_amp_inner_ri, xk=xk, yk=yk, zk=zk)
    '''
    mtx_amp_lst = \
        Parallel(n_jobs=-1,
                 backend='threading')(delayed(partial_mtx_amp_mtx_inner)(p_mic_x[:, band_loop],
                                                                         p_mic_y[:, band_loop],
                                                                         p_mic_z[:, band_loop])
                                      for band_loop in range(num_bands))
    '''
    mtx_amp_lst = \
        mtx_amp_lst = [partial_mtx_amp_mtx_inner(
        p_mic_x[:, band_loop],
        p_mic_y[:, band_loop],
        p_mic_z[:, band_loop]
    ) for band_loop in range(num_bands)]

    return linalg.block_diag(*mtx_amp_lst)


# ===== functions to build R and T matrices =====
def sph_Rmtx_row_major_ri(coef_ri, K, L):
    """
    the right dual matrix associated with the FRI sequence b_nm.
    :param coef_ri: annihilating filter coefficinets represented in terms of
                    real and imaginary parts, i.e., a REAL-VALUED vector!
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    num_real = K + 1
    coef_real = coef_ri[:num_real]
    coef_imag = coef_ri[num_real:]
    R_real = sph_Rmtx_row_major_no_extend(coef_real, K, L)
    R_imag = sph_Rmtx_row_major_no_extend(coef_imag, K, L)
    expand_mtx_real = sph_build_exp_mtx(L, 'real')
    expand_mtx_imag = sph_build_exp_mtx(L, 'imaginary')

    R = np.vstack((np.hstack((np.dot(R_real, expand_mtx_real),
                              -np.dot(R_imag, expand_mtx_imag))),
                   np.hstack((np.dot(R_imag, expand_mtx_real),
                              np.dot(R_real, expand_mtx_imag)))
                   ))
    return R


def sph_Rmtx_row_major_no_extend(coef, K, L):
    """
    the right dual matrix associated with the FRI sequence \tilde{b}.
    :param coef: annihilating filter coefficinets
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    # K = coef.size - 1
    coef = np.squeeze(coef)
    assert K <= L + 1
    Rmtx_half = np.zeros((np.int((L - K + 2) * (L - K + 1) * 0.5),
                          np.int((L + K + 2) * (L - K + 1) * 0.5)),
                         dtype=coef.dtype)
    v_bg_idx = 0
    h_bg_idx = 0
    for n in range(0, L - K + 1):
        # input singal length
        vec_len = L + 1 - n

        blk_size0 = vec_len - K
        if blk_size0 == 1:
            col = coef[-1]
            row = coef[::-1]
        else:
            col = np.concatenate((np.array([coef[-1]]), np.zeros(vec_len - K - 1)))
            row = np.concatenate((coef[::-1], np.zeros(vec_len - K - 1)))

        Rmtx_half[v_bg_idx:v_bg_idx + blk_size0,
        h_bg_idx:h_bg_idx + vec_len] = linalg.toeplitz(col, row)

        v_bg_idx += blk_size0
        h_bg_idx += vec_len
    # only part of the data is longer enough to be annihilated
    # extract that part, which is equiavalent to padding zero columns on both ends
    # expand the data by repeating the data with m = 0 (used for annihilations on both sides)
    mtx_col2row = trans_col2row_pos_ms(L)[:np.int((L + K + 2) * (L - K + 1) * 0.5), :]
    return np.kron(np.eye(2), np.dot(Rmtx_half, mtx_col2row))


def sph_Tmtx_row_major_ri(b_ri, K, L):
    """
    The REAL-VALUED data b_ri is assumed to be arranged as a column vector.
    The first half (to be exact, the first (L+1)(L+2)/2 elements) is the real
    part of the FRI sequence; while the second half is the imaginary part of
    the FRI sequence.

    :param b: the annihilable sequence
    :param K: the annihilating filter is of size K + 1
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    num_bands = b_ri.shape[1]
    num_real = (L + 1) ** 2
    blk_sz0 = 2 * (L + 1 - K) * (L - K + 2)
    T = np.zeros((blk_sz0 * num_bands, 2 * (K + 1)), dtype=float)
    idx0 = 0
    for loop in range(num_bands):
        b_real = b_ri[:num_real, loop]
        b_imag = b_ri[num_real:, loop]
        T_real = sph_Tmtx_row_major(b_real, K, L, 'real')
        T_imag = sph_Tmtx_row_major(b_imag, K, L, 'imaginary')
        T[idx0:idx0 + blk_sz0, :] = np.vstack((np.hstack((T_real, -T_imag)),
                                               np.hstack((T_imag, T_real))
                                               ))
        idx0 += blk_sz0
    return T


def sph_Tmtx_row_major(b, K, L, option='real'):
    """
    The data b is assumed to be arranged as a column vector
    :param b: the annihilable sequence
    :param K: the annihilating filter is of size K + 1
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    Tmtx_half_pos = np.zeros((np.int((L - K + 2) * (L - K + 1) * 0.5),
                              K + 1),
                             dtype=b.dtype)
    Tmtx_half_neg = np.zeros((np.int((L - K + 2) * (L - K + 1) * 0.5),
                              K + 1),
                             dtype=b.dtype)
    # exapand the sequence b and rearrange row by row
    expand_mtx = sph_build_exp_mtx(L, option)
    mtx_col2row = trans_col2row_pos_ms(L)
    b_pn = np.dot(expand_mtx, b)
    sz_b_pn = b_pn.size
    assert sz_b_pn % 2 == 0
    b_pos = np.dot(mtx_col2row, b_pn[:np.int(sz_b_pn * 0.5)])
    b_neg = np.dot(mtx_col2row, b_pn[np.int(sz_b_pn * 0.5):])
    data_bg_idx = 0
    blk_bg_idx = 0
    for n in range(0, L - K + 1):
        vec_len = L + 1 - n
        blk_size0 = vec_len - K
        vec_pos_loop = b_pos[data_bg_idx:data_bg_idx + vec_len]
        vec_neg_loop = b_neg[data_bg_idx:data_bg_idx + vec_len]

        Tmtx_half_pos[blk_bg_idx:blk_bg_idx + blk_size0, :] = \
            linalg.toeplitz(vec_pos_loop[K::], vec_pos_loop[K::-1])
        Tmtx_half_neg[blk_bg_idx:blk_bg_idx + blk_size0, :] = \
            linalg.toeplitz(vec_neg_loop[K::], vec_neg_loop[K::-1])

        data_bg_idx += vec_len
        blk_bg_idx += blk_size0
    return np.vstack((Tmtx_half_pos, Tmtx_half_neg))


# -----------------------------------------------
# def sph_Rmtx_col_real_coef_symb(coef, K, L):
#     """
#     the right dual matrix associated with the FRI sequence \tilde{b}.
#     Here we use explicitly the fact that the coeffiients are real.
#     Additionally, b_tilde is Hermitian symmetric.
#     :param coef: annihilating filter coefficinets
#     :param L: maximum degree of the spherical harmonics
#     :return:
#     """
#     coef = np.squeeze(coef)
#     assert K <= L + 1
#     R_sz0 = (L + 1 - K) ** 2
#
#     Rr_sz0 = np.int((L - K + 1) * (L - K + 2) * 0.5)
#     Rr_sz1 = np.int((L -K + 1) * (L + K + 2) * 0.5)
#
#     Ri_sz0 = np.int((L - K) * (L - K + 1) * 0.5)
#     Ri_sz1 = np.int((L - K) * (L + K + 1) * 0.5)
#
#     Rr = np.zeros((Rr_sz0, Rr_sz1), dtype=float)
#     Ri = np.zeros((Ri_sz0, Ri_sz1), dtype=float)
#
#     v_bg_idx = 0
#     h_bg_idx = 0
#
#     for m in range(-L + K, 1):
#         vec_len = L + 1 - np.abs(m)
#         blk_size0 = vec_len - K
#         if blk_size0 == 1:
#             col = coef[-1]
#             row = coef[::-1]
#         else:
#             col = np.concatenate((np.array([coef[-1]]), np.zeros(vec_len - K - 1)))
#             row = np.concatenate((coef[::-1], np.zeros(vec_len - K - 1)))
#
#         mtx_blk = linalg.toeplitz(col, row)
#         Rr[v_bg_idx:v_bg_idx + blk_size0, h_bg_idx:h_bg_idx + vec_len] = mtx_blk
#         if not m == 0:
#             Ri[v_bg_idx:v_bg_idx + blk_size0, h_bg_idx:h_bg_idx + vec_len] = mtx_blk
#
#     # only part of the data is longer enough to be annihilated
#     # extract that part, which is equiavalent to padding zero columns on both ends
#     extract_mtx_r = np.eye(Rr_sz1, np.int((L + 1) * (L + 2) * 0.5),
#                            np.int(K * (K + 1) * 0.5))
#     extract_mtx_i = np.eye(Ri_sz1, np.int(L * (L + 1) * 0.5),
#                            np.int(K * (K + 1) * 0.5))
#     return linalg.block_diag(np.dot(Rr, extract_mtx_r), np.dot(Ri, extract_mtx_i))


def sph_Rmtx_col_major_real_coef(coef, K, L):
    """
    the right dual matrix associated with the FRI sequence \tilde{b}.
    Here we use explicitly the fact that the coefficients are real.
    :param coef: annihilating filter coefficinets
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    return np.kron(np.eye(2), sph_Rmtx_col_major(coef, K, L))


def sph_Rmtx_col_major(coef, K, L):
    """
    the right dual matrix associated with the FRI sequence \tilde{b}.
    :param coef: annihilating filter coefficinets
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    # K = coef.size - 1
    coef = np.squeeze(coef)
    assert K <= L + 1
    Rmtx = np.zeros(((L + 1 - K) ** 2,
                     (K + 1 + L) * (L - K) + L + 1),
                    dtype=coef.dtype)
    v_bg_idx = 0
    h_bg_idx = 0
    for m in range(-L + K, L - K + 1):
        # input singal length
        vec_len = L + 1 - np.abs(m)

        blk_size0 = vec_len - K
        if blk_size0 == 1:
            col = coef[-1]
            row = coef[::-1]
        else:
            col = np.concatenate((np.array([coef[-1]]), np.zeros(vec_len - K - 1)))
            row = np.concatenate((coef[::-1], np.zeros(vec_len - K - 1)))

        Rmtx[v_bg_idx:v_bg_idx + blk_size0, h_bg_idx:h_bg_idx + vec_len] = \
            linalg.toeplitz(col, row)

        v_bg_idx += blk_size0
        h_bg_idx += vec_len
    # only part of the data is longer enough to be annihilated
    # extract that part, which is equiavalent to padding zero columns on both ends
    extract_mtx = np.eye((K + 1 + L) * (L - K) + L + 1, (L + 1) ** 2, np.int((K + 1) * K * 0.5))
    return np.dot(Rmtx, extract_mtx)


def sph_Tmtx_col_real_coef_symb(b_tilde_ri, K, L):
    """
    build Toeplitz matrix from the given data b_tilde.
    Here we also exploit the fact that b_tilde is Hermitian symmetric.
    :param b_tilde: the given data in matrix form with size: (L + 1)(2L + 1)
    :param K: the filter size is K + 1
    :return:
    """
    num_bands = b_tilde_ri.shape[1]
    num_real = np.int((L + 1) * (L + 2) * 0.5)
    Tblk_sz0 = (L + 1 - K) ** 2
    Tr_sz0 = np.int((L - K + 2) * (L - K + 1) * 0.5)
    Ti_sz0 = np.int((L - K) * (L - K + 1) * 0.5)
    T = np.zeros((Tblk_sz0 * num_bands, K + 1), dtype=float)
    idx0 = 0
    for loop in range(num_bands):
        b_tilde_r = b_tilde_ri[:num_real, loop]
        b_tilde_i = b_tilde_ri[num_real:, loop]
        Tr = np.zeros((Tr_sz0, K + 1), dtype=float)
        Ti = np.zeros((Ti_sz0, K + 1), dtype=float)
        v_bg_idx = 0
        data_bg_idx = np.int(K * (K + 1) * 0.5)
        for m in range(-L + K, 1):
            # input singal length
            vec_len = L + 1 - np.abs(m)
            blk_size0 = vec_len - K
            vec_r_loop = b_tilde_r[data_bg_idx:data_bg_idx + vec_len]
            Tr[v_bg_idx:v_bg_idx + blk_size0, :] = \
                linalg.toeplitz(vec_r_loop[K::], vec_r_loop[K::-1])

            if not m == 0:
                vec_i_loop = b_tilde_i[data_bg_idx:data_bg_idx + vec_len]
                Ti[v_bg_idx:v_bg_idx + blk_size0, :] = \
                    linalg.toeplitz(vec_i_loop[K::], vec_i_loop[K::-1])

            v_bg_idx += blk_size0

        T[idx0:idx0 + Tblk_sz0] = np.vstack((Tr, Ti))
        idx0 += Tblk_sz0
    return T


def sph_Tmtx_col_major_real_coef(b_tilde_ri, K, L):
    """
    build Toeplitz matrix from the given data b_tilde.
    :param b_tilde: the given data in matrix form with size: (L + 1)(2L + 1)
    :param K: the filter size is K + 1
    :return:
    """
    num_bands = b_tilde_ri.shape[1]
    num_real = (L + 1) ** 2
    Tblk_sz0 = 2 * (L + 1 - K) ** 2
    T = np.zeros((Tblk_sz0 * num_bands, K + 1), dtype=float)
    idx0 = 0
    for loop in range(num_bands):
        b_tilde_r = up_tri_from_vec(b_tilde_ri[:num_real, loop])
        b_tilde_i = up_tri_from_vec(b_tilde_ri[num_real:, loop])
        Tr = np.zeros(((L + 1 - K) ** 2, K + 1), dtype=float)
        Ti = np.zeros(((L + 1 - K) ** 2, K + 1), dtype=float)
        v_bg_idx = 0
        for m in range(-L + K, L - K + 1):
            # input singal length
            vec_len = L + 1 - np.abs(m)
            blk_size0 = vec_len - K
            vec_r_loop = b_tilde_r[:vec_len, m + L]
            vec_i_loop = b_tilde_i[:vec_len, m + L]

            Tr[v_bg_idx:v_bg_idx + blk_size0, :] = \
                linalg.toeplitz(vec_r_loop[K::], vec_r_loop[K::-1])
            Ti[v_bg_idx:v_bg_idx + blk_size0, :] = \
                linalg.toeplitz(vec_i_loop[K::], vec_i_loop[K::-1])

            v_bg_idx += blk_size0

        T[idx0:idx0 + Tblk_sz0] = np.vstack((Tr, Ti))
        idx0 += Tblk_sz0
    return T


# ========= utilities for building G matrix ===========
def sph_mtx_updated_G_col_major_ri(colatitude_recon, azimuth_recon, L, p_mic_x,
                                   p_mic_y, p_mic_z, mtx_freq2visibility):
    """
    update the linear transformation matrix that links the FRI sequence to the visibilities.
    Here the input vector to this linear transformation is FRI seuqnce that is re-arranged
    column by column.
    :param colatitude_recon: reconstructed co-latitude(s)
    :param azimuth_recon: reconstructed azimuth(s)
    :param L: degree of the spherical harmonics
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' y coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' z coordinates
    :param mtx_freq2visibility: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :return:
    """
    num_bands = p_mic_x.shape[1]
    # the spherical harmonics basis evaluated at the reconstructed Dirac locations
    m_grid, l_grid = np.meshgrid(np.arange(-L, L + 1, step=1, dtype=int),
                                 np.arange(0, L + 1, step=1, dtype=int))
    m_grid = vec_from_low_tri_col_by_col(m_grid)[:, np.newaxis]
    l_grid = vec_from_low_tri_col_by_col(l_grid)[:, np.newaxis]
    # reshape colatitude_recon and azimuth_recon to use broadcasting
    colatitude_recon = np.reshape(colatitude_recon, (1, -1), order='F')
    azimuth_recon = np.reshape(azimuth_recon, (1, -1), order='F')
    mtx_Ylm_ri = cpx_mtx2real(np.conj(sph_harm_ufnc(l_grid, m_grid, colatitude_recon, azimuth_recon)))
    # the mapping from FRI sequence to Diracs amplitude
    mtx_fri2freq_ri = cpx_mtx2real(sph_mtx_fri2freq_col_major(L))
    mtx_Tinv_Y_ri = np.kron(np.eye(num_bands),
                            linalg.solve(mtx_fri2freq_ri, mtx_Ylm_ri))
    # G1
    # mtx_fri2amp_ri = linalg.solve(np.dot(mtx_Tinv_Y_ri.T,
    #                                      mtx_Tinv_Y_ri),
    #                               mtx_Tinv_Y_ri.T)
    mtx_fri2amp_ri = linalg.lstsq(mtx_Tinv_Y_ri, np.eye(mtx_Tinv_Y_ri.shape[0]))[0]
    # the mapping from Diracs amplitude to visibilities
    mtx_amp2visibility_ri = \
        sph_mtx_amp_ri(colatitude_recon, azimuth_recon, p_mic_x, p_mic_y, p_mic_z)
    # the mapping from FRI sequence to visibilities (G0)
    mtx_fri2visibility_ri = linalg.block_diag(*[np.dot(cpx_mtx2real(mtx_freq2visibility[G_blk_count]),
                                                       mtx_fri2freq_ri)
                                                for G_blk_count in range(num_bands)])
    # projection to the null space of mtx_fri2amp
    # mtx_null_proj = np.eye(2 * num_bands * (L + 1) ** 2) - \
    #                 np.dot(mtx_fri2amp_ri.T,
    #                        linalg.solve(np.dot(mtx_fri2amp_ri,
    #                                            mtx_fri2amp_ri.T),
    #                                     mtx_fri2amp_ri))
    mtx_null_proj = np.eye(2 * num_bands * (L + 1) ** 2) - \
                    np.dot(mtx_fri2amp_ri.T,
                           linalg.lstsq(mtx_fri2amp_ri.T,
                                        np.eye(mtx_fri2amp_ri.shape[1]))[0])
    G_updated = np.dot(mtx_amp2visibility_ri, mtx_fri2amp_ri) + np.dot(mtx_fri2visibility_ri, mtx_null_proj)
    return G_updated


def sph_mtx_updated_G_row_major(colatitude_recon, azimuth_recon, L, p_mic_x,
                                p_mic_y, p_mic_z, mtx_freq2visibility):
    """
    update the linear transformation matrix that links the FRI sequence to the visibilities.
    Here the input vector to this linear transformation is FRI seuqnce that is re-arranged
    row by row.
    :param colatitude_recon: reconstructed co-latitude(s)
    :param azimuth_recon: reconstructed azimuth(s)
    :param L: degree of the spherical harmonics
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' y coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' z coordinates
    :param mtx_freq2visibility: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :return:
    """
    num_bands = p_mic_x.shape[1]
    # mtx_fri2freq_ri links the FRI sequence to spherical harmonics of degree L
    mtx_fri2freq_ri = sph_mtx_fri2freq_row_major(L)
    # -------------------------------------------------
    # the spherical harmonics basis evaluated at the reconstructed Dirac locations
    m_grid, l_grid = np.meshgrid(np.arange(-L, L + 1, step=1, dtype=int),
                                 np.arange(0, L + 1, step=1, dtype=int))
    m_grid = vec_from_low_tri_col_by_col(m_grid)[:, np.newaxis]
    l_grid = vec_from_low_tri_col_by_col(l_grid)[:, np.newaxis]
    # reshape colatitude_recon and azimuth_recon to use broadcasting
    colatitude_recon = np.reshape(colatitude_recon, (1, -1), order='F')
    azimuth_recon = np.reshape(azimuth_recon, (1, -1), order='F')
    mtx_Ylm_ri = cpx_mtx2real(np.conj(sph_harm_ufnc(l_grid, m_grid, colatitude_recon, azimuth_recon)))
    mtx_Tinv_Y_ri = np.kron(np.eye(num_bands),
                            linalg.solve(mtx_fri2freq_ri, mtx_Ylm_ri))
    # G1
    # mtx_fri2amp_ri = linalg.solve(np.dot(mtx_Tinv_Y_ri.T, mtx_Tinv_Y_ri),
    #                               mtx_Tinv_Y_ri.T)
    mtx_fri2amp_ri = linalg.lstsq(mtx_Tinv_Y_ri, np.eye(mtx_Tinv_Y_ri.shape[0]))[0]
    # -------------------------------------------------
    # the mapping from Diracs amplitude to visibilities (G2)
    mtx_amp2visibility_ri = \
        sph_mtx_amp_ri(colatitude_recon, azimuth_recon, p_mic_x, p_mic_y, p_mic_z)
    # -------------------------------------------------
    # the mapping from FRI sequence to visibilities (G0)
    mtx_fri2visibility_ri = linalg.block_diag(*[np.dot(cpx_mtx2real(mtx_freq2visibility[G_blk_count]),
                                                       mtx_fri2freq_ri)
                                                for G_blk_count in range(num_bands)])
    # -------------------------------------------------
    # projection to the null space of mtx_fri2amp
    # mtx_null_proj = np.eye(num_bands * 2 * (L + 1) ** 2) - \
    #                 np.dot(mtx_fri2amp_ri.T,
    #                        linalg.solve(np.dot(mtx_fri2amp_ri,
    #                                            mtx_fri2amp_ri.T),
    #                                     mtx_fri2amp_ri))
    mtx_null_proj = np.eye(num_bands * 2 * (L + 1) ** 2) - \
                    np.dot(mtx_fri2amp_ri.T,
                           linalg.lstsq(mtx_fri2amp_ri.T,
                                        np.eye(mtx_fri2amp_ri.shape[1]))[0])
    G_updated = np.dot(mtx_amp2visibility_ri, mtx_fri2amp_ri) + \
                np.dot(mtx_fri2visibility_ri, mtx_null_proj)
    return G_updated


def sph_mtx_freq2visibility(L, p_mic_x, p_mic_y, p_mic_z):
    """
    build the linear mapping matrix from the spherical harmonics to the measured visibility.
    :param L: the maximum degree of spherical harmonics
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' y coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' z coordinates
    :return:
    """
    num_mic_per_band, num_bands = p_mic_x.shape
    m_grid, l_grid = np.meshgrid(np.arange(-L, L + 1, step=1, dtype=int),
                                 np.arange(0, L + 1, step=1, dtype=int))
    m_grid = np.reshape(vec_from_low_tri_col_by_col(m_grid), (1, -1), order='F')
    l_grid = np.reshape(vec_from_low_tri_col_by_col(l_grid), (1, -1), order='F')

    # parallel implemnetation (over all the subands at once)
    partial_sph_mtx_inner = partial(sph_mtx_freq2visibility_inner, L=L, m_grid=m_grid, l_grid=l_grid)
    '''
    G_lst = Parallel(n_jobs=cpu_count() - 1)(
        delayed(partial_sph_mtx_inner)(p_mic_x[:, band_loop],
                                       p_mic_y[:, band_loop],
                                       p_mic_z[:, band_loop])
        for band_loop in range(num_bands))
    '''
    G_lst = [partial_sph_mtx_inner(
        p_mic_x[:, band_loop],
        p_mic_y[:, band_loop],
        p_mic_z[:, band_loop]
    ) for band_loop in range(num_bands)]

    # list comprehension (slower)
    # G_lst = [sph_mtx_freq2visibility_inner(p_mic_x[:, band_loop],
    #                                        p_mic_y[:, band_loop],
    #                                        p_mic_z[:, band_loop],
    #                                        L, m_grid, l_grid)
    #          for band_loop in range(num_bands)]
    return G_lst


def sph_mtx_freq2visibility_inner(p_mic_x_loop, p_mic_y_loop, p_mic_z_loop, L, m_grid, l_grid):
    """
    inner loop for the function sph_mtx_freq2visibility
    :param band_loop: the number of the sub-bands considered
    :param p_mic_x_loop: a vector that contains the microphone x-coordinates (multiplied by mid-band frequency)
    :param p_mic_y_loop: a vector that contains the microphone y-coordinates (multiplied by mid-band frequency)
    :param p_mic_z_loop: a vector that contains the microphone z-coordinates (multiplied by mid-band frequency)
    :param L: maximum degree of the spherical harmonics
    :param m_grid: m indices for evaluating the spherical bessel functions and spherical harmonics
    :param l_grid: l indices for evaluating the spherical bessel functions and spherical harmonics
    :return:
    """
    num_mic_per_band = p_mic_x_loop.size
    ells = np.arange(L + 1, dtype=float)[np.newaxis, :]
    G_blk = np.zeros(((num_mic_per_band - 1) * num_mic_per_band, (L + 1) ** 2),
                     dtype=complex, order='C')
    count_G_band = 0
    for q in range(num_mic_per_band):
        p_x_outer = p_mic_x_loop[q]
        p_y_outer = p_mic_y_loop[q]
        p_z_outer = p_mic_z_loop[q]
        for qp in range(num_mic_per_band):
            if not q == qp:
                p_x_qqp = p_x_outer - p_mic_x_loop[qp]
                p_y_qqp = p_y_outer - p_mic_y_loop[qp]
                p_z_qqp = p_z_outer - p_mic_z_loop[qp]
                norm_p_qqp = np.sqrt(p_x_qqp ** 2 + p_y_qqp ** 2 + p_z_qqp ** 2)
                # compute bessel function up to order L
                sph_bessel_ells = (-1j) ** ells * sp.special.jv(ells + 0.5, norm_p_qqp) / np.sqrt(norm_p_qqp)

                colatitude_qqp = np.arccos(p_z_qqp / norm_p_qqp)
                azimuth_qqp = np.arctan2(p_y_qqp, p_x_qqp)
                # here l_grid and m_grid is assumed to be a row vector
                G_blk[count_G_band, :] = sph_bessel_ells[0, l_grid.squeeze()] * \
                                         sph_harm_ufnc(l_grid, m_grid, colatitude_qqp, azimuth_qqp)
                count_G_band += 1
    return G_blk * (2 * np.pi) ** 1.5


def cpx_mtx2real(mtx):
    """
    extend complex valued matrix to an extended matrix of real values only
    :param mtx: input complex valued matrix
    :return:
    """
    return np.vstack((np.hstack((mtx.real, -mtx.imag)), np.hstack((mtx.imag, mtx.real))))


def sph_mtx_fri2visibility_col_major(L, mtx_freq2visibility):
    """
    build the linear transformation matrix that links the FRI sequence, which is
    arranged column by column, with the measured visibilities.
    :param L: maximum degree of spherical harmonics
    :param mtx_freq2visibility: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :return:
    """
    mtx_fri2freq = sph_mtx_fri2freq_col_major(L)
    return linalg.block_diag(*[np.dot(mtx_freq2visibility[G_blk_count], mtx_fri2freq)
                               for G_blk_count in range(len(mtx_freq2visibility))])


def sph_mtx_fri2visibility_col_major_ri(L, mtx_freq2visibility):
    """
    build the linear transformation matrix that links the FRI sequence, which is
    arranged column by column, with the measured visibilities.
    :param L: maximum degree of spherical harmonics
    :param mtx_freq2visibility: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :return:
    """
    mtx_fri2freq = sph_mtx_fri2freq_col_major(L)
    return linalg.block_diag(*[cpx_mtx2real(np.dot(mtx_freq2visibility[G_blk_count],
                                                   mtx_fri2freq))
                               for G_blk_count in range(len(mtx_freq2visibility))])


def sph_mtx_fri2freq_col_major(L):
    """
    build the linear transformation for the case of Diracs on the sphere.
    The spherical harmonics are arranged as a matrix of size:
        (L + 1) x (2L + 1)
    :param L: maximum degree of the available spherical harmonics
    :return:
    """
    G = np.zeros(((L + 1) ** 2, (L + 1) ** 2), dtype=float)
    count = 0
    horizontal_idx = 0
    for m in range(-L, L + 1):
        abs_m = np.abs(m)
        for l in range(abs_m, L + 1):
            Nlm = (-1) ** ((m + abs_m) * 0.5) * \
                  np.sqrt((2. * l + 1.) / (4. * np.pi) *
                          (sp.misc.factorial(l - abs_m) / sp.misc.factorial(l + abs_m))
                          )
            G[count, horizontal_idx:horizontal_idx + l - abs_m + 1] = \
                (-1) ** abs_m * Nlm * compute_Pmn_coef(l, m)
            count += 1
        horizontal_idx += L - abs_m + 1
    return G


def sph_mtx_fri2visibility_row_major(L, mtx_freq2visibility_cpx, aslist=False, symb=False):
    """
    build the linear transformation matrix that links the FRI sequence, which is
    arranged column by column, with the measured visibilities.
    :param L: maximum degree of spherical harmonics
    :param mtx_freq2visibility_cpx: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :param aslist: whether the linear mapping for each subband is returned as a list
                or a block diagonal matrix
    :return:
    """
    # mtx_freq2visibility_cpx
    # matrix with input: the complex-valued spherical harmonics and
    # output: the measured visibilities
    mtx_fri2freq = sph_mtx_fri2freq_row_major(L, symb=symb)
    # -----------------------------------------------------------------
    # cascade the linear mappings
    if aslist:
        return [np.dot(cpx_mtx2real(mtx_freq2visibility_cpx[G_blk_count]),
                       mtx_fri2freq)
                for G_blk_count in range(len(mtx_freq2visibility_cpx))]
    else:
        return linalg.block_diag(*[np.dot(cpx_mtx2real(mtx_freq2visibility_cpx[G_blk_count]),
                                          mtx_fri2freq)
                                   for G_blk_count in range(len(mtx_freq2visibility_cpx))])


def sph_mtx_fri2freq_row_major(L, symb=False):
    # -----------------------------------------------------------------
    # the matrix that connects the rearranged spherical harmonics
    # (complex conjugation and reverse indexing for m < 0) to the normal spherical
    # harmonics.
    # the input to this matrix is a concatenation of the real-part and the imaginary
    # part of the rearranged spherical harmonics.
    # size of this matrix: 2(L+1)^2 x 2(L+1)^2
    sz_non_zero_idx = np.int(L * (L + 1) * 0.5)
    sz_half_ms = np.int((L + 1) * (L + 2) * 0.5)
    # the building matrix for the mapping matrix (negtive m-s)
    mtx_neg_ms_map = np.zeros((sz_non_zero_idx, sz_non_zero_idx), dtype=float)
    idx0 = 0
    for m in range(-L, 0):  # m ranges from -L to -1
        blk_sz = L + m + 1
        mtx_neg_ms_map[idx0:idx0 + blk_sz,
        sz_non_zero_idx - idx0 - blk_sz:sz_non_zero_idx - idx0] = \
            np.eye(blk_sz)
        idx0 += blk_sz
    # the building matrix for the mapping matrix (positive m-s)
    mtx_pos_ms_map = np.eye(sz_half_ms)

    mtx_pn_map_real = np.vstack((np.hstack((np.zeros((sz_non_zero_idx, sz_half_ms)),
                                            mtx_neg_ms_map)),
                                 np.hstack((mtx_pos_ms_map,
                                            np.zeros((sz_half_ms, sz_non_zero_idx))))))
    # CAUTION: the negative sign in the first block
    # the rearranged spherical harmonics with negative m-s are complex CONJUGATED.
    # Hence, these components (which corresponds to the imaginary part of the spherical harmonics)
    # are the negative of the original spherical harmonics.
    mtx_pn_map_imag = np.vstack((np.hstack((np.zeros((sz_non_zero_idx, sz_half_ms)),
                                            -mtx_neg_ms_map)),
                                 np.hstack((mtx_pos_ms_map,
                                            np.zeros((sz_half_ms, sz_non_zero_idx))))))

    mtx_freqRemap2freq = linalg.block_diag(mtx_pn_map_real, mtx_pn_map_imag)
    # -----------------------------------------------------------------
    # the matrix that connects the FRI sequence b_mn to the remapped spherical harmonics
    # size of this matrix: 2(L+1)^2 x 2(L+1)^2
    mtx_fri2freqRemap_pos = sph_buildG_positive_ms(L, L)
    mtx_fri2freqRemap_neg = sph_buildG_strict_negative_ms(L, L)
    mtx_fri2freqRemap_pn_cpx = \
        np.vstack((np.hstack((mtx_fri2freqRemap_pos,
                              np.zeros((sz_half_ms, sz_non_zero_idx)))),
                   np.hstack((np.zeros((sz_non_zero_idx, sz_half_ms)),
                              mtx_fri2freqRemap_neg))
                   ))
    # now extend the matrix with real-valued components only
    mtx_fri2freqRemap_pn_real = np.vstack((np.hstack((np.real(mtx_fri2freqRemap_pn_cpx),
                                                      -np.imag(mtx_fri2freqRemap_pn_cpx))),
                                           np.hstack((np.imag(mtx_fri2freqRemap_pn_cpx),
                                                      np.real(mtx_fri2freqRemap_pn_cpx)))
                                           ))
    if symb:
        expansion_mtx = sph_build_exp_mtx_symb(L)
        return np.dot(mtx_freqRemap2freq,
                      np.dot(mtx_fri2freqRemap_pn_real,
                             linalg.block_diag(expansion_mtx, expansion_mtx)
                             )
                      )
    else:
        return np.dot(mtx_freqRemap2freq, mtx_fri2freqRemap_pn_real)


def sph_build_exp_mtx_symb(L):
    """
    build the expansion matrix such that the data matrix which includes the data for
    spherical harmonics with positive orders (m>=0) and complex conjugated ones with
    NEGATIVE ordres (m<0).
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    sz_id_mtx_pos = np.int((L + 1) * (L + 2) * 0.5)
    sz_id_mtx_neg = np.int((L + 1) * L * 0.5)
    expand_mtx = np.zeros((sz_id_mtx_pos + sz_id_mtx_neg,
                           sz_id_mtx_pos), dtype=int)
    expand_mtx[:sz_id_mtx_pos, :sz_id_mtx_pos] = np.eye(sz_id_mtx_pos, dtype=int)
    expand_mtx[sz_id_mtx_pos:, L + 1:] = np.eye(sz_id_mtx_neg, dtype=int)
    return expand_mtx


def sph_buildG_positive_ms(L, M):
    """
    build the linear transformation for the case of Diracs on the sphere.
    The spherical harmonics are arranged as a matrix of size:
        (L + 1) x (M + 1)
    :param L: maximum degree of the available spherical harmonics
    :param M: maximum order of the available spherical harmonics: 0 to M
    :return:
    """
    assert M <= L
    sz_G = np.int((2 * L + 2 - M) * (M + 1) * 0.5)
    G = np.zeros((sz_G, sz_G), dtype=float)
    count = 0
    horizontal_idx = 0
    for m in range(0, M + 1):
        for l in range(m, L + 1):
            Nlm = (-1) ** m * \
                  np.sqrt((2. * l + 1.) / (4. * np.pi) *
                          (sp.misc.factorial(l - m) / sp.misc.factorial(l + m))
                          )
            G[count, horizontal_idx:horizontal_idx + l - m + 1] = \
                (-1) ** m * Nlm * compute_Pmn_coef(l, m)
            count += 1
        horizontal_idx += L - m + 1
    return G


def sph_buildG_strict_negative_ms(L, M):
    """
    build the linear transformation for the case of Diracs on the sphere.

    The only difference with sph_buildG_positive_ms is the normalisation factor Nlm.

    :param L: maximum degree of the available spherical harmonics
    :param M: maximum order of the available spherical harmonics: -1 to -M
            (NOTICE: the reverse ordering!!)
    :return:
    """
    assert M <= L
    sz_G = np.int((2 * L + 2 - M) * (M + 1) * 0.5 - (L + 1))
    G = np.zeros((sz_G, sz_G), dtype=float)
    count = 0
    horizontal_idx = 0
    for m in range(1, M + 1):
        for l in range(m, L + 1):
            Nlm = np.sqrt((2. * l + 1.) / (4. * np.pi) *
                          (sp.misc.factorial(l - m) / sp.misc.factorial(l + m))
                          )
            G[count, horizontal_idx:horizontal_idx + l - m + 1] = \
                (-1) ** m * Nlm * compute_Pmn_coef(l, m)
            count += 1
        horizontal_idx += L - m + 1
    return G


# =========== utilities for spherical harmonics ============
def sph2cart(r, colatitude, azimuth):
    """
    spherical to cartesian coordinates
    :param r: radius
    :param colatitude: co-latitude
    :param azimuth: azimuth
    :return:
    """
    r_sin_colatitude = r * np.sin(colatitude)
    x = r_sin_colatitude * np.cos(azimuth)
    y = r_sin_colatitude * np.sin(azimuth)
    z = r * np.cos(colatitude)
    return x, y, z


def trans_col2row_pos_ms(L):
    """
    transformation matrix to permute a column by column rearranged vector of an
    upper triangle to a row by row rearranged vector.
    :param L: maximum degree of the spherical harmonics.
    :return:
    """
    sz = np.int((L + 1) * (L + 2) * 0.5)
    perm_idx = vec_from_up_tri_row_by_row_pos_ms(
        up_tri_from_vec_pos_ms(np.arange(0, sz, dtype=int)))
    return np.eye(sz, dtype=int)[perm_idx, :]


def vec_from_up_tri_row_by_row_pos_ms(mtx):
    """
    extract the data from the matrix, which is of the form:

        x x x x
        x x x 0
        x x 0 0
        x 0 0 0

    row by row.

    The function "extract_vec_from_up_tri" extracts data column by column.

    :param mtx: the matrix to extract the data from
    :return: data vector
    """
    L = mtx.shape[0] - 1
    assert mtx.shape[1] == L + 1
    vec_len = np.int((L + 2) * (L + 1) * 0.5)
    vec = np.zeros(vec_len, dtype=mtx.dtype)
    bg_idx = 0
    for l in range(L + 1):
        vec_len_loop = L - l + 1
        vec[bg_idx:bg_idx + vec_len_loop] = mtx[l, :vec_len_loop]
        bg_idx += vec_len_loop
    return vec


def vec_from_low_tri_col_by_col_pos_ms(mtx):
    """
    extract the data from the matrix, which is of the form:

        x 0 0 0
        x x 0 0
        x x x 0
        x x x x

    :param mtx: the matrix to extract the data from
    :return: data vector
    """
    L = mtx.shape[0] - 1
    assert mtx.shape[1] == L + 1
    vec = np.zeros(np.int((L + 1) * (L + 2) * 0.5), dtype=mtx.dtype)
    bg_idx = 0
    for loop in range(L + 1):
        vec_len_loop = L + 1 - loop
        vec[bg_idx:bg_idx + vec_len_loop] = mtx[loop:, loop]
        bg_idx += vec_len_loop
    return vec


def up_tri_from_vec_pos_ms(vec):
    """
    build a matrix of the form

        x x x x
        x x x 0     (*)
        x x 0 0
        x 0 0 0

    from the given data matrix. This function is the inverse operator
    of the function extract_vec_from_up_tri(mtx).
    :param vec: the data vector
    :return: a matrix of the form (*)
    """
    L = np.int((-3 + np.sqrt(1 + 8 * vec.size)) * 0.5)
    assert (L + 1) * (L + 2) == 2 * vec.size
    mtx = np.zeros((L + 1, L + 1), dtype=vec.dtype)
    bg_idx = 0
    for loop in range(0, L + 1):
        vec_len_loop = L + 1 - loop
        mtx[0:vec_len_loop, loop] = vec[bg_idx:bg_idx + vec_len_loop]
        bg_idx += vec_len_loop
    return mtx


def sph_build_exp_mtx(L, option='real'):
    """
    build the expansion matrix such that the data matrix which includes the data for
    spherical harmonics with positive orders (m>=0) and complex conjugated ones with
    strictly NEGATIVE ordres (m<0).
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    sz_id_mtx_pos = np.int((L + 1) * (L + 2) * 0.5)
    sz_id_mtx_neg = np.int((L + 1) * L * 0.5)
    expand_mtx = np.zeros((L + 1 + sz_id_mtx_pos + sz_id_mtx_neg,
                           sz_id_mtx_pos + sz_id_mtx_neg))
    expand_mtx[0:sz_id_mtx_pos, 0:sz_id_mtx_pos] = np.eye(sz_id_mtx_pos)
    if option == 'real':
        expand_mtx[sz_id_mtx_pos:sz_id_mtx_pos + L + 1, 0:L + 1] = np.eye(L + 1)
    else:
        expand_mtx[sz_id_mtx_pos:sz_id_mtx_pos + L + 1, 0:L + 1] = -np.eye(L + 1)
    expand_mtx[sz_id_mtx_pos + L + 1:, sz_id_mtx_pos:] = np.eye(sz_id_mtx_neg)
    return expand_mtx


def up_tri_from_vec(vec):
    """
    build a matrix of the form

        x x x x x x x
        0 x x x x x 0       (*)
        0 0 x x x 0 0
        0 0 0 x 0 0 0

    from the given data matrix. This function is the inverse operator
    of the function vec_from_up_tri_positive_ms(mtx).
    :param vec: the data vector
    :return: a matrix of the form (*)
    """
    L = np.int(np.sqrt(vec.size)) - 1
    assert (L + 1) ** 2 == vec.size
    mtx = np.zeros((L + 1, 2 * L + 1), dtype=vec.dtype)
    bg_idx = 0
    for loop in range(2 * L + 1):
        if loop <= L:
            vec_len_loop = loop + 1
        else:
            vec_len_loop = 2 * L + 1 - loop
        mtx[0:vec_len_loop, loop] = vec[bg_idx:bg_idx + vec_len_loop]
        bg_idx += vec_len_loop
    return mtx


def vec_from_low_tri_col_by_col(mtx):
    """
    extract the data from the matrix, which is of the form:

        0 0 0 x 0 0 0
        0 0 x x x 0 0
        0 x x x x x 0
        x x x x x x x

    :param mtx: the matrix to extract the data from
    :return: data vector
    """
    L = mtx.shape[0] - 1
    assert mtx.shape[1] == 2 * L + 1
    vec = np.zeros((L + 1) ** 2, dtype=mtx.dtype)
    bg_idx = 0
    for loop in range(2 * L + 1):
        if loop <= L:
            vec_len_loop = loop + 1
        else:
            vec_len_loop = 2 * L + 1 - loop
        vec[bg_idx:bg_idx + vec_len_loop] = mtx[-vec_len_loop:, loop]
        bg_idx += vec_len_loop
    return vec


def sph_harm_ufnc(l, m, colatitude, azimuth):
    """
    compute spherical harmonics with the built-in sph_harm function.
    The difference is the name of input angle names as well as the normlisation factor.
    :param l: degree of spherical harmonics
    :param m: order of the spherical harmonics
    :param colatitude: co-latitude
    :param azimuth: azimuth
    :return:
    """
    return (-1.) ** m * sp.special.sph_harm(m, l, azimuth, colatitude)


def compute_Pmn_coef(l, m):
    """
    compute the polynomial coefficients of degree l order m with canonical basis.
    :param l: degree of Legendre polynomial
    :param m: order of Legendre polynomial
    :return:
    """
    abs_m = np.abs(m)
    assert abs_m <= l
    pnm = np.zeros(l - abs_m + 1)
    pn = legendre_poly_canonical_basis_coef(l)
    for n in range(l - abs_m + 1):
        pnm[n] = pn[n + abs_m] * sp.misc.factorial(n + abs_m) / sp.misc.factorial(n)
    return pnm


def legendre_poly_canonical_basis_coef(l):
    """
    compute the legendred polynomial expressed in terms of canonical basis.
    :param l: degree of the legendre polynomial
    :return: p_n's: the legendre polynomial coefficients in ascending degree
    """
    p = np.zeros(l + 1, dtype=float)
    l_half = np.int(np.floor(l * 0.5))
    # l is even
    if l % 2 == 0:
        for n in range(0, l_half + 1):
            p[2 * n] = (-1) ** (l_half - n) * sp.misc.comb(l, l_half - n) * \
                       sp.misc.comb(l + 2 * n, l)
    else:
        for n in range(0, l_half + 1):
            p[2 * n + 1] = (-1) ** (l_half - n) * sp.misc.comb(l, l_half - n) * \
                           sp.misc.comb(l + 2 * n + 1, l)
    p /= (2 ** l)
    return p


def sph_beam_shape(p_mic_x, p_mic_y, p_mic_z, omega_bands, sound_speed,
                   azimuth_plt, colatitude_plt, show_fig=True, save_fig=False,
                   file_name='sph_beam_shape.pdf'):
    """
    Plot the average (over different subband frequencies) beamshape.
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param p_mic_z: a vector that contains microphones' z-coordinates
    :param omega_bands: mid-band (ANGULAR) frequencies [radian/sec]
    :param sound_speed: speed of sound
    :param azimuth_plt: azimuth coordinates of the plot
    :param colatitude_plt: co-latitude coordinates of the plot
    :return:
    """
    x_plt, y_plt, z_plt = sph2cart(1, colatitude_plt, azimuth_plt)
    x_plt = np.reshape(x_plt, (-1, 1), order='F')
    y_plt = np.reshape(y_plt, (-1, 1), order='F')
    z_plt = np.reshape(z_plt, (-1, 1), order='F')

    beam_shape = np.zeros(azimuth_plt.size, dtype=complex)

    norm_factor = sound_speed / omega_bands
    # normalised antenna coordinates in a matrix form
    # each column corresponds to the location in one subband,
    # i.e., the second dimension corresponds to different subbands
    # NOTE: The NEGATIVE sign here converts DOA to the propagation vector
    for loop in range(omega_bands.size):
        norm_factor_loop = norm_factor[loop]
        p_mic_x_normalised = np.reshape(-p_mic_x, (1, -1), order='F') / norm_factor_loop
        p_mic_y_normalised = np.reshape(-p_mic_y, (1, -1), order='F') / norm_factor_loop
        p_mic_z_normalised = np.reshape(-p_mic_z, (1, -1), order='F') / norm_factor_loop

        beam_shape += np.sum(np.exp(-1j * (x_plt * p_mic_x_normalised +
                                           y_plt * p_mic_y_normalised +
                                           z_plt * p_mic_z_normalised)),
                             axis=1)

    beam_shape = np.reshape(beam_shape, azimuth_plt.shape, order='F')

    if show_fig:
        fig = plt.figure(figsize=(6.47, 4), dpi=90)
        ax = fig.add_subplot(111, projection="mollweide")
        azimuth_plt_internal = azimuth_plt.copy()
        azimuth_plt_internal[azimuth_plt > np.pi] -= np.pi * 2
        p_hd = ax.pcolormesh(azimuth_plt_internal, np.pi / 2. - colatitude_plt,
                             np.abs(beam_shape), cmap='Spectral_r',
                             linewidth=0, alpha=0.5,
                             antialiased=True, zorder=0)

        p_hd.set_edgecolor('None')
        p_hdc = fig.colorbar(p_hd, orientation='horizontal', use_gridspec=False,
                             anchor=(0.5, 2.3), shrink=0.75, spacing='proportional')
        # p_hdc.formatter.set_powerlimits((0, 0))
        p_hdc.ax.tick_params(labelsize=8.5)
        p_hdc.update_ticks()

        ax.set_xticklabels([u'210\N{DEGREE SIGN}', u'240\N{DEGREE SIGN}',
                            u'270\N{DEGREE SIGN}', u'300\N{DEGREE SIGN}',
                            u'330\N{DEGREE SIGN}', u'0\N{DEGREE SIGN}',
                            u'30\N{DEGREE SIGN}', u'60\N{DEGREE SIGN}',
                            u'90\N{DEGREE SIGN}', u'120\N{DEGREE SIGN}',
                            u'150\N{DEGREE SIGN}'])

        title_str = r'number of microphones: {0}'
        ax.set_title(title_str.format(repr(p_mic_x.size)), fontsize=11)
        ax.set_xlabel(r'azimuth $\bm{\varphi}$', fontsize=11)
        ax.set_ylabel(r'latitude $90^{\circ}-\bm{\theta}$', fontsize=11)
        ax.xaxis.set_label_coords(0.5, 0.52)
        ax.grid(True)
        if save_fig:
            plt.savefig(file_name, format='pdf', dpi=300, transparent=True)

    return beam_shape
