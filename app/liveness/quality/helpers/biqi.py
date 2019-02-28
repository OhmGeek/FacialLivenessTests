"""
Blind Image Quality Index

Code sample from https://github.com/cvley/ImageQuality/blob/master/python/biqi.py
but converted to Python3 compliant code.

Raises:
    e -- [description]
    e -- [description]

Returns:
    [type] -- [description]
"""


import numpy as np
import pywt
import math
import subprocess
import os
import sys
import traceback

os.chdir(os.path.dirname(os.path.realpath(__file__)) + "/biqi_data")

def gammalist():
    r_gam = []
    gam = []
    for i in range(200, 1001):
        num = i / 1000.0
        gam.append(num)
        gamv = math.gamma(1.0/num)*math.gamma(3.0/num)/math.pow(math.gamma(2.0/num), 2)
        r_gam.append(gamv)

    return r_gam, gam

def wavedec2(arr):
    try:
        print("Enter try")

        if arr.shape[1] == 0:
            raise ValueError('Must be at least 1 value in the array')

        r = pywt.dwt2(arr, 'db9')
        return r[1]
    except Exception as e:
        print("wavedec2 error")

        raise e


def downscale(arr):
    h, w = arr.shape
    newshape = (int(h/2), int(w/2))
    newarr = np.resize(arr, newshape)
    return newarr

def jpg_quality(arr):
    m, n = arr.shape
    if m < 16 or n < 16:
        return -2.0

    arr = arr.astype(np.float)

    d_h = arr[:, 1:] - arr[:, :n-1]
    d_arr = np.zeros((m, int(n/8)-1), dtype=np.float)
    for i in range(1,int(n/8)-1):
        d_arr[:, i] = d_h[:, i*8-1]

    B_h = (np.abs(d_arr)).mean()
    A_h = (8.0*(np.abs(d_h)).mean()-B_h) / 7.0

    sign_h = np.zeros(d_h.shape)
    for i in range(d_h.shape[0]):
        for j in range(d_h.shape[1]):
            if d_h[i,j] > 0:
                sign_h[i, j]= 1
            elif d_h[i,j] == 0:
                sign_h[i,j]= 0
            else:
                sign_h[i,j] =-1

    left_sig = sign_h[:, :n-2]
    right_sig = sign_h[:, 1:]
    Z_h = (left_sig*right_sig<0).mean()

    d_v = arr[1:,:] - arr[:m-1, :]
    d_arr = np.zeros((int(m/8)-1, n), dtype=np.float)
    for i in range(1, int(m/8)-1):
        d_arr[i, :] = d_v[i*8-1,:]

    B_v = (np.abs(d_arr)).mean()
    A_v = (8.0*(np.abs(d_v)).mean()-B_v) / 7.0

    sign_v = np.zeros(d_v.shape)
    for i in range(d_v.shape[0]):
        for j in range(d_v.shape[1]):
            if d_v[i,j]>0:
                sign_v.itemset((i,j), 1)
            elif d_v[i,j]==0:
                sign_v.itemset((i,j), 0)
            else:
                sign_v.itemset((i,j), -1)

    up_sig = sign_v[:m-2, :]
    down_sig = sign_v[1:, :]
    Z_v = (up_sig*down_sig<0).mean()

    B = (B_h + B_v) /2.0
    A = (A_h + A_v) /2.0
    Z = (Z_h + Z_v) /2.0

    alpha = -927.4240
    beta = 850.8986
    gamma1 = 0.02354451
    gamma2 = 0.01287548
    gamma3 = -0.03414790
    score = alpha + beta * (complex(B) ** gamma1) * (complex(A) ** gamma2) * (complex(Z) ** gamma3)

    return score.real


def biqi(arr):


    jpeg_score = jpg_quality(arr)
    gam_list, gam = gammalist()
    mu_horz = []
    mu_vert = []
    mu_diag = []
    sigma_sq_horz = []
    sigma_sq_vert = []
    sigma_sq_diag = []
    gam_horz = []
    gam_vert = []
    gam_diag = []

    h, v, d = wavedec2(arr)
    h_curr = h.flatten('F')
    v_curr = v.flatten('F')
    d_curr = d.flatten('F')

    mu_horz.append(h_curr.mean())
    mu_vert.append(v_curr.mean())
    mu_diag.append(d_curr.mean())

    sigma_sq_horz.append(((h_curr-h_curr.mean())*(h_curr-h_curr.mean())).mean())
    sigma_sq_vert.append(((v_curr-v_curr.mean())*(v_curr-v_curr.mean())).mean())
    sigma_sq_diag.append(((d_curr-d_curr.mean())*(d_curr-d_curr.mean())).mean())

    E_horz = ((h_curr-h_curr.mean())*(h_curr-h_curr.mean())).mean()
    E_vert = ((v_curr-v_curr.mean())*(v_curr-v_curr.mean())).mean()
    E_diag = ((d_curr-d_curr.mean())*(d_curr-d_curr.mean())).mean()

    rho_horz = sigma_sq_horz[0] / math.pow(E_horz, 2)
    rho_vert = sigma_sq_vert[0] / math.pow(E_vert, 2)
    rho_diag = sigma_sq_diag[0] / math.pow(E_diag, 2)

    diff_horz = [(i, gam_list[i]-rho_horz) for i in range(len(gam_list))]
    diff_vert = [(i, gam_list[i]-rho_vert) for i in range(len(gam_list))]
    diff_diag = [(i, gam_list[i]-rho_diag) for i in range(len(gam_list))]

    diff_horz = sorted(diff_horz, key=lambda item: item[1])
    diff_vert = sorted(diff_vert, key=lambda item: item[1])
    diff_diag = sorted(diff_diag, key=lambda item: item[1])
    gam_horz.append(gam[diff_horz[0][0]])
    gam_vert.append(gam[diff_vert[0][0]])
    gam_diag.append(gam[diff_diag[0][0]])

    for i in range(1, 3):
        arr = downscale(arr)
        h, v, d = wavedec2(arr)
        h_curr = h.flatten('F')
        v_curr = v.flatten('F')
        d_curr = d.flatten('F')

        mu_horz.append(h_curr.mean())
        mu_vert.append(v_curr.mean())
        mu_diag.append(d_curr.mean())

        sigma_sq_horz.append(((h_curr-h_curr.mean())*(h_curr-h_curr.mean())).mean())
        sigma_sq_vert.append(((v_curr-v_curr.mean())*(v_curr-v_curr.mean())).mean())
        sigma_sq_diag.append(((d_curr-d_curr.mean())*(d_curr-d_curr.mean())).mean())

        E_horz = ((h_curr-h_curr.mean())*(h_curr-h_curr.mean())).mean()
        E_vert = ((v_curr-v_curr.mean())*(v_curr-v_curr.mean())).mean()
        E_diag = ((d_curr-d_curr.mean())*(d_curr-d_curr.mean())).mean()

        rho_horz = sigma_sq_horz[i] / math.pow(E_horz, 2)
        rho_vert = sigma_sq_vert[i] / math.pow(E_vert, 2)
        rho_diag = sigma_sq_diag[i] / math.pow(E_diag, 2)

        diff_horz = [(j, gam_list[j]-rho_horz) for j in range(len(gam_list))]
        diff_vert = [(j, gam_list[j]-rho_vert) for j in range(len(gam_list))]
        diff_diag = [(j, gam_list[j]-rho_diag) for j in range(len(gam_list))]

        diff_horz = sorted(diff_horz, key=lambda item: item[1])
        diff_vert = sorted(diff_vert, key=lambda item: item[1])
        diff_diag = sorted(diff_diag, key=lambda item: item[1])

        gam_horz.append(gam[diff_horz[0][0]])
        gam_vert.append(gam[diff_vert[0][0]])
        gam_diag.append(gam[diff_diag[0][0]])

    rep_vector = sigma_sq_horz
    rep_vector.extend(sigma_sq_vert)
    rep_vector.extend(sigma_sq_diag)
    rep_vector.extend(gam_horz)
    rep_vector.extend(gam_vert)
    rep_vector.extend(gam_diag)

    with open("test_ind.txt", "w") as f:
        l = "{} ".format(1)
        f.write(l)
        for idx, v in enumerate(rep_vector):
            ll = "{}:{} ".format(idx+1, v)
            f.write(ll)
        f.write("\n")

    os.system("svm-scale -r range2 test_ind.txt >> test_ind_scaled")
    os.system("svm-predict -b 1 test_ind_scaled model_89 output_89")
    os.system("rm -f test_ind_scaled")

    os.system("svm-scale -r range2_jp2k test_ind.txt >> test_ind_scaled")
    os.system("svm-predict -b 1 test_ind_scaled model_89_jp2k output_blur")
    f = open("output_blur", "rb")
    jp2k_score = float(f.readline().strip())
    f.close()
    os.system("rm output_blur test_ind_scaled")


    os.system("svm-scale -r range2_wn test_ind.txt >> test_ind_scaled")
    os.system("svm-predict -b 1 test_ind_scaled model_89_wn output_blur")
    f = open("output_blur", "rb")
    wn_score = float(f.readline().strip())
    f.close()
    os.system("rm output_blur test_ind_scaled")

    os.system("svm-scale -r range2_blur test_ind.txt >> test_ind_scaled")
    os.system("svm-predict -b 1 test_ind_scaled model_89_blur output_blur")
    f = open("output_blur", "rb")
    blur_score = float(f.readline().strip())
    f.close()
    os.system("rm output_blur test_ind_scaled")

    os.system("svm-scale -r range2_ff test_ind.txt >> test_ind_scaled")
    os.system("svm-predict -b 1 test_ind_scaled model_89_ff output_blur")
    f = open("output_blur", "rb")
    ff_score = float(f.readline().strip())
    f.close()
    os.system("rm output_blur test_ind_scaled")

    f = open("output_89", "rb")
    f.readline()
    line = f.readline()
    scores = [jp2k_score, jpeg_score, wn_score, blur_score, ff_score]
    probs = [float(i) for i in line.split()[1:]]
    f.close()
    sumprobs = probs[0] + probs[2] + probs[3] + probs[4]
    newprobs = [probs[0] / sumprobs, probs[2] / sumprobs, probs[3] / sumprobs, probs[4] / sumprobs]
    result = jp2k_score * newprobs[0] + wn_score * newprobs[1] + blur_score * newprobs[2] + ff_score * newprobs[3]
    return result

