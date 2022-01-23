import os
import math
import numpy as np
import time
from PIL import Image

def read_png(path_train_face, path_train_nonface, path_test_face, path_test_nonface, sub_img_size):
    # read trainset faces
    filelist = os.listdir(path_train_face)
    total_num = len(filelist)
    train_face = np.zeros(shape=(total_num, sub_img_size))
    for i in range(total_num):
        png_name = path_train_face + '\\' + filelist[i]
        image = np.array(Image.open(png_name))
        image = image.reshape((1,sub_img_size))
        train_face[i] = image

    # read trainset nonfaces
    filelist = os.listdir(path_train_nonface)
    total_num = len(filelist)
    train_nonface = np.zeros(shape=(total_num, sub_img_size))
    for i in range(total_num):
        png_name = path_train_nonface + '\\' + filelist[i]
        image = np.array(Image.open(png_name))
        image = image.reshape((1,sub_img_size))
        train_nonface[i] = image

    # read testset faces
    filelist = os.listdir(path_test_face)
    total_num = len(filelist)
    test_face = np.zeros(shape=(total_num, sub_img_size))
    for i in range(total_num):
        png_name = path_test_face + '\\' + filelist[i]
        image = np.array(Image.open(png_name))
        image = image.reshape((1, sub_img_size))
        test_face[i] = image

    # read testset nonfaces
    filelist = os.listdir(path_test_nonface)
    total_num = len(filelist)
    test_nonface = np.zeros(shape=(total_num, sub_img_size))
    for i in range(total_num):
        png_name = path_test_nonface + '\\' + filelist[i]
        image = np.array(Image.open(png_name))
        image = image.reshape((1, sub_img_size))
        test_nonface[i] = image
    return train_face, train_nonface, test_face, test_nonface

def integral_image(pic_set, sub_img_row, sub_img_col):
    s_2D = np.zeros(shape=(sub_img_row+1, sub_img_col+1))
    ii_2D = np.zeros(shape=(sub_img_row+1, sub_img_col+1))
    ii = np.zeros(shape=(pic_set.shape[0], (sub_img_row+1) * (sub_img_col+1)))
    for x in range(pic_set.shape[0]):
        i_2D = pic_set[x].reshape((sub_img_row, sub_img_col))
        for col in range(sub_img_col):
            for row in range(sub_img_row):
                s_2D[row+1,col+1] = s_2D[row,col+1] + i_2D[row,col]
                ii_2D[row+1,col+1] = ii_2D[row+1,col] + s_2D[row+1,col+1]
        ii[x] = ii_2D.reshape((1,(sub_img_row+1)*(sub_img_col+1)))
    return ii

# def Num_Harr_Feat(sub_img_row, sub_img_col):
#     ct = 0
#     # decide the Harr feature A
#     list_A = []
#     for row1 in range(14):
#         for row2 in range(row1, 15):
#             for col1 in range(5,14):
#                 for col2 in range(col1 + 1, 15, 2):
#                     list_A.append((row1, col1, row2, col2))
#                     ct = ct + 1
#     num_A = ct
#     # decide the Harr feature B
#     list_B = []
#     for row1 in range(14):
#         for row2 in range(row1 + 1, 15, 2):
#             for col1 in range(5,15):
#                 for col2 in range(col1, 15):
#                     list_B.append((row1, col1, row2, col2))
#                     ct = ct + 1
#     num_B = ct - num_A
#     # decide the Harr feature C
#     list_C = []
#     for row1 in range(14):
#         for row2 in range(row1, 15):
#             for col1 in range(5-1,15 - 1):
#                 for col2 in range(col1 + 2, 15, 3):
#                     list_C.append((row1, col1, row2, col2))
#                     ct = ct + 1
#     num_C = ct - num_A - num_B
#     # decide the Harr feature C1
#     list_C1 = []
#     for row1 in range(15-1):
#         for row2 in range(row1+2, 15, 3):
#             for col1 in range(5,15):
#                 for col2 in range(col1, 15):
#                     list_C.append((row1, col1, row2, col2))
#                     ct = ct + 1
#     num_C1 = ct - num_A - num_B -num_C
#     # decide the Harr feature D
#     list_D = []
#     for row1 in range(15):
#         for row2 in range(row1 + 1, 15, 2):
#             for col1 in range(5,15):
#                 for col2 in range(col1 + 1, 15, 2):
#                     list_D.append((row1, col1, row2, col2))
#                     ct = ct + 1
#     num_D = ct - num_A - num_B - num_C - num_C1
#     return ct, list_A, list_B, list_C, list_C1, list_D, num_A, num_B, num_C,num_C1, num_D

def Num_Harr_Feat(sub_img_row, sub_img_col):
    ct = 0
    # decide the Harr feature A
    list_A = []
    for row1 in range(sub_img_row):
        for row2 in range(row1, sub_img_row):
            for col1 in range(sub_img_col):
                for col2 in range(col1 + 1, sub_img_col, 2):
                    list_A.append((row1, col1, row2, col2))
                    ct = ct + 1
    num_A = ct
    # decide the Harr feature B
    list_B = []
    for row1 in range(sub_img_row):
        for row2 in range(row1 + 1, sub_img_row, 2):
            for col1 in range(sub_img_col):
                for col2 in range(col1, sub_img_col):
                    list_B.append((row1, col1, row2, col2))
                    ct = ct + 1
    num_B = ct - num_A
    # decide the Harr feature C
    list_C = []
    for row1 in range(sub_img_row):
        for row2 in range(row1, sub_img_row ):
            for col1 in range(sub_img_col - 1):
                for col2 in range(col1 + 2, sub_img_col, 3):
                    list_C.append((row1, col1, row2, col2))
                    ct = ct + 1
    num_C = ct - num_A - num_B
    # decide the Harr feature C1
    list_C1 = []
    for row1 in range(sub_img_row-1):
        for row2 in range(row1+2, sub_img_row, 3):
            for col1 in range(sub_img_col):
                for col2 in range(col1, sub_img_col):
                    list_C.append((row1, col1, row2, col2))
                    ct = ct + 1
    num_C1 = ct - num_A - num_B -num_C
    # decide the Harr feature D
    list_D = []
    for row1 in range(sub_img_row):
        for row2 in range(row1 + 1, sub_img_row, 2):
            for col1 in range(sub_img_col):
                for col2 in range(col1 + 1, sub_img_col, 2):
                    list_D.append((row1, col1, row2, col2))
                    ct = ct + 1
    num_D = ct - num_A - num_B - num_C - num_C1
    return ct, list_A, list_B, list_C, list_C1, list_D, num_A, num_B, num_C,num_C1, num_D

def Compute_Harr_Feat_A(row1, col1, row2, col2, ii):
    white = ii[row1 - 1, col1 - 1] + ii[row2, int((col1 + col2) / 2)] - (ii[row2, col1 - 1] + ii[row1 - 1, int((col1 + col2) / 2)])
    black = ii[row1 - 1, int((col1 + col2) / 2)] + ii[row2, col2] - (ii[row2, int((col1 + col2) / 2)] + ii[row1 - 1, col2])
    return white - black

def Compute_Harr_Feat_B(row1, col1, row2, col2, ii):
    white = ii[int((row1 + row2) / 2), col1 - 1] + ii[row2, col2] - (ii[int((row1 + row2) / 2), col2] + ii[row2, col1 - 1])
    black = ii[row1 - 1, col1 - 1] + ii[int((row1 + row2) / 2), col2] - (ii[int((row1 + row2) / 2), col1 - 1] + ii[row1 - 1, col2])
    return white - black

def Compute_Harr_Feat_C(row1, col1, row2, col2, ii):
    white = ii[row1 - 1, col1 - 1] + ii[row2, int((col2 - col1 + 1) / 3) + col1 - 1] + ii[row1 - 1, int(2 * (col2 - col1 + 1) / 3) + col1 - 1] + ii[row2, col2] \
            - (ii[row2, col1 - 1] + ii[row1 - 1, int((col2 - col1 + 1) / 3) + col1 - 1] + ii[row2, int(2 * (col2 - col1 + 1) / 3) + col1 - 1] + ii[row1 - 1, col2])
    black = ii[row1 - 1, int((col2 - col1 + 1) / 3) + col1 - 1] + ii[row2, int(2 * (col2 - col1 + 1) / 3) + col1 - 1] \
            - (ii[row2, int((col2 - col1 + 1) / 3) + col1 - 1] + ii[row1-1, int(2 * (col2 - col1 + 1) / 3) + col1 - 1])
    return white - black

def Compute_Harr_Feat_C1(row1, col1, row2, col2, ii):
    white = ii[row1 - 1, col1 - 1] + ii[int((row2 - row1 + 1) / 3) + row1 - 1, col2] + ii[int(2 * (row2 - row1 + 1) / 3) + row1 - 1, col1 - 1] + ii[row2, col2] \
            - (ii[row1 - 1, col2] + ii[int((row2 - row1 + 1) / 3) + row1 - 1, col1 - 1 ] + ii[int(2 * (row2 - row1 + 1) / 3) + row1 - 1, col2] + ii[row2, col1 - 1])
    black = ii[int((row2 - row1 + 1) / 3) + row1 - 1, col1 - 1] + ii[int(2 * (row2 - row1 + 1) / 3) + row1 - 1, col2] \
            - (ii[int((row2 - row1 + 1) / 3) + row1 - 1, col2] + ii[int(2 * (row2 - row1 + 1) / 3) + row1 - 1, col1-1])
    return white - black

def Compute_Harr_Feat_D(row1, col1, row2, col2, ii):
    white = ii[row1 - 1, int((col1 + col2) / 2)] + ii[int((row1 + row2) / 2), col2] + ii[int((row1 + row2) / 2), col1 - 1] + ii[row2, int((col1 + col2) / 2)] \
            - (ii[row2, col1 - 1] + 2 * ii[int((row1 + row2) / 2), int((col1 + col2) / 2)] + ii[row1 - 1, col2])
    black = ii[row1 - 1, col1 - 1] + 2 * ii[int((row1 + row2) / 2), int((col1 + col2) / 2)] + ii[row2, col2] \
            - (ii[int((row1 + row2) / 2), col1 - 1] + ii[row1 - 1, int((col1 + col2) / 2)] + ii[row2, int((col1 + col2) / 2)] + ii[int((row1 + row2) / 2), col2])
    return white - black


######################
# Input
######################
sub_img_row = 19
sub_img_col = 19
sub_img_size = sub_img_row * sub_img_col
T = 10      # total round

###########################################
# Read in the training set and testing set
###########################################
time_start=time.time()
train_face, train_nonface, test_face, test_nonface \
= read_png('C:\\Users\\dong\\Desktop\\New folder (2)\\trainset\\faces', \
           'C:\\Users\\dong\\Desktop\\New folder (2)\\trainset\\nonfaces', \
           'C:\\Users\\dong\\Desktop\\New folder (2)\\testset\\faces', \
           'C:\\Users\\dong\\Desktop\\New folder (2)\\testset\\non-faces',\
            sub_img_size)

# train_face, train_nonface, test_face, test_nonface \
# = read_png('C:\\Users\\dong\\Desktop\\ecen 649\\hw\\hw5\\trainset\\faces', \
#            'C:\\Users\\dong\\Desktop\\ecen 649\\hw\\hw5\\trainset\\non-faces', \
#            'C:\\Users\\dong\\Desktop\\ecen 649\\hw\\hw5\\testset\\faces', \
#            'C:\\Users\\dong\\Desktop\\ecen 649\\hw\\hw5\\testset\\non-faces',\
#             sub_img_size)
time_end=time.time()
print('time read',time_end-time_start,'s')

################
# Initialization
################
time_start=time.time()
# w is the distribution array
w = np.zeros(shape=(train_face.shape[0]+train_nonface.shape[0], 1))
w[:train_face.shape[0]] = 0.5/train_face.shape[0]
w[train_face.shape[0]:] = 0.5/train_nonface.shape[0]

# y is the label col array
y = np.zeros(shape=(train_face.shape[0]+train_nonface.shape[0], 1))
y[:train_face.shape[0]] = 1
y[train_face.shape[0]:] = -1

y_test = np.zeros(shape=(test_face.shape[0]+test_nonface.shape[0]))
y_test[:test_face.shape[0]] = 1
y_test[test_face.shape[0]:] = -1

# ii_train is the rectangle value 2D np array
ii_train_face = integral_image(train_face, sub_img_row, sub_img_col)
ii_train_nonface = integral_image(train_nonface, sub_img_row, sub_img_col)
ii_train = np.vstack((ii_train_face,ii_train_nonface))

ii_test_face = integral_image(test_face, sub_img_row, sub_img_col)
ii_test_nonface = integral_image(test_nonface, sub_img_row, sub_img_col)
ii_test = np.vstack((ii_test_face, ii_test_nonface))

# d is the number of total Harr features. list_A - D are the arrays containing the left top and
# -right bottom coordinates of the Harr features
d, list_A, list_B, list_C, list_C1, list_D, num_A, num_B, num_C, num_C1, num_D= Num_Harr_Feat(sub_img_row, sub_img_col)
j_star_list = np.array([])
theta_star_list = np.array([])
alpha_list = np.array([])

time_end=time.time()
print('time initialization',time_end-time_start,'s')

#######################
# Main program starts
#######################
time_start=time.time()
for t in range(T):
    # Normalize w
    w = w/np.sum(w)
    theta_star = 0
    j_star = 0
    F_star = 10 ** 10
    # Find out which Harr feature to choose in the tth round and also the threshold
    for j in range(d):    # d is num of Harr features
        x = np.zeros(ii_train.shape[0])
        if j < len(list_A):   # compute Harr feature A
            row1 = list_A[j][0] + 1
            col1 = list_A[j][1] + 1
            row2 = list_A[j][2] + 1
            col2 = list_A[j][3] + 1
            for i in range(ii_train.shape[0]):
                ii = ii_train[i].reshape((sub_img_row+1, sub_img_col+1))
                x[i] = Compute_Harr_Feat_A(row1, col1, row2, col2, ii)
        elif j < len(list_A) + len(list_B):    # compute Harr feature B
            k = j - len(list_A)
            row1 = list_B[k][0] + 1
            col1 = list_B[k][1] + 1
            row2 = list_B[k][2] + 1
            col2 = list_B[k][3] + 1
            for i in range(ii_train.shape[0]):
                ii = ii_train[i].reshape((sub_img_row+1, sub_img_col+1))
                x[i] = Compute_Harr_Feat_B(row1, col1, row2, col2, ii)
        elif j < len(list_A) + len(list_B) + len(list_C): # compute Harr feature C
            k = j - len(list_A) - len(list_B)
            row1 = list_C[k][0] + 1
            col1 = list_C[k][1] + 1
            row2 = list_C[k][2] + 1
            col2 = list_C[k][3] + 1
            for i in range(ii_train.shape[0]):
                ii = ii_train[i].reshape((sub_img_row+1, sub_img_col+1))
                x[i] = Compute_Harr_Feat_C(row1, col1, row2, col2, ii)
        elif j < len(list_A) + len(list_B) + len(list_C) + len(list_C1):
            k = j - len(list_A) - len(list_B) - len(list_C)
            row1 = list_C1[k][0] + 1
            col1 = list_C1[k][1] + 1
            row2 = list_C1[k][2] + 1
            col2 = list_C1[k][3] + 1
            for i in range(ii_train.shape[0]):
                ii = ii_train[i].reshape((sub_img_row+1, sub_img_col+1))
                x[i] = Compute_Harr_Feat_C1(row1, col1, row2, col2, ii)
        else: # compute Harr feature D
            k = j - len(list_A) - len(list_B) - len(list_C) - len(list_C1)
            row1 = list_D[k][0] + 1
            col1 = list_D[k][1] + 1
            row2 = list_D[k][2] + 1
            col2 = list_D[k][3] + 1
            for i in range(ii_train.shape[0]):
                ii = ii_train[i].reshape((sub_img_row+1, sub_img_col+1))
                x[i] = Compute_Harr_Feat_D(row1, col1, row2, col2, ii)
        index_x = np.argsort(x)
        x = np.sort(x)
        x = np.append(x, x[-1] + 1)
        y_temp = y[index_x]
        w_temp = w[index_x]
        y_eq_one = np.where(y_temp == 1)[0]
        F = np.sum(w_temp[y_eq_one])
        if F < F_star:
            F_star = F
            theta_star = x[0] - 1
            j_star = j
        for i in range(ii_train.shape[0]):
            F = F - y_temp[i] * w_temp[i]
            if (F < F_star) and (x[i] != x[i+1]):
                F_star = F
                theta_star = 0.5*(x[i] + x[i+1])
                j_star = j

    j_star_list = np.append(j_star_list, j_star)
    theta_star_list = np.append(theta_star_list, theta_star)
    # Compute the weak learner
    x_temp1 = np.zeros(ii_train.shape[0])
    if j_star < len(list_A):  # compute Harr feature A
        row1 = list_A[j_star][0] + 1
        col1 = list_A[j_star][1] + 1
        row2 = list_A[j_star][2] + 1
        col2 = list_A[j_star][3] + 1
        for i in range(ii_train.shape[0]):
            ii = ii_train[i].reshape((sub_img_row + 1, sub_img_col + 1))
            x_temp1[i] = Compute_Harr_Feat_A(row1, col1, row2, col2, ii)
    elif j_star < len(list_A) + len(list_B):  # compute Harr feature B
        k = j_star - len(list_A)
        row1 = list_B[k][0] + 1
        col1 = list_B[k][1] + 1
        row2 = list_B[k][2] + 1
        col2 = list_B[k][3] + 1
        for i in range(ii_train.shape[0]):
            ii = ii_train[i].reshape((sub_img_row + 1, sub_img_col + 1))
            x_temp1[i] = Compute_Harr_Feat_B(row1, col1, row2, col2, ii)
    elif j_star < len(list_A) + len(list_B) + len(list_C):  # compute Harr feature C
        k = j_star - len(list_A) - len(list_B)
        row1 = list_C[k][0] + 1
        col1 = list_C[k][1] + 1
        row2 = list_C[k][2] + 1
        col2 = list_C[k][3] + 1
        for i in range(ii_train.shape[0]):
            ii = ii_train[i].reshape((sub_img_row + 1, sub_img_col + 1))
            x_temp1[i] = Compute_Harr_Feat_C(row1, col1, row2, col2, ii)
    elif j_star < len(list_A) + len(list_B) + len(list_C) + len(list_C1): # compute Harr feature C1
        k = j_star - len(list_A) - len(list_B) - len(list_C)
        row1 = list_C1[k][0] + 1
        col1 = list_C1[k][1] + 1
        row2 = list_C1[k][2] + 1
        col2 = list_C1[k][3] + 1
        for i in range(ii_train.shape[0]):
            ii = ii_train[i].reshape((sub_img_row + 1, sub_img_col + 1))
            x_temp1[i] = Compute_Harr_Feat_C1(row1, col1, row2, col2, ii)
    else:  # compute Harr feature D
        k = j_star - len(list_A) - len(list_B) - len(list_C) - len(list_C1)
        row1 = list_D[k][0] + 1
        col1 = list_D[k][1] + 1
        row2 = list_D[k][2] + 1
        col2 = list_D[k][3] + 1
        for i in range(ii_train.shape[0]):
            ii = ii_train[i].reshape((sub_img_row + 1, sub_img_col + 1))
            x_temp1[i] = Compute_Harr_Feat_D(row1, col1, row2, col2, ii)
    h_array = np.sign(theta_star - x_temp1)
    h_array = h_array.reshape(len(h_array),1)

    # e = h_array != y
    # err = F_star
    # beta = err/(1-err)
    # alpha_list = np.append(alpha_list, math.log(1/beta))
    # w = w * (beta**(1-e))

    err = np.sum(w * np.absolute(h_array - y)/2)
    if err == 0:
        err = 10**(-10)
    alpha_list = np.append(alpha_list, 0.5*math.log(1/err - 1))
    w = w * np.exp(-alpha_list[-1]*(y == h_array))/np.sum(w*np.exp(-alpha_list[-1]*(y == h_array)))


print('j_star_list',j_star_list)
print('theta_star_list',theta_star_list)
print('alpha_list',alpha_list)
time_end = time.time()
print('time main', time_end - time_start, 's')


###############
# Test
###############
face_or_not = np.zeros(shape=(T, ii_test.shape[0]))
correct_or_not = np.zeros(shape=(T, ii_test.shape[0]))
correct_num = np.zeros(shape=(T))
accuracy = np.zeros(shape=(T))
false_negative = np.zeros(shape=(T)) # A result that appears negative when it should not +1 -> -1
false_positive = np.zeros(shape=(T)) # negative -> positive -1 -> +1
for i in range(ii_test.shape[0]):
    ii = ii_test[i].reshape((sub_img_row + 1, sub_img_col + 1))
    ht = np.array([])
    for t in range(T):
        theta = theta_star_list[t]
        alpha = alpha_list[t]
        if j_star_list[t] < len(list_A):
            row1 = list_A[int(j_star_list[t])][0] + 1
            col1 = list_A[int(j_star_list[t])][1] + 1
            row2 = list_A[int(j_star_list[t])][2] + 1
            col2 = list_A[int(j_star_list[t])][3] + 1
            Harr_test = Compute_Harr_Feat_A(row1, col1, row2, col2, ii)
        elif j_star_list[t] < len(list_A) + len(list_B):
            k = int(j_star_list[t]) - len(list_A)
            row1 = list_B[k][0] + 1
            col1 = list_B[k][1] + 1
            row2 = list_B[k][2] + 1
            col2 = list_B[k][3] + 1
            Harr_test = Compute_Harr_Feat_B(row1, col1, row2, col2, ii)
        elif j_star_list[t] < len(list_A) + len(list_B) + len(list_C):
            k = int(j_star_list[t]) - len(list_A) - len(list_B)
            row1 = list_C[k][0] + 1
            col1 = list_C[k][1] + 1
            row2 = list_C[k][2] + 1
            col2 = list_C[k][3] + 1
            Harr_test = Compute_Harr_Feat_C(row1, col1, row2, col2, ii)
        elif j_star_list[t] < len(list_A) + len(list_B) + len(list_C) + len(list_C1):
            k = int(j_star_list[t]) - len(list_A) - len(list_B) - len(list_C)
            row1 = list_C1[k][0] + 1
            col1 = list_C1[k][1] + 1
            row2 = list_C1[k][2] + 1
            col2 = list_C1[k][3] + 1
            Harr_test = Compute_Harr_Feat_C1(row1, col1, row2, col2, ii)
        else:
            k = int(j_star_list[t]) - len(list_A) - len(list_B) - len(list_C) - len(list_C1)
            row1 = list_D[k][0] + 1
            col1 = list_D[k][1] + 1
            row2 = list_D[k][2] + 1
            col2 = list_D[k][3] + 1
            Harr_test = Compute_Harr_Feat_D(row1, col1, row2, col2, ii)
        ht = np.append(ht, np.sign(theta - Harr_test))
    for t in range(T):
        if np.sum(alpha_list[:t+1]*ht[:t+1]) >= 0: #0.5 * np.sum(alpha_list[:t+1]):
            face_or_not[t][i] = 1
        else:
            face_or_not[t][i] = -1
for t in range(T):
    correct_or_not[t] = face_or_not[t] == y_test
    correct_num[t] = np.sum(correct_or_not[t] == 1)
    accuracy[t] = np.sum(correct_or_not[t] == 1)/len(correct_or_not[t])

for t in range(T):
    fn = 0
    fp = 0
    for i in range(ii_test.shape[0]):
        if face_or_not[t][i] == 1:
            if y_test[i] == -1:
                fp = fp + 1
        else:
            if y_test[i] == 1:
                fn = fn + 1
    false_negative[t] = fn
    false_positive[t] = fp
recognition_rate = (ii_test_face.shape[0] - false_negative)/ii_test_face.shape[0]

print('false_negative',false_negative)
print('false_positive',false_positive)
print('correct_num',correct_num)
print('accuracy',accuracy)
print('recognition_rate',recognition_rate)

#####################
# Print location
#####################
for j in j_star_list:
    if j < len(list_A):
        print('A: ', list_A[int(j)])
    elif j < len(list_A) + len(list_B):
        k = j - len(list_A)
        print('B: ', list_B[int(k)])
    elif j < len(list_A) + len(list_B) + len(list_C):
        k = j - len(list_A) - len(list_B)
        print('C: ', list_C[int(k)])
    elif j < len(list_A) + len(list_B) + len(list_C) + len(list_C1):
        k = j - len(list_A) - len(list_B) - len(list_C)
        print('C1: ', list_C1[int(k)])
    else:
        k = j - len(list_A) - len(list_B) - len(list_C) - len(list_C1)
        print('D: ', list_D[int(k)])