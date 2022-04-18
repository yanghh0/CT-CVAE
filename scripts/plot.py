import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

ground_true_length0 = [0, 1, 133, 192, 223, 340, 412, 454, 450, 424, 424, 406, 348, 303, 280, 268, 255, 209, 193, 195,
                       133, 144, 109, 109, 96, 94, 59, 90, 74, 53, 56, 49, 45, 41, 36, 38, 26, 25, 23, 20, 16, 19, 26,
                       12, 14, 11, 12, 7, 12, 15]
ground_true_length1 = [0, 3, 137, 203, 246, 392, 490, 553, 567, 552, 570, 588, 535, 474, 487, 470, 479, 431, 389, 375,
                       314, 327, 267, 275, 235, 241, 173, 184, 172, 149, 147, 117, 110, 89, 75, 86, 63, 60, 45, 42, 36,
                       35, 40, 27, 23, 21, 23, 13, 19, 22]
ground_true_length2 = [0, 15, 164, 261, 350, 625, 777, 869, 879, 905, 908, 908, 858, 807, 800, 778, 733, 661, 594, 557,
                       466, 471, 377, 389, 322, 322, 238, 261, 219, 198, 179, 149, 134, 121, 98, 107, 76, 80, 56, 55,
                       44, 40, 46, 32, 25, 27, 27, 16, 21, 28]
ground_true_length3 = [0, 15, 164, 262, 360, 781, 1142, 1311, 1465, 1468, 1508, 1543, 1545, 1452, 1453, 1403, 1260,
                       1127, 977, 799, 594, 532, 390, 398, 325, 323, 238, 261, 219, 198, 179, 149, 134, 121, 98, 107,
                       76, 80, 56, 55, 44, 40, 46, 32, 25, 27, 27, 16, 21, 28]

gpt2_length0 = [0, 253, 1281, 1527, 1505, 861, 608, 456, 227, 125, 75, 57, 35, 17, 13, 9, 4, 2, 3, 2, 2, 0, 4, 0, 2, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
gpt2_length1 = [0, 259, 1307, 1625, 1638, 1082, 893, 849, 605, 482, 471, 454, 402, 313, 234, 207, 155, 117, 139, 79, 45,
                37, 39, 29, 19, 6, 7, 3, 3, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
gpt2_length2 = [0, 276, 1537, 1862, 2539, 2039, 1903, 1691, 1021, 790, 667, 683, 607, 395, 282, 231, 169, 123, 147, 80,
                47, 41, 39, 29, 19, 6, 7, 4, 3, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
gpt2_length3 = [0, 276, 1538, 1876, 2726, 3675, 3134, 3883, 1807, 1450, 1142, 972, 803, 475, 326, 237, 169, 125, 148,
                80, 47, 42, 39, 29, 19, 6, 7, 4, 3, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

blender90M_length0 = [0, 0, 288, 755, 453, 1846, 952, 945, 747, 350, 203, 150, 111, 82, 43, 45, 23, 18, 14, 10, 8, 3, 4,
                      4, 4, 2, 1, 0, 3, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
blender90M_length1 = [0, 3, 293, 781, 508, 1949, 1116, 1192, 1049, 644, 409, 375, 386, 391, 364, 290, 328, 278, 253,
                      193, 147, 115, 97, 87, 67, 52, 40, 31, 26, 16, 5, 6, 7, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0]
blender90M_length2 = [0, 10, 332, 946, 685, 2679, 1615, 1972, 2078, 978, 727, 627, 710, 650, 535, 531, 544, 365, 290,
                      221, 155, 126, 107, 91, 71, 52, 40, 33, 27, 17, 8, 6, 7, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0]
blender90M_length3 = [0, 11, 335, 956, 742, 2927, 2806, 2778, 3707, 1572, 1675, 1215, 1187, 1144, 798, 748, 685, 435,
                      317, 236, 159, 129, 113, 95, 73, 54, 40, 34, 27, 17, 8, 6, 7, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0]

transformer_length0 = [0, 0, 156, 690, 346, 1301, 1360, 1111, 676, 477, 185, 173, 114, 139, 97, 73, 44, 31, 25, 18, 18,
                       11, 6, 2, 3, 5, 3, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
transformer_length1 = [0, 1, 160, 701, 381, 1368, 1506, 1355, 996, 778, 418, 382, 463, 588, 449, 244, 282, 245, 219,
                       172, 144, 140, 98, 98, 87, 113, 36, 23, 23, 11, 6, 3, 8, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0]
transformer_length2 = [0, 1, 173, 780, 519, 2125, 2003, 1986, 2640, 936, 688, 661, 812, 816, 587, 458, 453, 313, 259,
                       193, 160, 147, 105, 102, 88, 117, 36, 23, 24, 11, 6, 3, 10, 1, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0]
transformer_length3 = [0, 1, 173, 783, 539, 2283, 3319, 2914, 4384, 1487, 1569, 1171, 1119, 1281, 909, 722, 631, 379,
                       307, 214, 171, 151, 107, 103, 88, 117, 37, 23, 24, 11, 6, 3, 10, 1, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0]

cvae_length0 = [0, 0, 377, 462, 446, 939, 747, 706, 634, 484, 374, 313, 257, 226, 199, 169, 150, 117, 97, 72, 46, 41, 43,
                24, 20, 21, 17, 12, 14, 16, 6, 8, 6, 2, 4, 5, 1, 3, 0, 0, 2, 3, 3, 0, 1, 0, 1, 0, 0, 0]
cvae_length1 = [0, 4, 10, 14, 51, 113, 209, 244, 292, 274, 277, 265, 277, 278, 298, 302, 240, 214, 182, 149, 151, 123, 85,
                85, 49, 52, 36, 37, 22, 20, 18, 20, 11, 9, 5, 5, 4, 3, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
cvae_length2 = [0, 23, 86, 157, 305, 580, 558, 614, 608, 443, 379, 334, 339, 267, 245, 189, 149, 111, 83, 72, 37, 35, 31,
                19, 14, 10, 7, 11, 5, 3, 6, 1, 3, 1, 3, 1, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
cvae_length3 = [0, 0, 0, 5, 68, 287, 696, 887, 985, 896, 917, 768, 632, 547, 391, 284, 185, 138, 61, 25, 17, 8, 1, 0, 2, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

CTCVAE_length0 = [0, 0, 0, 0, 1, 2, 22, 132, 251, 391, 402, 354, 331, 271, 281, 354, 426, 442, 343, 307, 286, 326, 333, 334,
                  306, 274, 181, 151, 124, 121, 95, 73, 47, 43, 28, 13, 13, 8, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CTCVAE_length1 = [0, 0, 0, 0, 0, 0, 9, 46, 114, 159, 165, 161, 189, 180, 197, 246, 303, 285, 259, 209, 217, 207, 227, 248,
                  224, 206, 140, 85, 83, 70, 56, 55, 42, 22, 15, 9, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CTCVAE_length2 = [0, 0, 0, 0, 0, 2, 34, 170, 254, 236, 225, 246, 231, 290, 342, 421, 385, 340, 259, 268, 275, 307, 313, 252,
                  203, 166, 128, 80, 93, 68, 62, 41, 26, 8, 7, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CTCVAE_length3 = [0, 0, 0, 0, 0, 5, 55, 300, 682, 735, 769, 621, 491, 497, 529, 631, 617, 469, 427, 286, 232, 177, 133, 82,
                  31, 22, 6, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

x = list(range(len(gpt2_length0)))

folderName = "F:\\newest"

name = "xxx"


def plot_curve(x_data,
               gpt2_length0,
               gpt2_length1,
               gpt2_length2,
               gpt2_length3,
               ground_true_length0,
               ground_true_length1,
               ground_true_length2,
               ground_true_length3,
               blender90M_length0,
               blender90M_length1,
               blender90M_length2,
               blender90M_length3,
               transformer_length0,
               transformer_length1,
               transformer_length2,
               transformer_length3,
               cvae_length0,
               cvae_length1,
               cvae_length2,
               cvae_length3,
               CTCVAE_length0,
               CTCVAE_length1,
               CTCVAE_length2,
               CTCVAE_length3):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.xticks(rotation=55)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.rcParams['axes.unicode_minus'] = False
    f, ax = plt.subplots(2, 2)
    plt.subplots_adjust(bottom=0., left=0, top=1., right=1, hspace=0.3)

    ax[0][0].plot(x_data, ground_true_length0, label='GroundTruth', color='green')
    ax[0][0].plot(x_data, gpt2_length0, label='DialoGPT', color='b')
    ax[0][0].plot(x_data, blender90M_length0, label='Blender90M', color='orange')
    ax[0][0].plot(x_data, CTCVAE_length0, label='CT-CVAE', color='purple')
    ax[0][0].plot(x_data, cvae_length0, label='w/o c', color='red')
    ax[0][0].set_title('Dailydialog')
    ax[0][0].legend()

    ax[0][1].plot(x_data, ground_true_length1, label='GroundTruth', color='green')
    ax[0][1].plot(x_data, gpt2_length1, label='DialoGPT', color='b')
    ax[0][1].plot(x_data, blender90M_length1, label='Blender90M', color='orange')
    ax[0][1].plot(x_data, CTCVAE_length1, label='CT-CVAE', color='purple')
    ax[0][1].plot(x_data, cvae_length1, label='w/o c', color='red')
    ax[0][1].set_title('Wizard_of_wikipedia')
    ax[0][1].set_ylim(0, 600)
    ax[0][1].legend()

    ax[1][0].plot(x_data, ground_true_length2, label='GroundTruth', color='green')
    ax[1][0].plot(x_data, gpt2_length2, label='DialoGPT', color='b')
    ax[1][0].plot(x_data, blender90M_length2, label='Blender90M', color='orange')
    ax[1][0].plot(x_data, CTCVAE_length2, label='CT-CVAE', color='purple')
    ax[1][0].plot(x_data, cvae_length2, label='w/o c', color='red')
    ax[1][0].set_title('Empathetic_dialogues')
    ax[1][0].legend()

    ax[1][1].plot(x_data, ground_true_length3, label='GroundTruth', color='green')
    ax[1][1].plot(x_data, gpt2_length3, label='DialoGPT', color='b')
    ax[1][1].plot(x_data, blender90M_length3, label='Blender90M', color='orange')
    ax[1][1].plot(x_data, CTCVAE_length3, label='CT-CVAE', color='purple')
    ax[1][1].plot(x_data, cvae_length3, label='w/o c', color='red')
    ax[1][1].set_title('ConvAI2')
    ax[1][1].legend()

    plt.legend()
    plt.savefig(os.path.join(folderName, '1.png'), bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.clf()
    plt.show()


plot_curve(x,
           gpt2_length0,
           # gpt2_length1,
           # gpt2_length2,
           # gpt2_length3,
           list(map(lambda x: x[0] - x[1], zip(gpt2_length1, gpt2_length0))),
           list(map(lambda x: x[0] - x[1], zip(gpt2_length2, gpt2_length1))),
           list(map(lambda x: x[0] - x[1], zip(gpt2_length3, gpt2_length2))),
           ground_true_length0,
           # ground_true_length1,
           # ground_true_length2,
           # ground_true_length3,
           list(map(lambda x: x[0] - x[1], zip(ground_true_length1, ground_true_length0))),
           list(map(lambda x: x[0] - x[1], zip(ground_true_length2, ground_true_length1))),
           list(map(lambda x: x[0] - x[1], zip(ground_true_length3, ground_true_length2))),
           blender90M_length0,
           # blender90M_length1,
           # blender90M_length2,
           # blender90M_length3,
           list(map(lambda x: x[0] - x[1], zip(blender90M_length1, blender90M_length0))),
           list(map(lambda x: x[0] - x[1], zip(blender90M_length2, blender90M_length1))),
           list(map(lambda x: x[0] - x[1], zip(blender90M_length3, blender90M_length2))),
           transformer_length0,
           # transformer_length1,
           # transformer_length2,
           # transformer_length3,
           list(map(lambda x: x[0] - x[1], zip(transformer_length1, transformer_length0))),
           list(map(lambda x: x[0] - x[1], zip(transformer_length2, transformer_length1))),
           list(map(lambda x: x[0] - x[1], zip(transformer_length3, transformer_length2))),
           cvae_length0,
           cvae_length1,
           cvae_length2,
           cvae_length3,
           CTCVAE_length0,
           CTCVAE_length1,
           CTCVAE_length2,
           CTCVAE_length3,
           )
