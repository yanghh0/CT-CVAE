
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# S2S_length = [0, 4, 404, 821, 782, 2615, 2943, 2832, 4002, 1545, 1764, 1176, 1101, 1233, 774,
#               775, 591, 413, 286, 233, 146, 115, 105, 86, 73, 53, 42, 40, 23, 18, 12, 8, 6, 6,
#               4, 3, 4, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0]
# HRED_length = [0, 2, 352, 633, 559, 2467, 2571, 2763, 3942, 1431, 1915, 1414, 1231, 1134, 823,
#                870, 709, 517, 368, 299, 209, 184, 140, 105, 91, 84, 41, 53, 36, 24, 14, 20, 7,
#                13, 6, 6, 2, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# CVAE_length = [0, 24, 374, 494, 780, 1601, 1791, 2010, 2111, 1927, 1884, 1759, 1572, 1451, 1203,
#                1066, 821, 692, 566, 443, 395, 322, 278, 223, 215, 182, 148, 106, 95, 86, 66, 71,
#                42, 36, 28, 22, 19, 23, 15, 12, 9, 13, 11, 4, 13, 4, 6, 4, 6, 2]

ground_truth = [0, 15, 164, 262, 360, 781, 1142, 1311, 1465, 1468, 1508, 1543, 1545, 1452, 1453,
                1403, 1260, 1127, 977, 799, 594, 532, 390, 398, 325, 323, 238, 261, 219, 198, 179,
                149, 134, 121, 98, 107, 76, 80, 56, 55, 44, 40, 46, 32, 25, 27, 27, 16, 21, 28]

dialoggpt_length = [0, 276, 1538, 1876, 2726, 3675, 3134, 3883, 1807, 1450, 1142, 972, 803, 475, 326,
               237, 169, 125, 148, 80, 47, 42, 39, 29, 19, 6, 7, 4, 3, 0, 1, 2, 1, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


x = list(range(len(dialoggpt_length)))

folderName = "F:\\newest"

def plot_curve(x_data, ground_truth, gpt2_length):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.xticks(rotation=55)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.rcParams['axes.unicode_minus'] = False
    # f,ax = plt.subplots(2,2)
    plt.subplots_adjust(bottom=0., left=0, top=1., right=1)
    plt.plot(x_data, ground_truth, label='GroundTruth')
    plt.plot(x_data, gpt2_length, label='DialoGPT')
    plt.xlabel('Length')
    plt.ylabel('Number')
    plt.legend()
    plt.savefig(os.path.join(folderName, '1.png'), bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.clf()
    plt.show()

    # ax[0][0].plot(x_data, ground_truth, label='GroundTruth', color='b') # , linewidth=2
    # ax[0][0].plot(x_data, S2S_length, label='Seq2Seq', color='r')
    # ax[0][0].plot(x_data, HRED_length, label='HRED', color='orange')
    # ax[0][0].legend()
    # ax[0][1].plot(x_data, S2S_length, label='Seq2Seq', color='r')
    # ax[0][1].plot(x_data, gpt2_length, label='DialoGPT', color='green')
    # ax[0][1].legend()
    # ax[1][0].plot(x_data, ground_truth, label='GroundTruth', color='b')
    # ax[1][0].plot(x_data, S2S_length, label='Seq2Seq', color='r')
    # ax[1][0].plot(x_data, CVAE_length, label='CVAE', color='gray')
    # ax[1][0].legend()
    # plt.subplot(2, 2, 3)
    # plt.xlabel('Length')
    # plt.ylabel('Number')
    # plt.plot(x_data, transformer, label='transformer')
    # plt.title('Length statistics of generated responses')

plot_curve(x, ground_truth, dialoggpt_length)