import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import os


def select_matches(img0, img1, n_pts):
    img = np.concatenate((img0, img1), axis=1)

    plt.ion()
    plt.imshow(img)
    pts_list = np.array(plt.ginput(2 * n_pts, timeout=0))

    x0 = pts_list[pts_list[:, 0] < img0.shape[1]]
    x1 = pts_list[pts_list[:, 0] >= img0.shape[1]] - np.array([img0.shape[1], 0])

    if len(x0) != len(x1):
        raise Exception('Number of points is not equal on both sides!')

    x0 = np.vstack((x0.T, np.ones(n_pts)))
    x1 = np.vstack((x1.T, np.ones(n_pts)))
    return x0, x1


def main():
    img0 = imageio.imread('house1.jpg')
    img1 = imageio.imread('house2.jpg')

    x0_selected, x1_selected = select_matches(img0, img1, n_pts=20)

    np.savetxt('house_manual_matches_x1.csv', x0_selected, delimiter=',')
    np.savetxt('house_manual_matches_x2.csv', x1_selected, delimiter=',')


if __name__ == '__main__':
    main()
