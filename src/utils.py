import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import torch
import torch.nn as nn

def drawFigure(img,preds,img_path):
    '''
    draw a png figure with rect of ground truth and prediction
    col == x, row == y
    :param img:
    :param labels:
    :param img_path
    :return:
    '''
    fig, ax = plt.subplots(1)
    row, col = img.shape
    ax.imshow(img)
    preds = preds * row
    # draw predict patch
    x1, y1, x2, y2 = preds[:4]
    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect1)
    x1, y1, x2, y2 = preds[4:]
    rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect2)
    # save image
    plt.savefig(img_path, dpi=300)
    plt.tight_layout()
    plt.close()

def getKneeWithBbox(img,bbox):
    row, col = img.shape
    x1, y1, x2, y2 = bbox[:4]

    x1 = int(x1 * col)
    x2 = int(x2 * col)
    y1 = int(y1 * row)
    y2 = int(y2 * row)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    # max is used to avoid negative index
    left = img[max(cy - 512, 0): max(cy - 512, 0) + 1024, max(cx - 512, 0): max(cx - 512, 0) + 1024]
    x1, y1, x2, y2 = bbox[4:]
    x1 = int(x1 * col)
    x2 = int(x2 * col)
    y1 = int(y1 * row)
    y2 = int(y2 * row)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    right = img[max(cy - 512, 0): max(cy - 512, 0) + 1024, max(cx - 512, 0): max(cx - 512, 0) + 1024]
    return left,right


def model_predict(model, left, right):

    left = np.expand_dims(left, axis=2)
    left = np.repeat(left[:, :], 3, axis=2)
    left = center_crop(left)
    right = np.expand_dims(right, axis=2)
    right = np.repeat(right[:, :], 3, axis=2)
    right = center_crop(right)

    left = torch.from_numpy(left).float()
    right = torch.from_numpy(right).float()
    left = left.permute(2, 0, 1).unsqueeze(0)
    right = right.permute(2, 0, 1).unsqueeze(0)

    left_pred = model(left)
    right_pred = model(right)
    sm = nn.Softmax(dim=1)
    left_pred = sm(left_pred)
    right_pred = sm(right_pred)

    left_pred = left_pred.detach().numpy()
    right_pred = right_pred.detach().numpy()
    left_pred = np.argmax(left_pred, axis=1)
    right_pred = np.argmax(right_pred, axis=1)

    return left_pred.tolist(), right_pred.tolist()


def center_crop(img, size=(898,898)):

    w, h = img.shape[0], img.shape[1]
    tw, th, = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img[y1:y1 + th, x1:x1 + tw]  # img.crop((x1, y1, x1 + tw, y1 + th))


