import numpy as np


def TV_denoising(y0, lambda_val, iter=100):
    if iter is None:
        iter = 100
    z = np.zeros(np.maximum(np.array(y0.shape) - 1, 1))  # the differential
    alpha = 5

    for it in range(iter):
        if y0.shape[1] == 1:  # 1D vector
            x0 = y0 - dvt(z)
            z = clip(z + 1 / alpha * dv(x0), lambda_val / 2)
        elif y0.shape[1] > 1 and y0.shape[2] == 1:  # 2D image
            if it == 0:
                zh = np.zeros((y0.shape[0], y0.shape[1] - 1))
                zv = np.zeros((y0.shape[0] - 1, y0.shape[1]))
            x0h = y0 - dht(zh)
            x0v = y0 - dvt(zv)
            x0 = (x0h + x0v) / 2
            zh = clip(zh + 1 / alpha * dh(x0), lambda_val / 2)
            zv = clip(zv + 1 / alpha * dv(x0), lambda_val / 2)
        elif y0.shape[2] > 1 and y0.shape[3] == 1:  # 3D video or hyperspectral images, but TV is done by each 2D frame
            if it == 0:
                zh = np.zeros((y0.shape[0], y0.shape[1]-1, y0.shape[2]))
                zv = np.zeros((y0.shape[0]-1, y0.shape[1], y0.shape[2]))
            x0h = y0 - np.expand_dims(dht_3d(zh.squeeze()), axis=-1)
            x0v = y0 - np.expand_dims(dvt_3d(zv.squeeze()), axis=-1)
            x0 = (x0h + x0v) / 2
            x0 = x0.squeeze()
            zh = clip(zh + 1 / alpha * dh(x0), lambda_val / 2)
            zv = clip(zv + 1 / alpha * dv(x0), lambda_val / 2)
        elif y0.shape[3] > 1 and y0.shape[4] == 1:  # 4D hyperspectral-video, but TV is done by each 2D frame
            if it == 0:
                zh = np.zeros((y0.shape[0], y0.shape[1] - 1, y0.shape[2], y0.shape[3]))
                zv = np.zeros((y0.shape[0] - 1, y0.shape[1], y0.shape[2], y0.shape[3]))
            x0h = y0 - dht_4d(zh)
            x0v = y0 - dvt_4d(zv)
            x0 = (x0h + x0v) / 2
            zh = clip(zh + 1 / alpha * dh(x0), lambda_val / 2)
            zv = clip(zv + 1 / alpha * dv(x0), lambda_val / 2)

    return x0


def dv(x):
    y = np.diff(x, axis=0)
    return y


def dh(x):
    y = np.diff(x, axis=1)
    return y


def dvt(x):
    y = np.concatenate(([-x[0]], -np.diff(x, axis=0), [x[-1]]))
    return y


def dht(x):
    y = np.concatenate(([-x[:, 0]], -np.diff(x, axis=1), [x[:, -1]]))
    return y


def dvt_3d(x):
    y = np.concatenate((-x[:1, :, :], -np.diff(x, axis=0), x[-1:, :, :]), axis=0)
    return y


def dht_3d(x):
    y = np.concatenate((-x[:, :1, :], -np.diff(x, axis=1), x[:, -1:, :]), axis=1)
    return y


def dvt_4d(x):
    y = np.concatenate((-x[0:1], -np.diff(x, axis=0), x[-1:]), axis=0)
    return y


def dht_4d(x):
    y = np.concatenate((-x[:, 0:1], -np.diff(x, axis=1), x[:, -1:]), axis=1)
    return y


def clip(x, lambda_val):
    y = np.sign(x) * (np.minimum(np.abs(x), lambda_val))
    return y