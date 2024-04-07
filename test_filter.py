import numpy as np
import cv2
import matplotlib.pyplot as plt

def ideal_low_pass_filter(img, d0):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols), np.uint8)
    for x in range(0, rows):
        for y in range(0, cols):
            d = np.sqrt((crow-x)**2+(ccol-y)**2)
            if d <= d0:
                mask[x, y] = 1

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(np.log(np.abs(dft_shift)), cmap='gray')
    plt.title('Frequency Domain')

    dft_shift *= mask
    plt.subplot(2, 2, 3)
    plt.imshow(np.log(1 + np.abs(dft_shift)), cmap='gray')
    plt.title('Filtered Frequency Domain')

    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(2, 2, 4)
    plt.imshow(img_back, cmap='gray')
    plt.title('Filtered Image')

    plt.show()
    return img_back.astype(np.uint8)


def Butterworth_low_pass_filter(img, d0, n):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols))
    for x in range(rows):
        for y in range(cols):
            d = np.sqrt((x-crow)**2 + (y-ccol)**2)
            mask[x, y] = 1 / (1 + (d/d0)**(2*n))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(np.log(np.abs(dft_shift)), cmap='gray')
    plt.title('Frequency Domain')

    dft_shift *= mask
    plt.subplot(2, 2, 3)
    plt.imshow(np.log(1 + np.abs(dft_shift)), cmap='gray')
    plt.title('Filtered Frequency Domain')

    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(2, 2, 4)
    plt.imshow(img_back, cmap='gray')
    plt.title('Filtered Image')

    plt.show()
    return img_back.astype(np.uint8)

def Gaussian_low_pass_filter(img, d0):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros((rows, cols))
    for x in range(rows):
        for y in range(cols):
            d = (x - crow)**2 + (y - ccol)**2
            mask[x, y] = np.exp(-d / (2 * d0**2))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(np.log(np.abs(dft_shift)), cmap='gray')
    plt.title('Frequency Domain')

    dft_shift *= mask
    plt.subplot(2, 2, 3)
    plt.imshow(np.log(1 + np.abs(dft_shift)), cmap='gray')
    plt.title('Filtered Frequency Domain')

    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(2, 2, 4)
    plt.imshow(img_back, cmap='gray')
    plt.title('Filtered Image')

    plt.show()
    return img_back.astype(np.uint8)


# 读取图像并应用理想低通滤波器
gray = cv2.imread("1.webp", cv2.IMREAD_GRAYSCALE)
gray_smooth = Gaussian_low_pass_filter(gray,15)
cv2.imwrite()