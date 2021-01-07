import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from scipy import ndimage
import imageio

def psnr(clean, img):
    img = img
    clean = clean
    mse = np.mean((clean-img)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX/np.sqrt(mse))


input_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Set14/'
noisy_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/GaussianNoise/'
output_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Output/Paint2image/'

image = imageio.imread(input_path+"Lenna.png")/255
noisy = imageio.imread(noisy_path+'sigma=30-Lenna.png')/255
noisy = resize(noisy, image.shape, mode='reflect')
plt.imshow(noisy)
plt.show()
plt.imshow(image)
plt.show()

output = imageio.imread(output_path+'sigma=30-Lenna/sigma=30-Lenna_out/start_scale=9.png')/255
output = resize(output, image.shape, mode='reflect')
plt.imshow(output)
plt.show()


print("Noisy image PSNR", psnr(image, noisy))
print("Denoised image PSNR", psnr(image, output))


