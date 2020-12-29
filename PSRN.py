"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
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

image_name = 'baboon.png'
noisy = imageio.imread(noisy_path+image_name)/255
image = imageio.imread(input_path+image_name)/255
image = resize(image, noisy.shape, mode='reflect')

output = imageio.imread(output_path+'/baboon/baboon_out/start_scale=4.png')/255
output = resize(output, noisy.shape, mode='reflect')
plt.imshow(output)
plt.show()


print("Noisy image PSNR", psnr(image, noisy))
print("Denoised image PSNR", psnr(image, output))

