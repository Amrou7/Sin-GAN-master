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
filtered_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Filtered/'
NLmeans_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/NLmeans/Gaussian/'


#### Clean image
image = imageio.imread(input_path+"bridge.png")/255
plt.imshow(image)
plt.show()

#### Noisy image
noisy = imageio.imread(noisy_path+'sigma=30-bridge.png')/255
noisy = resize(noisy, image.shape, mode='reflect')
plt.imshow(noisy)
plt.show()
print("Noisy image PSNR", psnr(image, noisy))


#### SinGAN result
#output = imageio.imread(output_path+'NL-sigma=30-lenna/sigma=30-lenna_out/start_scale=7.png')/255
output = imageio.imread(output_path+'f-sigma=30-bridge/start_scale=8.png')/255
output = resize(output, image.shape, mode='reflect')
plt.imshow(output)
plt.show()
print("sinGAN Denoised image PSNR", psnr(image, output))


### Median filter
filtered = imageio.imread(filtered_path+"f-sigma=30-bridge.png")/255
filtered = resize(filtered, image.shape, mode='reflect')
print("Median-Filter denoised image PSNR", psnr(image, filtered))
plt.imshow(filtered)
plt.show()


#### NLmeans
NLmeans_denoised = imageio.imread(NLmeans_path+"NL-sigma=30-barbara.png")/255
NLmeans_denoised = resize(NLmeans_denoised, image.shape, mode='reflect')
print("NLmeans denoised image PSNR", psnr(image, NLmeans_denoised))
plt.imshow(NLmeans_denoised)
plt.show()
