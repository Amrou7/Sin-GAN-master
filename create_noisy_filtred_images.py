import numpy as np
import matplotlib
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from scipy import ndimage
import imageio


input_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Set14/'
noisy_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/GaussianNoise/'
filtered_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Filtered/'

image_name = 'Lenna.png'
image = imageio.imread(input_path+image_name)/255
new_shape = (image.shape[0]/2, image.shape[1]/2, image.shape[2])
image = resize(image, new_shape, mode='reflect')
plt.imshow(image)
plt.show()

sigma = 30
noisy_image = image + np.random.normal(0, sigma/255, image.shape)
noisy_image[noisy_image>1] = 1 
noisy_image[noisy_image<0] = 0 
plt.imshow(noisy_image)
plt.show()
imageio.imwrite(noisy_path+"sigma={}-".format(sigma)+image_name, noisy_image)


filtered = np.zeros(image.shape)
filtered[:,:,0] = ndimage.median_filter(noisy_image[:,:,0], 5)
filtered[:,:,1] = ndimage.median_filter(noisy_image[:,:,1], 5)
filtered[:,:,2] = ndimage.median_filter(noisy_image[:,:,2], 5)
plt.imshow(filtered)
plt.show()
imageio.imwrite(filtered_path+"f-sigma={}-".format(sigma)+image_name, filtered)
