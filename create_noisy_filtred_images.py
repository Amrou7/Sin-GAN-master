import numpy as np
import matplotlib
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from scipy import ndimage
import imageio
from os import walk
import random


input_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Set14/'
noisy_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/GaussianNoise/'
filtered_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Filtered/'


_, _, filenames = next(walk(input_path))
print(filenames)


noise_type = "Gaussian" ## Gaussian Or SP

for image_name in filenames:
    image = imageio.imread(input_path+image_name)/255
    
    if max(image.shape) > 600:  
        new_shape = (image.shape[0]//3, image.shape[1]//3, image.shape[2])
        image = resize(image, new_shape, mode='reflect')
    elif max(image.shape) > 400:
        new_shape = (image.shape[0]//2, image.shape[1]//2, image.shape[2])
        image = resize(image, new_shape, mode='reflect')
        
    plt.imshow(image)
    plt.show()
    
    if noise_type=="Gaussian":
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
    else:
        
        noisy_image = image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < 0.05:
                    noisy_image[i][j] = 0
                elif rdn > 0.95:
                    noisy_image[i][j] = 1
                else:
                    noisy_image[i][j] = image[i][j]  
                    
        plt.imshow(noisy_image)
        plt.show()
        imageio.imwrite(noisy_path+"SP-"+image_name, noisy_image)
        
        
        filtered = np.zeros(image.shape)
        filtered[:,:,0] = ndimage.median_filter(noisy_image[:,:,0], 5)
        filtered[:,:,1] = ndimage.median_filter(noisy_image[:,:,1], 5)
        filtered[:,:,2] = ndimage.median_filter(noisy_image[:,:,2], 5)
        plt.imshow(filtered)
        plt.show()
        imageio.imwrite(filtered_path+"f-SP-"+image_name, filtered)
