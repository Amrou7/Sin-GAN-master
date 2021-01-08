import numpy as np
import matplotlib.pylab as plt
from skimage.transform import resize
import imageio
from os import walk
from skimage.restoration import denoise_nl_means, estimate_sigma



noisy_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/GaussianNoise/'
NLmeans_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Output/NLmeans/Gaussian/'


_, _, filenames = next(walk(noisy_path))

for image_name in filenames:
    print(image_name)
    noisy = imageio.imread(noisy_path+image_name)/255
    plt.imshow(noisy)
    plt.show()
    
    sigma_estimation = np.mean(estimate_sigma(noisy, multichannel=True))
    denoised = denoise_nl_means(noisy, h=0.7*sigma_estimation, fast_mode=True)
    plt.imshow(denoised)
    plt.show()  
    
    imageio.imwrite(NLmeans_path+image_name, denoised)
