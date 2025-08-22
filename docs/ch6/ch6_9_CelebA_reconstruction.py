"""
=========================================
6.4.1 Generating complex distributions
=========================================

We reproduce the Figure 6.10 from the book, reconstructing a new image not in the dataset.
We train our model on the CelebA dataset, and then, using a test image, we reconstruct it from its latent representation.
"""

#########################################################################
# Necessary Imports
# ------------------------
#Everything is already made in the previous file, so we just import it here.


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

from ch6_9_CelebA import * 

config = {
    'input_shape':[218,178],
    'rescale_shape':[50,50],
    'Nx':1000,  # Number of images to use for training
    'Nz':16, # Numer of images to generate
    'seed':43,
    "main_folder":os.path.join(data_path,'celebA'),
}
def reconstruction_high_dimension(
        kwargs = config,
        selected_features=['Young', 'Blond_Hair','Attractive','Smiling'],
        drop_features=['Male'],
        dimensions = [40]
        ):

    kwargs['Nz'] = 1 

    # Load images
    celeba = CelebA_data_generator(selected_features=selected_features,drop_features=drop_features)
    x_target,fx,y = celeba.get_data(**kwargs)
    # We keep the first one for testing
    x_target = x_target.values

    test = x_target[0]
    print(f"Loaded {kwargs['Nx']} images")

    # This will enable us to sample from the latent space
    # We don't pass the first image, as it is used for testing
    for d in dimensions:
        sampler = Sampler(x=x_target[1:], latent_dim=d, iter=0)
        # We need a kernel to encode the testing image into latent space
        origin_latent = sampler.get_x() # (Nx, d)
        encoder = Kernel(x=x_target[1:], fx=origin_latent, order=2)
    
        # Encode the testing image into latent space
        test_latent = encoder(z=test.reshape(1,-1))
        # Reconstruct it in the downascaled space
        test_reconstruct = sampler(z=test_latent) # (1, kwargs['input_shape'])    

        # Computing distance between original and generated in latent space
        dist = sampler.dnm(x=test_latent, y=origin_latent) # (1, Nx)
        
        # Aligned idxs in latent space
        indices = np.argmin(dist, axis=1) # (1,)
        
        # Getting back original closest to the reconstructed image in latent space
        origin = sampler(origin_latent[indices]) # (1, kwargs['input_shape'])
        
        result = np.concatenate([test.reshape(1,-1), test_reconstruct, origin], axis=0) # (3, kwargs['input_shape'])
        result_tile = tiles(result,pic_shape=kwargs['input_shape'],tile_shape=[1,3])
        pic_name = str(kwargs['Nx'])+"D_"+str(d)+".png"
        plt.imsave(os.path.join(proj_path,"pic_reconstructed_N"+pic_name),result_tile, vmin=0., vmax = 1.)
        plt.show()
        print("Saved",pic_name)

reconstruction_high_dimension(dimensions = [8])