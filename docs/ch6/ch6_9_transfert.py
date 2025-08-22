"""
==================================================================
6.4.5 Conditioning on discrete distributions of celebA dataset
==================================================================

We reproduce the Figure 6.15 from the book.
We train a model on the CelebA dataset, conditionned on the features. 
We then select 10 images with hats and eyeglasses, and slowly generate new images 
By varying the strength of those features, resulting in a verion of "no hats and eyeglasses"
"""

#########################################################################
# Necessary Imports
# ------------------------
#Everything is already made in the previous file, so we just import it here.

from ch6_9_CelebA import * 


config = {
    'input_shape':[218,178],
    'rescale_shape':[50,50],
    'Nx':1000,  # Number of images to use for training
    'Nz':4, # Numer of images to generate
    'seed':40,
    "main_folder":os.path.join(data_path,'celebA'),
}
def style_transferts(
      selected_features=['Young','Attractive'],
      drop_features=['Male','Heavy_Makeup'],
      conditionned_features=['Wearing_Hat','Eyeglasses'],
      other_features=['Blond_Hair','Smiling'],
      dimension = 2,
      **kwargs
    ):

    n_images = config['Nx']
    Nz = config['Nz']
    # Collect all images
    celeba = CelebA_data_generator(selected_features=selected_features,drop_features=drop_features)
    x_target_all, img_paths_all, features_all = celeba.get_data(conditionned_features=conditionned_features,**kwargs)
    features_all.drop(columns='path', inplace=True)  # Remove the path column
    # pics = tiles(x_target_all.loc[['005350.jpg']].values,pic_shape=kwargs['input_shape'],tile_shape=[1,1])  
    # plt.imshow(pics)  

    # We get those with hats and/or eyeglasses
    if len(other_features) > 0:
        features_all = features_all[conditionned_features+other_features]
    
    all_conditionned_and = features_all.copy()
    for f in conditionned_features:
        all_conditionned_and = all_conditionned_and.loc[all_conditionned_and[f]==+1]    
    # idx_and = all_conditionned_and.index[:Nz]
    idx_and = all_conditionned_and.index

    print(f"selected ids: {idx_and}")
    # We get again new images with none of those, whatever remains from the required initial number of images
    features_and = features_all.loc[idx_and].copy()
    # all_features = features.values[:,:-1] # Remove the last column which is the image path

    all_features = features_all.reset_index()
    x_target_all = x_target_all.reset_index()  
    idx_and = all_features[all_features['image_id'].isin(idx_and)].index
    x_target_all.drop(columns='image_id', inplace=True)  # Remove the image_id column
    all_features.drop(columns='image_id', inplace=True)  # Remove the image_id column
    print(f"Dataset Ready")

    # We condition y | x
    # y are our images
    # x are our features
    conditionner = codpy.conditioning.ConditionerKernel(
        x=all_features,
        y=x_target_all,
        latent_dim_y=dimension
    )

    conditionner.set_maps(iter=0)

    permut_idx_and = map_invertion(np.array(conditionner.sampler_xy.permutation))[idx_and]
    latent_values_y = conditionner.latent_y
    latent_values_y_and = latent_values_y[permut_idx_and]
    latent_values_x = conditionner.latent_x
    latent_values_x_and = latent_values_x[permut_idx_and]

    #compute latent values of selected pictures
    latent_images_and = np.concatenate([latent_values_x_and,latent_values_y_and],axis=1) # -1 because we remove the img path
    #check the reconstruction. comment after checking
    # pics = tiles(conditionner.get_y()[idx_and],pic_shape=kwargs['input_shape'],tile_shape=[1,len(idx_and)])  
    # plt.imshow(pics) 
    # plt.show() 
    # reconstructed_images = conditionner.sampler_xy(latent_images_and)[:,all_features.shape[1]:]
    # pics = tiles(reconstructed_images,pic_shape=kwargs['input_shape'],tile_shape=[1,len(idx_and)])  
    # plt.imshow(pics)  
    # plt.show() 


    print("conditionner ready")
    results=None
    

    latent_images_and = latent_images_and[:Nz]
    cond_feat = latent_images_and.copy()

    # feature_strength_list = [1, 0.5, 0.0, -0.5, -1]
    feature_strength_list = [1, 0.5, 0.0, -0.5, -1]
    for feature_strength in feature_strength_list:
        # Update the and features on the conditionned features and extract values 
        cond_feat[:,:len(conditionned_features)] = latent_images_and[:,:len(conditionned_features)]*feature_strength

        # Sampling new images 
        sampled_images = conditionner.sampler_xy(cond_feat)[:,all_features.shape[1]:]
        if results is None:
            results = sampled_images
        else:
            results = np.concatenate([results, sampled_images], axis=0)
        # print(f"Added {feature_strength} feature results")
    pics = tiles(results,pic_shape=config['input_shape'],tile_shape=[len(feature_strength_list),results.shape[0] // len(feature_strength_list)])
    plt.imshow(pics) 
    plt.show()
    pic_name = str(n_images)+"D_"+str(dimension)+".png"
    plt.imsave(os.path.join(proj_path,"pic_transfert_N"+pic_name),pics, vmin=0., vmax = 1.)
    print("Saved",pic_name)
    pass

style_transferts(
      selected_features=['Young','Attractive'],
      drop_features=['Male','Heavy_Makeup'],
      conditionned_features=['Eyeglasses','Wearing_Hat'],
      other_features=['Blond_Hair','Smiling'],
      dimension = 4,
      **config)