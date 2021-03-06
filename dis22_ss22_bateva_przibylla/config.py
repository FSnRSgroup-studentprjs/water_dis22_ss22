import os
import numpy as np
import random
import itertools
########################################################################################################################
#                                   Performance Settings, Multi-GPU-usage
########################################################################################################################
# list of gpus to use
# (only use 1 GPU!) Otherwise I will kill your process!
# We all need to calculate on this machine - might get lowered to one, if there are bottlenecks!
gpus = [0,1]

########################################################################################################################
#                                           Verbosity settings
########################################################################################################################
#handles verbosity of the program (use 1 or above to get feedback!)
verbose = 1
#turn off cryptic warnings, Note that you might miss important warnings! If unexpected stuff is happening turn it on!
#https://github.com/tensorflow/tensorflow/issues/27023
#Thanks @Mrs Przibylla
#'1' = Infos, '2' = warnings, '3' = Errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

########################################################################################################################
#                                           Base Folder
########################################################################################################################
### The folder where the project path and data is located, subfolder paths will be automatically generated
base_folder = '/mnt/datadisk/data/'
# The folder where your project files are located
prj_folder = '/mnt/datadisk/data/Projects/water_dis22_ss22/'
# train history will be saved in a subfolder of the project path (base_folder + /projects/water/)
# assign a name according to your group, to separate your results from all others! Create this folder manually!

trainHistory_subname = 'trainHistory_bateva_przibylla_3'

########################################################################################################################
#                                            Run Name
########################################################################################################################
### The name of the model run gets generated by mutiple Settings (e.g. model_name, normalization and augmentation
# settings and many more, it will be created as a folder where all results model checkpoints, evaluation charts etc. pp
# are saved
# run_name_custom_string is a custom string you can provide to add to the run name
run_name_custom_string = ''

########################################################################################################################
#                                       Dataset parameters
########################################################################################################################
# 3 Main label categories - everything else is thrown away
main_labels = ['piped', 'groundwater', 'bottled water']
num_labels = len(main_labels)
# [Minimum, maximum values] for clipping above and below those pixel values
clipping_values = [0, 3000]
# Channels (define channels which should be used, if all should be used provide an empty list = [])
channels = [4, 3, 2]
channel_size = len(channels)

########################################################################################################################
#                                  Basic neural network parameters
########################################################################################################################
### Maximum amount of epochs

epochs = 5

### Learning rate (to start with - might get dynamically lowered with callback options)
lr = 0.001 #1e-4
# Learning rate decay (used to get rid of the noise)
decay = 1
lr_decay = (1/(1 + decay * epochs)) * lr
# How many pictures are used to train before readjusting weights
batch_size = 32
### The model to use
# available are vgg16/19, resnet50/152, inceptionv3, xception, densnet121/201
model_name = 'vgg16'
# loss function
loss = 'categorical_crossentropy'
# chose your optimizer
optimizer = "SGD"
# momentum influences the lr (high lr when big changes occur and low lr when low changes occur)
momentum = 0.9

### The shape of X and Y with ((batch_size, height, width, channels), (batch_size, number of labels))
input_shape = ((batch_size, 200, 200, channel_size), (batch_size, num_labels))

### CNN settings which are parameters of the tf.keras.applications Model
# 'include_top': Use the same top layers (aka final layers with softmax on output classes) - should always be "False" here
# 'weights': 'imagenet' or False - transfer learning option - will be overwritten if model weights are given in
# load_model_weights. Only takes effect with include_top=False
# classifier activation: A str or callable. The activation function to use on the "top" layer. Ignored unless include_top=True.
# Set classifier_activation=None to return the logits of the "top" layer.
# When loading pretrained weights, classifier_activation can only be None or "softmax".
# others kinda explain themselves
cnn_settings_d = {'include_top': False, 'weights': 'imagenet', 'input_tensor': None,
                   'input_shape': (input_shape[0][1], input_shape[0][2], input_shape[0][3]), 'pooling': False, 'classes': 3}
                  #'classifier_activation': 'softmax'}
# Use dropout on top layers - use 0 to 100 (percent)
dropout_top_layers = 0
# Make weights trainable. Unfreezes layers beginning at top layers. Use 0 to 100 (percent)
### Dictionary containing values


unfreeze_dict = {"unfreeze_type": "unfreeze_blocks",
          "unfreeze_layers_perc": 86, # Use custom top layers (necessary when using transferlearning),
          "unfreeze_at": 17,
          "unfreeze_blocks": ['input', 'block1']}

 #list(unfreeze_dict)[2]

### Use custom top layers (necessary when using transferlearning)
# 2 layers right now. Their neurons can be adjusted
add_custom_top_layers = True
# define how many hidden layers shall be added as top layers (len(neurons_l)) and how many neurons they should have (int)
neurons_l = [1024, 512]

########################################################################################################################
#                                       Callback options
########################################################################################################################
# automatically decrease lr (first value True) if there was no decrease in loss after x epochs (2nd value)
# 3rd multiplicator for lr
auto_adjust_lr = (False, 4, 0.9)
# model stops (first value True) when loss doesnt decrease over epochs (2nd value)
early_stopping = (True, 30)

########################################################################################################################
#                       ImageDataGenerator (IDG - Keras) Settings
########################################################################################################################
### Use 'ImageDataGenerator' or False - Shannons generator gets used when False
generator = 'ImageDataGenerator'

### Normalization settings for IDG
# only featurewise settings and zca_whitening get fitted or respectively (Mean, standarddeviation, PCA) get precalculated
# IDG.fit() only gets called for parameters in here
# dict with sub dicts to be able to use in a testing strategy
# these get applied to train, test and validation data
IDG_normalization_d = {
   # 'samplewise': {'samplewise_center': True,  # Boolean. Set each sample mean to 0. #needs IDG.fit()
   #                'samplewise_std_normalization': True},  # needs IDG.fit()
    'featurewise': {'featurewise_center': True, #Boolean. Set input mean to 0 over the dataset, feature-wise. #needs IDG.fit()
    'featurewise_std_normalization': True},  # needs IDG.fit()
    # 'manually_normalize': {},  # dummy for no normalization
    # 'rescale': 100,
    # 'zca_whitening': True  # Boolean. Apply ZCA whitening. #needs IDG.fit(), takes ages and kills process regularly
}
### ZCA is extremly expensive - thus we only wanna calculate it on a subset of input images
# use a number from 1 to 100 (percent)
# Experimental state - usually crashes!!!
zca_whitening_perc_fit = 1

### Image Augmentation settings
# all possible values below
# accuracy decreases with to many Augmentation settings, though - why?
# dict of dicts to be able to test multiple settings in a testing strategy
# augmentation only gets applied to train data
IDG_augmentation_settings_d = {'subset1': {
        #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        #'zoom_range': 0.2,
        #'channel_shift_range': 0.3,
        'horizontal_flip': True,
        'vertical_flip': True,
        #'rotation_range': 20, #Int. Degree range for random rotations.
        'width_shift_range': 0.2,
        'height_shift_range': 0.2
        }}

########################################################################################################################
#                               Continuing from earlier runs
########################################################################################################################
### False or modelcheckp(oint) folder from which to load weights
load_model_weights = False
#os.path.join(base_folder, "Projects/water/trainHistory_augmentation/0206/68acc_vgg19img_net_momentum0.9_optimizerSGD_brightness_0.9_1.1shear_0.2channel_shift_0.3horizontal_flip_vertical_flip__1/modelcheckp/")

########################################################################################################################
#                               Evaluation Settings & Images
########################################################################################################################
### You can reload the best model epoch (True/False) - in that case evaluation is done on both, the best model epoch and
# the last one
# overrides other weights imports (e.g. imagenet in Keras models)
reload_best_weights_for_eval = True
### You can show and/or save your augmented images to become an idea of what actually goes into the model
# False or Number of images (for train, val and test gen)
save_augmented_images = 15
tensorboard = True

########################################################################################################################
#                                                   Fine-Tuning
########################################################################################################################
# 
fine_tuning = True # If you set fine-tuning as true you will run 2 iterations of the model
epochs_freezing = 2 # Number of epochs you want to train with the aforementioned settings
epochs_unfreezing = 4 # Number of Epochs you want to train unfreezed.

########################################################################################################################
#                                                   Testsettings
########################################################################################################################
# Activate / Deactivate the test mode. Could be True or False
testing = False

# Set the test mode. In case of 'all' combinations of all hyperparameter specified in the dictionary below (model_test_param)
# will be generated. If testing = True the model will be trained with all these combinations. The 'random' mode
# picks a random set of hyperparameters from the generated combinations. If the testing mode is activaed the model
# will be trained with these hyperparameters

mode = 'all' #'random'

# Specify single hyperparameters to be in testing mode
test_lr = True
test_unfreeze_layers_perc = True
test_dropout_top_layers = True
test_IDG_augmentation_settings_d = True

test_epochs = 25
########################################################################################################################
#                                                   Testcode
########################################################################################################################
# Define a function that generates all possible hyperparameter combinations from dictionaries of hyperparameters. The
# hyperparameter values that need to be tested are stored in lists representing the values of the dictionaries.
def generate_all_param_combinations():
    model_test_param = {"learning_rates": [0.001, 0.01, 0.1, 1e-4],
                        "dropout_top_layers": [0.2, 0.3, 0.4, 0.5],
                        "unfreezed_layers_perc": [20, 40, 60, 80]}

    # Create list of dictionares with all possible hyperparameter combinations from model_test_param
    keys, values = zip(*model_test_param.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Generate a dictionary of the augmentation settings that have to be tested
    IDG_augmentation_settings_d_params = {'subset1': {
        # 'brightness_range': [0.9, 1.1],
        'shear_range': 0.2,  # Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        'zoom_range': [0.8, 1.2],
        'channel_shift_range': 0.3,
        'horizontal_flip': True,
        'vertical_flip': True,  # Degree range for random rotations.
        'width_shift_range': 0.2,
        'height_shift_range': 0.2
    },
        'subset2': {
            # 'brightness_range': [0.3, 1.7],
            'shear_range': 0.5,  # Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
            'zoom_range': [0.5, 1.1],
            'channel_shift_range': 0.7,
            'horizontal_flip': True,
            'vertical_flip': False,  # Degree range for random rotations.
            'width_shift_range': 0.6,
            'height_shift_range': 0.2
        }}

    # create list of dictionaries with all possible hyperparameter and augmentation setting combinations
    # values are passed as lists to the dictionaries containing hyperparameter and augmentation settings respectively

    # create an empty list in which the generated pyrameter combinations will be stored
    list_all_param_combinations = []
    for combination in param_combinations:
        for subset, settings in IDG_augmentation_settings_d_params.items():
            augm_dict = {}  # create an empty list with the augmentation setting, which will be attached to the big combi_dict
            augm_dict[subset] = settings # fill the combined augm_dict and pass "subset" as key
            combined_dic = {**combination, **augm_dict}
            list_all_param_combinations.append(combined_dic) #append the combined dictionaries to the list of all parameters

    return list_all_param_combinations


# Source: https://towardsdatascience.com/how-to-rapidly-test-dozens-of-deep-learning-models-in-python-cb839b518531
# How to rapidly test dozens of deep learning models in Python

# Pick a random combination from the parameter combinations generated from the previous function. If the test mode is
# set to random the model would be trained with that hyperparameter set

'''
def generate_random_param_combinations():
    list_all_param_combinations = generate_all_param_combinations()
    random_candidate = random.randrange(0, len(list_all_param_combinations))
    param_combination = list_all_param_combinations[random_candidate]
    lr = param_combination['learning_rates']
    dropout_top_layers = param_combination['dropout_top_layers']
    unfreezed_layers_perc = param_combination['unfreezed_layers_perc']
    return unfreezed_layers_perc, dropout_top_layers, lr, IDG_augmentation_settings_d

'''

# <Kommentar>
'''
if testing:
    if mode == "random":
        random_param_combinations = generate_random_param_combinations()
        if test_unfreeze_layers_perc:
            unfreeze_layers_perc = random_param_combinations[0]
        if test_dropout_top_layers:
            dropout_top_layers = random_param_combinations[1]
        if test_lr:
            lr = random_param_combinations[2]
        if test_IDG_augmentation_settings_d:
            IDG_augmentation_settings_d = random_param_combinations[3]
else:
    pass
'''
