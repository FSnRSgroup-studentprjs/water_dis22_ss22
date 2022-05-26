import numpy as np
import random

def generate_random_cnn():
    learning_rates = [0.001, 0.01, 0.1, 1e-4]

    unfreeze_layers_perc = random.randrange(80, 100)
    dropout_top_layers = random.randrange(20, 50, 10)
    lr = np.random.choice(learning_rates)
    # Learning rate decay (used to get rid of the noise)
    ###decay = 1
    ###lr_decay = (1 / (1 + decay * epochs)) * lr

    IDG_augmentation_settings_d = {'subset1': {
        #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        'zoom_range': round(random.uniform(0.8, 1.2), 1),
        #'channel_shift_range': 0.3,
        'horizontal_flip': True,
        'vertical_flip': True,
        #'rotation_range': 20, #Int. Degree range for random rotations.
        #'width_shift_range': 0.2,
        #'height_shift_range': 0.2
        }}
    return unfreeze_layers_perc, dropout_top_layers, lr, IDG_augmentation_settings_d
