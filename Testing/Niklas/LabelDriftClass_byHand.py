#%% Import necessary libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

import tensorflow as tf

import time
from IPython.display import clear_output

#%% Hyperparameters data loading

base_file_path = 'C:/Users/nikoLocal/Documents/Opencampus/Machine_Vision_challenge_data/'
image_path = base_file_path + '/input_train/input_train'

label_csv_name = 'Y_train_eVW9jym.csv'

# load data from csv to pandas dataframe
train_df = pd.read_csv(os.path.join(base_file_path, label_csv_name))

train_df.head()

# Import random Test Dataset

submission_csv_name = 'Y_random_nKwalR1.csv'
submission_image_path = base_file_path + '/input_test_1a4aqAg/input_test'

submission_test_df = pd.read_csv(os.path.join(base_file_path, submission_csv_name))

#ToDo - assign each label a number according to the website so that there is no confusion as to what is what!

#add another column to the dataframe according

dict_numbers = {'GOOD': 0,'Boucle plate':1,'Lift-off blanc':2,'Lift-off noir':3,'Missing':4,'Short circuit MOS':5}
dict_strings = {'GOOD': '0_GOOD','Boucle plate':'1_Flat loop','Lift-off blanc':'2_White lift-off','Lift-off noir':'3_Black lift-off','Missing':'4_Missing','Short circuit MOS':'5_Short circuit MOS'}

label_list = ['0_GOOD','1_Flat loop','2_White lift-off','3_Black lift-off','4_Missing','5_Short circuit MOS']
#label_list.sort()

train_df['LabelNum'] = train_df['Label'].map(dict_numbers)
train_df['LabelStr'] = train_df['Label'].map(dict_strings)

#Process label submission test dataframe

dict_strings_sub = {0: '0_GOOD',1:'1_Flat loop',2:'2_White lift-off',3:'3_Black lift-off',4:'4_Missing',5:'5_Short circuit MOS',6:'6_Drift'}

submission_test_df['LabelStr'] = submission_test_df['Label'].map(dict_strings_sub)

submission_test_df.head()

#%% load pretrained model

Trained_baselineModel = tf.keras.models.load_model(os.path.join(base_file_path, 'BaselineCNN_Model.keras'))

#%%

#import matplotlib
#%matplotlib ipympl
#%matplotlib notebook

import keyboard

i_run_start = 50
stop_index = 53

#make dataframe to store processed data
#    submission_test_df,
#    submission_image_path

#need to install package pyqt5
#matplotlib.use('inline')
#plt.ion()
#plt.ioff()

#important to set the batch index to 0 so that we look at the same data every time!
test_generator_submission.batch_index = i_run_start

i_run = i_run_start
for data in test_generator_submission:

    prediction = model_simple_CNN.predict(data,verbose=0)
    predicted_label = np.argmax(prediction, axis=-1)[0]

    label_certainty = prediction[0][predicted_label]

    #check whether label is predicted with high certainty
    if label_certainty < certainty_threshold:
        predicted_label = 6

    temp_img_generator = np.squeeze(data[0])

    plt.figure()
    plt.imshow(temp_img_generator,cmap='grey')
    plt.title('Pred. Label: {}'.format(label_list_test[predicted_label]))

    plt.pause(0.5)
    #time.sleep(2)

    Input_class = input("Press Enter to continue...")
    #key = keyboard.wait()
    print('Input class: {}'.format(Input_class))

    plt.close()

    #clear output after user input is registered
    clear_output()

    #move to next data point
    i_run = i_run + 1
    #if i_run == len(test_generator_submission):
    if i_run >= stop_index:
        print("{} items processed".format(i_run))
        break
