# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 21:33:19 2022

@author: JAGADEESH
"""
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np 
import skimage.io
import warnings
import mutagen
from mutagen.wave import WAVE
import cv2
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import IPython.display as ipd
from tqdm import tqdm
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import backend as K

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.io import wavfile
import noisereduce as nr
import joblib
import pickle

from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,classification_report
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from pathlib import Path
from sklearn.model_selection import train_test_split


model = keras.models.load_model('C:/AIML-Project/data/Final_basemodel')
model1 = keras.models.load_model('C:/AIML-Project/data/Final_model')


def features_extractor(audio, sample_rate):
    
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features


st.set_page_config(layout="wide")
st.title("Enter path for data input ")

breathing_deep=st.text_input("breathing-deep recording path",'C:/AIML-Project/data/Model_testing/breathing-deep.wav')
breathing_shallow=st.text_input("breathing-shallow recording path",'C:/AIML-Project/data/Model_testing/breathing-shallow.wav')
cough_heavy=st.text_input("cough_heavy recording path",'C:/AIML-Project/data/Model_testing/cough-heavy.wav')
cough_shallow=st.text_input("cough_shallow recording path",'C:/AIML-Project/data/Model_testing/cough-shallow.wav')

fever=st.selectbox("fever", ["1"
 ,"0"])


cough=st.selectbox("Cough", ["1"
 ,"0"])


loss_of_smell=st.selectbox("loss_of_smell", ["1"
 ,"0"])


ftg=st.selectbox("fatigue", ["1"
 ,"0"])


bd=st.selectbox("breathing difficulty", ["1"
 ,"0"])


fever=int(fever)
cough=int(cough)
loss_of_smell=int(loss_of_smell)
ftg=int(ftg)
bd=int(bd)



val = st.radio( "select choice", ( 'No cal','submit'))

if val == 'submit':
    
    
    mfcc_features=[]
    Health_ail=[]
    
    # breathing_deep='C:/AIML-Project/data/Model_testing/breathing-deep.wav'

    audio, sample_rate = librosa.load(breathing_deep, res_type='kaiser_fast') 
    audio1, sample_rate1 = librosa.load(breathing_shallow, res_type='kaiser_fast')
    audio2, sample_rate2 = librosa.load(cough_heavy, res_type='kaiser_fast')
    audio3, sample_rate3 = librosa.load(cough_shallow, res_type='kaiser_fast')

    # final_class_labels=row["cmp_test_status"]


    data=np.concatenate([features_extractor(audio, sample_rate),features_extractor(audio1, sample_rate1),features_extractor(audio2, sample_rate2),features_extractor(audio3, sample_rate3)])
    mfcc_features.append([data])
    Health_ail.append([fever,cough,loss_of_smell,ftg,bd])

    data_df=pd.DataFrame(mfcc_features,columns=['feature'])
    Health_ail_df=pd.DataFrame(Health_ail,columns=['fever','cough','loss_of_smell','ftg','bd'])

    val_df=np.array(data_df['feature'].tolist())    
    predictions = model.predict(val_df)

    Health_ail_df['predictions']=predictions
    
        
    final_predictions = model1.predict(Health_ail_df)
    labels = (predictions > 0.5).astype(np.int)
    
    if labels==1:
    
        st.error('COVID', icon="🚨")
        
    else:

        st.success('Healthy', icon="✅")

else:

    st.write("No Calc Running")





