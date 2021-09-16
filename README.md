# Google Colab link:
Neural Network link:
https://colab.research.google.com/drive/11VI_GEIIs2ScDfJcvj5xh2yjAbm92N2P#scrollTo=0vBl9dcWc97E

# LSTM_Music_Genre_Classification
A Deep Learning algorithm for music genre classification. The network was trained on the GTZAN dataset

# Feature Extraction
The GTZAN dataset is used to extract features for each of the song in it. Mel Frequency Cepstral Coefficients (MFCCs) was chosen as the prefered extracted features as it immitates the way the human ear works and is one of the most commonly used features in audio signal processing. 

- Visualization of spectogram and MFCCs

  ![spectogram](https://user-images.githubusercontent.com/58198596/133607202-d45efc3b-2e06-48c9-b7a4-0d10bc2c28cc.png)
  
  ![mfcc](https://user-images.githubusercontent.com/58198596/133607199-afca9a28-54d6-48b4-ae05-311a908bd9a0.png)


Due to the small amount of data contained in the dataset data augmentation is performed by splitting each song into 10 parts and using them as new songs. This increased the accuracy of the model by more than 30%
