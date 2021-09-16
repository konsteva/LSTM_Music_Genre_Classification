# Google Colab link:
Neural Network link:
https://colab.research.google.com/drive/11VI_GEIIs2ScDfJcvj5xh2yjAbm92N2P#scrollTo=0vBl9dcWc97E

# LSTM_Music_Genre_Classification
A Deep Learning algorithm for music genre classification. The network was trained on the GTZAN dataset. The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

Link to the dataset:
https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

# Feature Extraction
The GTZAN dataset is used to extract features for each of the song in it. The script iterates through the folders of the songs and extracts the Mel Frequency Cepstral Coefficients (MFCCs) for each one. MFCCs was chosen as the prefered extracted features as it immitates the way the human ear works and is one of the most commonly used features in audio signal processing. The features, as well as the labels of the song and their mapping to genre is temporarily stored into a dictionary and afterwards they are saved into a json file for ease in processing and training later on.

Due to the small amount of data contained in the dataset, data augmentation is performed by splitting each song into 10 parts and using them as new songs. This increased the accuracy of the model by more than 30%

- Visualization of spectogram and MFCCs

  ![spectogram](https://user-images.githubusercontent.com/58198596/133607202-d45efc3b-2e06-48c9-b7a4-0d10bc2c28cc.png)
  
  ![mfcc](https://user-images.githubusercontent.com/58198596/133607199-afca9a28-54d6-48b4-ae05-311a908bd9a0.png)
  
  # Neural Network
  
  After loading the data, the MFCCs and the labels are put into numpy arrays and then split into training (60%), test(20%) and validation(20%) sets.
  
  A Recurrent Neural Network (RNN) was built using Keras. Since the data can be perceived as a time-series using a Long Short Term Memory architecture seemed intuitive. The Network has depth of 3 (hidden) layers of decreasing width and an output layer of 10 nodes that correspond to each music genre. To minimize overfitting L1, L2 regularization and Dropout were tried but Dropout had the best results so it's the one used in the finalized network. The number of epochs set for the network training was 150 and the batch size was set to 20. The layers used the ReLU activation function, the model optimizer was Adam and for the loss Cross Entropy function was used. After training the network the model was fit over the data and also saved as an h5 file for later use.
  
  # Results, Visualization and Comments
  
  After the 150 epochs the final results are:
  
      Epoch 150/150
      300/300 [==============================] - 20s 66ms/step - loss: 0.0520 - accuracy: 0.9882 - val_loss: 1.3496 - val_accuracy: 0.8214
      63/63 - 1s - loss: 1.2672 - accuracy: 0.8225

      Test accuracy: 0.8224999904632568
      
  The final accuracy of the model is 82%. Below, the accuracy and loss graphs can be seen:
      
![accuracy](https://user-images.githubusercontent.com/58198596/133620949-5c534090-b2df-4304-beab-609c9bf8f545.png)

![error](https://user-images.githubusercontent.com/58198596/133620939-b98f9feb-14d7-41a5-ba71-81bcd8701704.png)

- Looking at the accuracy graph it is clear that after about epoch 15 the model starts overfitting and by the end of 150 epochs it is heavily overfitting. It is also interresting that in the first epochs the validation accuracy is greater than the train accuracy. This is probably caused because the training set has less information availliable due to the Dropout rate and as a result it makes the prediction for the train set harder than the prediction of the validation set in the first few epochs. 

- Looking at the loss graph we notice that the loss is steep decreasing until about epoch 15 and it starts slowly and steadily increasing at around epoch 20 until the end. This contradiction with the high accuracy implies once again that the model starts overfitting at about epoch 15. This increase in the loss might also be due to the selected Cross Entropy loss function as it penalizes wrong predictions more than it rewards correct. Thus wrong predictions would cause a small decrease in accuracy but a big increase in the loss. Another interesting thing we can observe if we compare the validation and the train loss, is that the validation loss is less stable locally, meaning that even though the trend is obvious and relatively stable, there are greater differences in the loss value between successive epochs compared to the difference in loss value in the training set. This is probably caused due to the small batch size number. We can expect to observe smaller differences if we increase the batch size.

# Comparison with other models

