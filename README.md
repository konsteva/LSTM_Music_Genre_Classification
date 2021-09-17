# LSTM_Music_Genre_Classification
A Deep Learning algorithm for music genre classification. The network was trained on the GTZAN dataset. The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

Link to the dataset:
https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

# Feature Extraction
The GTZAN dataset is used to extract features for each of the song in it. The script iterates through the folders of the songs and extracts the Mel Frequency Cepstral Coefficients (MFCCs) for each one. MFCCs was chosen as the prefered extracted features as it immitates the way the human ear works and is one of the most commonly used features in audio signal processing. The features, as well as the labels of the song and their mapping to genre is temporarily stored into a dictionary and afterwards they are saved into a json file for ease in processing and training later on.

Due to the small amount of data contained in the dataset, data augmentation is performed by splitting each song into 10 parts and using them as new songs. This increased the accuracy of the model by more than 30%

- Visualization of spectogram and MFCCs

  ![spectogram](https://user-images.githubusercontent.com/58198596/133607202-d45efc3b-2e06-48c9-b7a4-0d10bc2c28cc.png) 		![mfcc](https://user-images.githubusercontent.com/58198596/133607199-afca9a28-54d6-48b4-ae05-311a908bd9a0.png)
  
# Neural Network
  
After loading the data, the MFCCs and the labels are put into numpy arrays and then split into training (60%), test(20%) and validation(20%) sets.
  
A Recurrent Neural Network (RNN) was built using Keras. Since the data can be perceived as a time-series using a Long Short Term Memory architecture seemed intuitive. The Network has depth of 3 (hidden) layers of decreasing width and an output layer of 10 nodes that correspond to each music genre. To minimize overfitting L1, L2 regularization and Dropout were tried but Dropout had the best results so it's the one used in the finalized network. The network is trained for 150 epochs and the batch size was set to 20. The layers used the ReLU activation function, the model optimizer was Adam and for the loss Cross Entropy function was used. After training the network the model was fit over the data and also saved as an h5 file for later use.
  
# Results, Visualization and Comments
  
After 150 epochs the final results are:
  
	Epoch 150/150
	300/300 [==============================] - 9s 30ms/step - loss: 0.3915 - accuracy: 0.8803 - val_loss: 1.0546 - val_accuracy: 0.7449
	63/63 - 1s - loss: 1.0176 - accuracy: 0.7675

	Test accuracy: 0.7674999833106995


The final accuracy of the model is ~77%. Below, the accuracy and loss graphs can be seen:
  
![lstm2 err](https://user-images.githubusercontent.com/58198596/133780555-6722ec26-10c1-4c4f-852b-0386e0aefb6f.png) ![lstm2 acc](https://user-images.githubusercontent.com/58198596/133780564-597fbf35-cec4-41e2-85d5-7990aa34b703.png)

            

- Looking at the accuracy graph it is clear that after about epoch 20 the model starts overfitting and by the end of 150 epochs it is significantly overfitting. It is also interresting that in the first epochs the validation accuracy is greater than the train accuracy. This is probably caused because the training set has less information availliable due to the Dropout rate and as a result it makes the prediction for the train set harder than the prediction of the validation set in the first few epochs. 

- Looking at the loss graph we notice that the loss is steep decreasing until about epoch 20 where it stops and is relatively the same until the end (with very minor increase). This implies once again that the model starts overfitting at about epoch 20. This increase in the loss might also be due to the selected Cross Entropy loss function as it penalizes wrong predictions more than it rewards correct. Thus wrong predictions would cause a small decrease in accuracy but a big increase in the loss.


# Comparison with other models
- Multilayer Perceptron (MLP):

Model:

      model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


Results:

      Epoch 150/150
      300/300 [==============================] - 1s 4ms/step - loss: 0.2354 - accuracy: 0.9421 - val_loss: 4.8255 - val_accuracy: 0.6013
      63/63 - 0s - loss: 4.4274 - accuracy: 0.6005

      Test accuracy: 0.6004999876022339
   
  
![mlp acc](https://user-images.githubusercontent.com/58198596/133784877-a35d667c-e31a-46ad-beaa-b1035bc30615.png) ![mlp err](https://user-images.githubusercontent.com/58198596/133784871-71b5a958-deb8-417e-ab0b-47524b371d3c.png)


- Convolutional Neural Network (CNN):

Model:

      model = keras.Sequential()
	    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
	    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
	    model.add(keras.layers.BatchNormalization())
	    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
	    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
	    model.add(keras.layers.BatchNormalization())
	    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
	    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
	    model.add(keras.layers.BatchNormalization())
	    model.add(keras.layers.Flatten())
	    model.add(keras.layers.Dense(64, activation='relu'))
	    model.add(keras.layers.Dropout(0.3))
	    model.add(keras.layers.Dense(10, activation='softmax'))


Results:

      Epoch 150/150
      300/300 [==============================] - 2s 7ms/step - loss: 0.0504 - accuracy: 0.9830 - val_loss: 2.0805 - val_accuracy: 0.7354
      63/63 - 0s - loss: 2.0810 - accuracy: 0.7340

      Test accuracy: 0.734000027179718
      

![cnn acc](https://user-images.githubusercontent.com/58198596/133783726-73cc1ef1-628a-4d4c-88dd-0c45c0d4d987.png) ![cnn err](https://user-images.githubusercontent.com/58198596/133783719-ec68ddc8-1023-4557-b4b9-04db975e5c23.png)


- Recurrent Neural Network (RNN) - LSTM

Model:

	model = keras.Sequential()

    	model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
	model.add(LSTM(128))
    	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
    	model.add(Dropout(0.3))
	model.add(Dense(32, activation='relu'))
    	model.add(Dropout(0.3))
	model.add(Dense(10, activation='softmax'))
      
      
Results:

	Epoch 150/150
      	300/300 [==============================] - 20s 66ms/step - loss: 0.0520 - accuracy: 0.9882 - val_loss: 1.3496 - val_accuracy: 0.8214
      	63/63 - 1s - loss: 1.2672 - accuracy: 0.8225

      	Test accuracy: 0.8224999904632568
  
  
![accuracy](https://user-images.githubusercontent.com/58198596/133620949-5c534090-b2df-4304-beab-609c9bf8f545.png) ![error](https://user-images.githubusercontent.com/58198596/133620939-b98f9feb-14d7-41a5-ba71-81bcd8701704.png)



# Possible ways of improving the model
- Adding more features other than MFCCs, eg. tempo, speechiness, loudness etc.
- Performing early stopping to deal with overfitting
- Using a different optimizer and loss function
- Performing more data augmentation
- Using computer vision to train the model based on the spectogram and/or MFCCs (?)
