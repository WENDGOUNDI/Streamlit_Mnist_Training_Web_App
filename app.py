import streamlit as st
import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Softmax
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.title("MNIST DATASET TRAINING: CUSTOMIZABLE NEURAL NETWORK")

number_neurons = st.sidebar.slider("Number of Neurons in the Hidden Layer:", 1, 150)
number_epochs = st.sidebar.slider("Number of epochs", 1, 100)
number_batch_size = st.sidebar.slider("Batch Size", 1, 256)
activation_func = st.sidebar.text_input("Activation function. Relu is set as default")

"The number of neurons is " + str(number_neurons)
"The number of epochs is " + str(number_epochs)
activation_func = 'relu'
"The Activation Function is " + str(activation_func)

if st.button('Train the model'):
    "We are Training"

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train/255.0
    X_test = X_test/255.0
    
    model = Sequential()
    model.add(InputLayer((28,28)))
    model.add(Flatten())
    model.add(Dense(number_neurons, activation_func))
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    "Build Model is"
    tensorflow.keras.utils.plot_model(model,to_file="model.png",)
    
    image = Image.open("C:/Users/savadogo_abdoul/Desktop/Learn/image_classifcation_streamlit/model.png")
    st.image(image)  #st.image(image, caption='Sunrise by the mountains')
    #st.spinner(text="Training In Progress...")
    with st.spinner('Training In Progress...'):
    
        save_model = ModelCheckpoint("model", save_best_only=True)
        history = tensorflow.keras.callbacks.CSVLogger('history.csv', separator=',') 
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=number_epochs, batch_size=number_batch_size, 
                  callbacks=[save_model, history])
    st.success('Done!')
    
    eval_model = model.evaluate(X_test, y_test)
    test_acc = round(eval_model[1]*100, 2)
    test_loss = round(eval_model[0], 2)
    "Training Completed"
    f"Testing Accuracy: {test_acc}%"
    f"Testing Loss: {test_loss}"
    
    
if st.button('Plot Model Performance'):
    
    "We are Plotting the model Performance"
    
    history = pd.read_csv("history.csv")
    fig = plt.figure()
    plt.plot(history['epoch'], history['accuracy'])
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title("Model accuracy vs epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(['Train', 'Val'])
    st.pyplot(fig)
