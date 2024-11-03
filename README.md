# The Future of Cloud Computing

Imagine you could point your phone at the sky and immediatly know how your day is going to turn out. 

## What is this app?

This app is a proof of concept - simply upload an image of a cloudy sky. You will find out if (based on the image you took) if it's going to rain. 

## Extending this concept to Future Work

This concept can be extended to all types of sky - at the moment, clouds (sunny) and rain clouds are recognized by the deep learning model. In the future, people could use the application to take photos of the sky and check if they are in danger for storms and other dangerous weather events. Augmenting the deep learning model that classifies images with more training classes will be relatively simple, using tensorflow.  

## Theme of Exploration

This project has allowed me to learn tensorflow and extend my knowledge of deep learning. It has also introduced me to streamlit, one of my new favourite technologies! :-) 

## Running this yourself

### Set Up

Firstly, you need to install all the relevant tools using pip:

`pip install streamlit` \
`pip install keras` \
`pip install tensorflow` \
`pip install tensorflow-gpu` \
`pip install opencv-python` \
`pip install numpy` \
`pip install matplotlib` 

### Building and Training the Deep Learning Model

Then, run the [Cloud Classifier](CloudClassifier.ipynb), which builds the deep learning model and trains it. You only have to run the first ten code blocks (the remaining two are for testing that everything ran correctly). I would recommend running the .ipynb file using Google Collab.

### Running the website

In your terminal (making sure that the `trained_model.keras` file is in the same file as `website.py`), enter the following command:

`streamlit run website.py`

Please note that this app was built on a Mac-OS based computer.

### References and Acknowledgements

- 
