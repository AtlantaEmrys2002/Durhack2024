# The Future of Cloud Computing

Imagine you could point your phone at the sky and immediatly know how your day is going to turn out. 

## What is this app?

This app is a proof of concept - simply upload an image of a cloudy sky. You will find out if (based on the image you took) if it's going to rain. 

## Extending this concept to Future Work

This concept can be extended to all types of sky - at the moment, clouds (sunny) and rain clouds are recognized by the deep learning model. In the future, people could use the application to take photos of the sky and check if they are in danger for storms and other dangerous weather events. Augmenting the deep learning model that classifies images with more training classes will be relatively simple, using tensorflow.  

## Theme of Exploration

This project has allowed me to learn tensorflow and extend my knowledge of deep learning. It has also introduced me to streamlit (I had never heard of it till I got the MLH email earlier this week), one of my new favourite technologies! :-) 

## Running this yourself

### Set Up

Firstly, you need to install all the relevant tools using pip:

`pip install streamlit` \
`pip install keras` \
`pip install tensorflow` \
`pip install tensorflow-gpu` \
`pip install opencv-python` \
`pip install numpy` \
`pip install matplotlib` \
`pip install kagglehub` \

### Building and Training the Deep Learning Model

Then, run the [Cloud Classifier](CloudClassifier.ipynb), which builds the deep learning model and trains it. You only have to run the first ten code blocks (the remaining two are for testing that everything ran correctly). I would recommend running the .ipynb file using Google Collab.

### Running the website

In your terminal (making sure that the `trained_model.keras` file is in the same file as `website.py`), enter the following command:

`streamlit run website.py`

Please note that this app was built on a Mac-OS based computer.

### References and Acknowledgements

#### Acknowledgements

[Cloud Images Dataset Used to Train Model](https://www.kaggle.com/datasets/nuttidalapthanachai/cloud-image-dataset) - created by 
Nuttida Lapthanachai. 

[Tutorial on Building CNN Deep Learning Models](https://medium.com/@sssspppp/image-classification-using-cnn-0fad8367acfd)

[Streamlit App Framework](https://streamlit.io/) - used the file uploader and framework to build the front end of the application.


#### References

- [Markdown File Links](https://stackoverflow.com/questions/7653483/github-relative-link-in-markdown-file)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Markdown Text Alignment](https://stackoverflow.com/questions/14051715/markdown-native-text-alignment)
- [Emoji Shorthand GitHub Advice](https://github.com/ikatyang/emoji-cheat-sheet)
- [Python Documentation on Temporary Files](https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile)
- [Streamlit Delta Drive](https://stackoverflow.com/questions/74423171/streamlit-image-file-upload-to-deta-drive)
- [opencv Colour Conversions](https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html)
- [Converting RGB image to numpy Array](https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array)
- [tensorflow js converter (NOT USED)](https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/demo/mobilenet/index.js)
- [keras and Serving TensorFlow model](https://www.tensorflow.org/tfx/serving/serving_basic?_gl=1*10sx45e*_up*MQ..*_ga*Mjk0OTY1NTcxLjE3MzA1NzY1NTY.*_ga_W0YLR4190T*MTczMDU3NjU1NS4xLjAuMTczMDU3NjU1NS4wLjAuMA..#load_exported_model_with_standard_tensorflow_modelserver)
- [Guide to Deploying Deep Learning Models](https://medium.com/@maheshkkumar/a-guide-to-deploying-machine-deep-learning-model-s-in-production-e497fd4b734a)
- [Importing, Loading, and Saving Tensor Flow Models](https://www.tensorflow.org/js/tutorials/conversion/import_keras?_gl=1*15j7e7u*_up*MQ..*_ga*OTA0OTkxMDI3LjE3MzA1NzYyNjM.*_ga_W0YLR4190T*MTczMDU3NjI2My4xLjAuMTczMDU3NjI2My4wLjAuMA..)
- [Guide to using tensorflowjs](https://medium.com/@mandava807/importing-a-keras-model-into-tensorflow-js-b09600a95e40)
- [Running Streamlit Code in Google Collab (NOT USED)](https://medium.com/@yash.kavaiya3/running-streamlit-code-in-google-colab-involves-a-few-steps-c43ea0e8c0d9)
- [Downloading Files from Google Collab](https://stackoverflow.com/questions/50453428/how-do-i-download-multiple-files-or-an-entire-folder-from-google-colab)
- [Dealing with Overfitting and Underfitting Tensorflow Documentation](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
- [Tensorflow Documentation Tutorials](https://www.tensorflow.org/tutorials?_gl=1*12uygoi*_up*MQ..*_ga*MTQ0NzQ0MzQxNi4xNzMwNTczMTQ0*_ga_W0YLR4190T*MTczMDYyNjI3Mi4yLjAuMTczMDYyNjI3Mi4wLjAuMA..)
- [Tensorflow API Documentation (MULTIPLE PAGES USED)](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
- [OpenCV Tutorial](https://colab.research.google.com/github/YoniChechik/AI_is_Math/blob/master/c_01_intro_to_CV_and_Python/OpenCV_tutorial.ipynb)
- [Copying Google Collab file to Google Drive](https://stackoverflow.com/questions/59710439/google-colab-and-google-drive-copy-file-from-colab-to-google-drive)
- [shutil Questions](https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth)
- [scikitlearn Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Python Try-Except](https://www.w3schools.com/python/python_try_except.asp)
- [if/else List Comprehension](https://stackoverflow.com/questions/4260280/if-else-in-a-list-comprehension)
- [Maintaining Generated Files in Google Collab](https://juicefs.com/en/blog/usage-tips/colab-persist-data#:~:text=Persist%20data%20in%20Colab%20using%20Google%20Drive&text=As%20shown%20in%20the%20figure,Google%20Drive%20when%20used%20again.)
- [Reading and Displaying Image with OpenCV](https://learnopencv.com/read-display-and-write-an-image-using-opencv/)
- [Modifying Python Strings](https://www.w3schools.com/python/python_strings_modify.asp)
- [Moving Files in Python](https://stackoverflow.com/questions/8858008/how-do-i-move-a-file-in-python)
- [Moving Files in Python 2](https://www.learndatasci.com/solutions/python-move-file/)
- [Reading json File](https://stackoverflow.com/questions/20199126/reading-json-from-a-file)
- [Navigating Directories from Python](https://builtin.com/data-science/python-list-files-in-directory)
- [Reading json files](https://stackoverflow.com/questions/20199126/reading-json-from-a-file)
- [Python Dictionaries](https://www.w3schools.com/python/ref_dictionary_get.asp)

