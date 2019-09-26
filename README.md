Classification of Job Announcements through Contextual Understanding using RNN
====================

Introduction
----
In this study, the job announcement image is input and the name of the company that issued job announcement is output.
So this study consists of three steps.

1. Read the image of the job announcement and convert it to character.
2. Transform characters into appropriate data types through data preprocessing.
3. After training character data with RNN, finally output the company name.

With this process, this study aims at helping to manage the employment announcements efficiently.

About Data and codes
----
As mentioned before, image of job announcement is converted to character by following code;
` imageRecognition.py `
This code use Google Cloud Vision API, so if you want to run this code, you should set appropriate environment first.
For further information about environment setting, please check [here.](https://cloud.google.com/vision/docs/)

The next step is training data using RNN. There are pre-processed text data and sequence data in text folder; And the data
are trained using RNN by following code;
` rnn_train_test.py`
There are several steps for data to be processed appropriately. After training, you can test the model by changing test file path.


References
----

•	Hun Kim’s Class
-	https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-2-char-seq-rnn.py
-	https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-4-xor_tensorboard.py

•	GitHub
-	https://github.com/golbin/TensorFlow-Tutorials/blob/master/10%20-%20RNN/02%20-%20Autocomplete.py

•	Tensorflow
-	https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
-	https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
-	https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer
