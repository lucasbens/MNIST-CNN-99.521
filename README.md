# Kaggle Digit Recognizer (MNIST) Project - README

This is the README file for the [Kaggle MNIST Digit Recognizer project](https://www.kaggle.com/c/digit-recognizer), in which I achieved an [accuracy of 99.521%](https://www.kaggle.com/code/lucasbensaid/digit-recognizer-top-8-99-521) on the leaderboard (top 8%).

## Project Overview

The Kaggle Digit Recognizer (MNIST) project is a classification problem in which we are asked to predict the handwritten digits from the MNIST dataset.
* `train.csv` and `test.csv`: The original data provided by Kaggle are available [here](https://www.kaggle.com/competitions/digit-recognizer/data) .


## Approach

My approach to solving this problem involved several steps:

1. **Import data:** Train dataset: 42,000 labeled images. Test set: 28,000 unlabeled images.

2. **Data Preprocessing:** I performed data preprocessing by normalizing pixel values and reshaping the data to have a 3D tensor shape.

3. **Model Architecture:** I implemented a Convolutional Neural Network (CNN) model using Keras and TensorFlow. The model consisted of several convolutional layers, followed by max pooling layers, dropout layers, a flatten layer, and finally a dense layer with softmax activation function.

4. **Model Training:** I trained the model using the RMSprop optimizer and a categorical cross-entropy loss function for 30 epochs, with a batch size of 64. I also use EarlyStopping callback and learning rates annealer to accelerate training.

5. **Model Evaluation:** In this notebook, I tested:

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; - **RandomForest** submission achieve **0.98496** in the leaderboard **(top 47%)**

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; - **CNN** submission achieve **0.99339** in the leaderboard **(top 17%)**

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; - **[Ensemble 11 CNNs](https://www.kaggle.com/code/lucasbensaid/digit-recognizer-top-8-99-521)** submission achieve **[0.99517](https://www.kaggle.com/code/lucasbensaid/digit-recognizer-top-8-99-521)** in the leaderboard **([top 8%](https://www.kaggle.com/code/lucasbensaid/digit-recognizer-top-8-99-521))**



## Files in the Repository

This repository contains the following files:

- `Kaggle_MNIST_Digit_Recognizer.ipynb`: The Jupyter notebook containing our code for data preprocessing, model architecture, training, evaluation, and submission.
- `submission.csv`: The final predictions submitted to the Kaggle leaderboard.


## Dependencies

To run the code in the Jupyter notebook, you will need the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `keras`

You can install these libraries using `pip` or `conda`.

## Conclusion

I was able to achieve an [accuracy of 99.521%](https://www.kaggle.com/code/lucasbensaid/digit-recognizer-top-8-99-521) on the Kaggle MNIST Digit Recognizer leaderboard by using an ensemble of 11 Convolutional Neural Network (CNN) model and training it using RMSprop optimizer and a categorical cross-entropy loss function for 30 epochs, with a batch size of 64. I also use EarlyStopping callback and learning rates annealer to accelerate training.

I hope that my code and methods will be useful for others working on this problem or similar image classification problems in the future.
