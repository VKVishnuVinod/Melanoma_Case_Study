# Melanoma Skin Cancer Detection

## Abstract

Cancer encompasses over 200 distinct types, with melanoma recognized as the deadliest form of skin cancer. The diagnostic process for melanoma typically begins with clinical screening, followed by dermoscopic imaging and histopathological examination. Early detection of melanoma is crucial, as it greatly increases the likelihood of successful treatment. The first step in diagnosing melanoma involves a visual inspection of the affected skin area. Dermatologists use high-resolution dermatoscopic images of skin lesions, achieving diagnostic accuracy rates of 65% to 80% without additional technical support. With further evaluation by oncologists and detailed dermatoscopic image analysis, the overall accuracy for melanoma diagnosis can be improved to between 75% and 84%. This project aims to develop an automated classification system that uses image processing techniques to identify skin cancer from images of skin lesions.

## Problem statement

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

## Table of Contents

- [General Info](#general-information)
- [Model Architecture](#model-architecture)
- [Model Summary](#model-summary)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Acknowledgements](#acknowledgements)
- [Collaborators](#collaborators)

<!-- You can include any other section that is pertinent to your problem -->

## General Information

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The aim of this task is to assign a specific class label to a particular type of skin cancer.

![image](https://github.com/user-attachments/assets/ffcae2d0-c1f3-4a74-afe5-edfed0f3442d)

![image](https://github.com/user-attachments/assets/76f1a8ef-6d1c-4b8c-b971-5c30d604468f)


## Model Architecture


1. **Data Augmentation** : The augmentation_data variable applies various augmentation techniques to the training data. By introducing random transformations like rotation, scaling, and flipping, data augmentation increases the diversity of the training dataset, which enhances the model's ability to generalize.
2. **Normalization** : A Rescaling(1./255) layer is used to normalize pixel values, scaling them to a range between 0 and 1. Normalization stabilizes training and speeds up convergence.

3. **Convolutional Layers** : Three Conv2D layers are used in sequence, each followed by a rectified linear unit (ReLU) activation function to introduce non-linearity. Using padding='same' maintains the spatial dimensions of the feature maps after each convolution. The numbers (32, 64, 128) represent the number of filters in each layer, which sets the depth of the feature maps.

4. **Pooling Layers**: Each convolutional layer is followed by a max-pooling layer (MaxPooling2D), which downsamples the feature maps, reducing spatial dimensions while preserving key information. Max-pooling reduces computational complexity and helps control overfitting.

5. **Dropout Layer**: A Dropout layer with a 0.2 dropout rate is included after the final max-pooling layer to prevent overfitting by randomly disabling a portion of neurons during training.

6. **Flatten Layer**: The Flatten layer converts the 2D feature maps into a 1D vector, making it compatible for input into fully connected layers.

7. **Fully Connected Layers**: Two dense layers (Dense) with ReLU activation follow the flattening layer. The first dense layer has 256 neurons, while the final dense layer provides the classification probabilities.
8. **Output Layer**: The number of neurons in the output layer matches the target_labels variable, representing the number of classes. No activation function is specified here, as the loss function will handle it during training.

9. **Model Compilation** : The model is compiled using the Adam optimizer (optimizer='adam') and Sparse Categorical Crossentropy loss (loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)), ideal for multi-class classification. Accuracy (metrics=['accuracy']) is selected as the evaluation metric.
    
11. **Training**: The model is trained with the fit method for a specified number of epochs (epochs=50). Two callbacks, ModelCheckpoint and EarlyStopping, monitor validation accuracy. ModelCheckpoint saves the model with the best validation accuracy, while EarlyStopping halts training if validation accuracy does not improve after a specified patience period (patience=5). These callbacks help avoid overfitting and ensure optimal model performance.

## Model Summary

![image](https://github.com/user-attachments/assets/df96e198-b214-49aa-8684-cd9d291dd6db)

## Model Evaluation
![image](https://github.com/user-attachments/assets/05271e39-2765-4b83-b2ad-42d621a0b9f8)



## Technologies Used

- [Python](https://www.python.org/) - version 3.11.4
- [Matplotlib](https://matplotlib.org/) - version 3.7.1
- [Numpy](https://numpy.org/) - version 1.26.3
- [Pandas](https://pandas.pydata.org/) - version 2.2.2
- [Seaborn](https://seaborn.pydata.org/) - version 0.13.2
- [Tensorflow](https://www.tensorflow.org/) - version 2.17.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements

- UpGrad tutorials on Convolution Neural Networks (CNNs) on the learning platform

- [Melanoma Skin Cancer](https://www.cancer.org/cancer/melanoma-skin-cancer/about/what-is-melanoma.html)

## Collaborators

Created by [@VKVishnuVinod](https://github.com/VKVishnuVinod)
