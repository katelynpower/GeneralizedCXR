# GeneralizedCXR

### Overview

This project investigates the nature of generalizability in AI through training a CNN to predict pleural effusion from chest x-ray images with various iterations of dataset diversity, target data incorporation, and finetuning methods to understand how discrimination and calibration respond to each. The data come from three distinct datasets: VinBigData from Vietnam and NIH  and Stanford CheXpert from the United States. The US datasets serve as the source training data, with NIH as the primary and Standford as supplemental data, and VinBigData serves as the primary target dataset that the model will need to generalize to. The iterations of training will be the following:

1.	Target only training: the CNN is trained on only VinBigData
2.	Source only training: the CNN is trained on only NIH
3.	Mixed training: the CNN is trained on NIH and Stanford CheXpert (50/50)
4.	Source + Target training: the CNN is trained on NIH and VinBigData with varying proportions (5, 10, 25, 50 percent)
5.	Finetuning: the CNN is first trained on NIH and fine-tuned on VinBigData

In each of these iterations, the held-out test set comes from VinBigData. Throughout, the model architecture, image sizes, x-ray projection, training set size, and training set pleural effusion prevalence remain constant as applicable. Additionally, if time allows, I may also explore federated fine-tuning, which is particularly relevant for this clinical, cross-site context, and the effect of varying prevalence.

### Default Modeling Parameters

| parameter | value |
| --------- | ----- |
| architecture | **DenseNet121** with all layers unfrozen, followed by GlobalAveragePooling2D(), Dropout(0.3), Dense(64, activation="relu"), Dropout(0.2), Dense(1, activation="sigmoid", dtype="float32", bias_initializer=Constant(np.log(prevalence / (1 - prevalence)))) layers
| image size | 224 x 224
| x-ray projection type | posterior-anterior (PA)
| pleural effusion prevalence | 0.069 (lowest across the datasets)

