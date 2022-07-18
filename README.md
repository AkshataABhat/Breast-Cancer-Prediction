Breast Cancer Histology Image Classification Using DenseNet

Introduction
		Breast cancer has emerged as a significant public health concern due to its highest morbidity rate among all cancers worldwide.The most prevalent subtype of breast cancer is invasive ductal carcinoma (IDC). Early diagnosis in breast cancer can increase the chances of successful treatment and survival. When grading the aggressiveness of a whole mount sample, pathologists frequently concentrate on the areas that contain the IDC. As a result, identifying the precise IDC regions inside of a whole mount slide is one of the typical pre-processing processes for automatic aggressiveness rating. Pathologists can take a significant amount of time to evaluate whole mount slides for a given patient. Computer-aided Systems for diagnosis help this process run more efficiently and more inexpensively. Traditional classification strategies rely on feature extraction techniques created for a particular problem based on domain expertise. Deep learning techniques have proven to be a  key alternative to feature-based approaches for overcoming their numerous drawbacks.
		A fundamental and crucial computer vision task is image classification. Convolutional neural networks (CNNs) have taken over as the main machine learning strategy for computer vision recognition in recent years. The original LeNet5 had 5 layers, VGG had 19 layers, and Residual Networks (ResNet) have crossed the 100-layer threshold. Training these models present challenges such as too many parameters, gradient vanishing, and complicated training regimes to prevent overfitting. In comparison to models like VGG and Resnet, Dense Convolutional Network (DenseNet) has dense connection and is superior to other models. Direct connections from any layer to all subsequent layers distinguish the DenseNet model from other CNNs and potentially enhance the information. The feature maps from the previous layer are used to generate new feature maps in the following layer. Each feature map in this second layer is a combination of the feature maps in the previous layer. And the value of the feature map in the second layer is determined at each given pixel by multiplying each feature in the first layer by a convolution kernel, with a distinct kernel for each feature map in the first layer. The responses are then added to a bias term and modified using a simple non-linear technique.
In this work we have used applied densenet architecture to learn features using patches created from whole mount slide images to then use these features to classify IDC vs Non-IDC.

Dataset
The original dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens scanned at 40x. From that, 277,524 patches of size 50 x 50 were extracted (198,738 IDC negative and 78,786 IDC positive). Each patch’s file name is of the format: uxXyYclassC.png — > example 10253idx5x1351y1101class0.png . Where u is the patient ID (10253idx5), X is the x-coordinate of where this patch was cropped from, Y is the y-coordinate of where this patch was cropped from, and C indicates the class where 0 is non-IDC and 1 is IDC. The images in the dataset are small image patches extracted from larger digital pathology scans. Small image patches extracted from larger digital pathology scans are used with DenseNet Block to detect IDC cancer. Link to the dataset: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images



	
Methods
In contrast to models like VGG and Resnet , Dense Convolutional Network (DenseNet) has dense connectivity. DenseNet can reduce the number of parameters, improve feature map propagation, and solve the vanishing-gradient problem. The images in the dataset are resized into 50 x 50 images and have exactly 3 input channels(RGB).

Pre-processing
We divide the data into 3 parts: training data, testing data and validation data. Data is divided into 3 sub folders: train_seg, test_seg and val_seg. The data was split as follows:
Train Data Size= 0.7*Total_Data
Validation Data Size= 0.21* Total_Data
Test Data Size= 0.09 * Total_Data
Each of these 3 sub-folders contain 2 folders idc-minus and idc-plus respectively. 

Data Augmentation 
We performed data augmentation to make the model more robust using the ImageDataGenerator class in keras to increase the size of the dataset. Each Image is rotated through 45 degree angles.
The base model used in this project is the DenseNet121. The architecture of the model is shown in Fig 1.

					Fig 1: Densenet Model

We define a sequential model. After adding the base model, we add a flattening layer to convert all the resultant 2-Dimensional arrays from pooled feature maps into a single long continuous linear feature vector. Batch Normalization is applied to normalize output of previous layers.

				Fig 2:  Architecture of the model used 
 
	

Add a section called training where you write about the kaggle training gpu used, approx training time, no of gpu used, memory size, and also the hyperparameter values, etc. These things help other people implement your work and reproduce them. It also helps you later if you want to check what values you used for a particular training regime.

Training:


Results:
	The results from the training are as follows. The accuracy and loss of the model is plotted in Fig 3.
  ![image](https://user-images.githubusercontent.com/74967139/179501455-dcfd154c-ccff-408c-937e-534d4b30aa91.png)


					Fig 3

Report of test data is shown in Fig 4.
![image](https://user-images.githubusercontent.com/74967139/179501419-2f9b2e67-4737-4100-9979-ef7247cb5d96.png)


					Fig 4

