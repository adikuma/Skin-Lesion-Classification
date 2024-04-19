# Skin Lesion Classification | [Report](https://github.com/adikuma/Skin-Lesion-Classification/raw/main/ADL%20Final%20Report.docx)

## Motivation

The primary motivation for this project is to enhance early skin cancer detection capabilities amongst the general population. Skin cancer is one of the most common cancer types globally, and early detection of this can significantly increase the chances of successful treatment for patients. Traditional methods for skin cancer diagnosis such as visual examinations, biopsy and histopathological analysis, are all effective but costly and time-consuming. By leveraging machine learning technologies to analyze skin lesions, we can provide a faster, more accessible preliminary diagnostic tool. 

## Problem Statement

How might we develop an automated and reliable system to identify the presence or absence of skin cancer in individuals using limited information?

##	Dataset

The dataset utilized is the HAM10000 dataset, sourced from Harvard Dataverse. Selecting the appropriate dataset has a large impact on how the model actually performs, as it fundamentally influences the model's predictive accuracy. The following details about the dataset are as follows:

1.	Images: The dataset comprises 10,015 dermatoscopic images. All images are of different pixel sizes and are predominantly in JPEG format. This format is commonly used in medical image datasets due to its balance of quality and file size.
2.	Demographics: The images are gathered from skin cancer clinics worldwide, including Austria, Australia, and the United States. The patients featured in the dataset do not belong to any specific age group, but are mostly Caucasian in terms of skin type. This lack of diversity will affect the overall data that the model is exposed to, and thereby negatively impact its accuracy.
3.	Metadata: Each image is annotated with details such as the type of skin lesion, the bodily location of the lesion, and the patient's age and gender

## Models

### Capsule Neural Network

The Capsule Neural Network (CapsNet) then receives the pre-processed image tensor as input. 

In the early phases of this research, we initially focused on using a convolutional neural network (CNN). However, upon testing various models and conducting further research, we decided to adopt a more exploratory approach by utilizing the less common CapsNet architecture instead. 

Unlike CNNs, CapsNet processes information using a hierarchical grouping of neurons known as "capsules." These capsules are designed to encode and preserve spatial hierarchies between features, allowing the network to recognize objects and their variations with better generalization. Hence, it would theoretically work better than regular CNNs in tasks such as image classification. (Mensah, Adekoya, Ayidzoe, & Edward, 2022).

The CapsNet then has two outputs:

1.	A tensor that assigns classification scores to each of the 7 types of skin lesions from the dataset, reflecting the probability of each possible diagnosis. 
2.	A feature map output used for visual explanations via Grad-CAM.

The specifics of the network, including the number and size of layers, will be determined during evaluation. 

![FixCaps Model Architecture](https://github.com/adikuma/Skin-Lesion-Classification/blob/main/fixcaps.jpg?raw=true)

### Fusion Model

We have conducted a hypothesis that we can further enhance the accuracy of our CapsNet model if we were to add in the metadata context of the image. Hence, we have developed a custom fusion model that integrates data from two distinct sources: the CapsNet model and the MetadataModel class.

The following is the general structure of the fusion model:

1.	Feature Extraction: The model extracts relevant features from the primary output of the CapsNet model.
2.	Metadata Feature Extraction: It also extracts relevant features from the MetadataModel.
3.	Feature Combination: The extracted features from both sources are then combined into a single feature set.
4.	Classification: This combined feature set is processed through an AdvancedClassifier model.

The AdvancedClassifier is a neural network designed to integrate and classify input data effectively. The following describes how the overall structure operates:

1.	The classifier expands the input into a higher dimensional space, followed by a ReLU activation to introduce non-linearity. Furthermore, a dropout is used in order to prevent overfitting.
2.	A residual connection is used to add the original input directly to the output of the later layers, helping to preserve possible lost information described by the vanishing gradient problem.
3.	The final layer maps the processed features into class logits, representing the prediction scores of the various classes.

The fusion model then has two outputs:

1.	The class logits returned by the final layer of the AdvancedClassifier.
2.	The feature map from the CapsNet model’s secondary output.

Output Processor
The output processor module handles the conversion of the two outputs from the Fusion model into a user-friendly format. 

![FixCaps Model Architecture](https://github.com/adikuma/Skin-Lesion-Classification/blob/main/fusion.jpg?raw=true)

## Repository

### Structure

- **`/work`:** Contains all the files and scripts that were developed and tested using Google Colab. Files in this folder may require minor adjustments for local execution.
- **`/evaluation`:** Stores the evaluation results and metrics computed from the models.
- **`app.py`:** The main application script located in the root directory of the repository.

### Execution Environment

The project was developed and run on Google Colab, so some environment-specific adjustments might be necessary for local execution, such as package installations or file path configurations.

## About The Application

This application utilizes machine learning models to analyze skin lesions and predict potential skin cancer types. The models can classify skin lesions into several types based on the image appearance and provide a heatmap overlay for better insight. It leverages a Streamlit framework for easy use and interaction.

## Running the Application

**Clone the Repository:**

 ```bash
    git clone https://github.com/adikuma/Skin-Lesion-Classification.git
 ```

**Run the Application:**
Use the following command to start the application:

```bash
    streamlit run app.py
```

This will start the Streamlit server, and you should be able to access the application by navigating to `localhost:8501` in your web browser.
