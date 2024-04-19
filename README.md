# Skin Lesion Classification

# Skin Lesion Classifier

## Problem Statement

The primary challenge addressed in this project is the development of an automated and reliable system to identify the presence or absence of skin cancer in individuals using limited information. The goal is to enhance early skin cancer detection capabilities amongst the general population using a predictive model that can accurately determine whether an individual is healthy or has potential skin cancer from an uploaded image of a suspected skin lesion.

## About The Application

This application utilizes machine learning models to analyze skin lesions and predict potential skin cancer types. The models can classify skin lesions into several types based on the image appearance and provide a heatmap overlay for better insight. It leverages a Streamlit framework for easy use and interaction.

## Models

### FixCaps Model

- **Fixed Capsule Network (FixCaps):** A novel approach that encodes spatial hierarchies between features, enhancing the model’s ability to interpret complex medical images with an accuracy potential of 96.49%.

### Fusion Model

- **Fusion Model:** Integrates the outputs from the FixCaps model with metadata inputs to potentially enhance diagnostic accuracy, though it demonstrated weaker results in trials.

## Running the Application

**Clone the Repository:**

    ```bash
    git clone <repository-url>
    ```

4. **Run the Application:**

    Use the following command to start the application:

    ```bash
    streamlit run app.py
    ```

    This will start the Streamlit server, and you should be able to access the application by navigating to `localhost:8501` in your web browser.
