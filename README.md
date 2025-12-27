📩 SMS Spam Detection using Machine Learning
<p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue" /> 
    <img src="https://img.shields.io/badge/Machine%20Learning-NLP-orange" /> 
    <img src="https://img.shields.io/badge/Streamlit-Web%20App-red" /> 
    <img src="https://img.shields.io/badge/License-MIT-green" /> 
    <img src="https://img.shields.io/badge/Status-Complete-brightgreen" /> 
</p>

📌 Description

An end-to-end SMS Spam Detection system developed using Machine Learning and Natural Language Processing (NLP) techniques and deployed as an interactive Streamlit web application.

The system classifies SMS messages as Spam or Ham (Not Spam) and provides prediction confidence along with token-level interpretability to help understand model behavior.

🌐 Live Application:
https://sms-spam-detection-amitx2209.streamlit.app/

🚀 Project Overview

Spam messages are a major problem in mobile communication systems, often leading to fraud and poor user experience.

This project demonstrates a complete machine learning workflow, including:

• Data preprocessing
• Feature extraction using TF-IDF
• Model training and evaluation
• Real-time inference
• Deployment using Streamlit

During experimentation, multiple machine learning models were evaluated.
Based on performance and efficiency, Multinomial Naive Bayes was selected for deployment.

✨ Key Features

• End-to-end machine learning pipeline
• TF-IDF based feature extraction
• Multinomial Naive Bayes classifier
• Interactive Streamlit web interface
• Prediction confidence visualization
• Token frequency visualization for explainability
• Clean dark-themed user interface
• Deployment-ready and version-controlled project

🧠 Machine Learning Approach
Dataset

• SMS Spam Collection Dataset (UCI Machine Learning Repository)
• Total messages: 5,572
• Classes:
• Spam
• Ham (Not Spam)

Data Preprocessing

• Conversion of text to lowercase
• Removal of punctuation and special characters
• Cleaning of text before vectorization

Feature Engineering

• TF-IDF vectorization
• Unigrams and bigrams
• Vocabulary limited to top 1000 features

🔬 Model Experimentation

The following machine learning models were trained and evaluated:

• Multinomial Naive Bayes
• Logistic Regression
• Support Vector Machine (SVM)
• Random Forest

Each model was evaluated based on:

• Classification accuracy
• Consistency across validation sets
• Computational efficiency

This comparative analysis enabled informed selection of the final deployed model.

🏆 Deployed Model

Multinomial Naive Bayes was selected for deployment due to:

• Low inference time enabling real-time classification
• Lightweight and simple model architecture
• Consistent performance on short SMS messages
• Effective integration with TF-IDF features

Although other models achieved competitive accuracy, Multinomial Naive Bayes offered the best trade-off between performance and efficiency.

📊 Model Performance

• Accuracy: ~98% on a held-out test dataset

🌐 Streamlit Web Application

The Streamlit-based web application allows users to:

• Enter a custom SMS message
• Instantly classify it as Spam or Ham
• View prediction confidence
• Explore influential tokens affecting predictions

UI Highlights

• Clean dark-themed interface
• Confidence bar visualization
• Sidebar with project information
• Token frequency charts for interpretability
• Responsive and lightweight design

🔍 Token Frequency Visualization

To enhance model interpretability, the application visualizes:

• Tokens most strongly associated with Spam messages
• Tokens most strongly associated with Ham messages

These tokens are derived from learned TF-IDF weights and Naive Bayes feature probabilities.

🛠 How to Run the Project Locally
Step 1: Clone the Repository

git clone https://github.com/amitx2209/SMS-Spam-Detection

cd SMS-Spam-Detection

Step 2: Install Dependencies

python -m pip install -r requirements.txt

Step 3: Run the Application

python -m streamlit run app.py

⚠️ Limitations

• Model performance depends on historical dataset patterns
• Very short or ambiguous messages may be misclassified
• Designed primarily for English-language SMS messages

🔮 Future Enhancements

• Multilingual spam detection
• Real-time SMS integration
• Automated model retraining pipeline
• Advanced explainability techniques (SHAP, LIME)

👥 Team Contributions

This project was developed as a collaborative academic effort.

• Amit Sharma
– Project Lead
– Problem formulation
– Data preprocessing and feature engineering
– Model training, evaluation, and selection
– Streamlit application development
– Deployment and repository maintenance

• Priyanka Kumari
– Technical documentation
– README preparation
– Report formatting support

• Praveen Prakash
– Dataset review
– Data validation
– Exploratory analysis support

• Aatish Raj
– Presentation design
– Project demonstration support

• Jay Prakash Kumar
– Project review
– Result validation
– Technical feedback

• Sarfarazur Rehman
– Literature survey
– Background research

• Ziyaur Rehman
– Testing assistance
– Result verification


📄 License

This project is licensed under the MIT License.

⭐ If you find this project useful, feel free to star the repository.
