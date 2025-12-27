\# 📩 SMS Spam Detection using Machine Learning



!\[Python](https://img.shields.io/badge/Python-3.8+-blue)

!\[Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-orange)

!\[Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

!\[License](https://img.shields.io/badge/License-MIT-green)

!\[Status](https://img.shields.io/badge/Status-Complete-brightgreen)



---



\## 📌 Description



An end-to-end \*\*SMS Spam Detection system\*\* developed using \*\*Machine Learning and Natural Language Processing (NLP)\*\* techniques and deployed as an interactive \*\*Streamlit web application\*\*.



The system classifies SMS messages as \*\*Spam\*\* or \*\*Ham (Not Spam)\*\* and provides prediction confidence along with token-level interpretability to help users understand model behavior.



🔗 \*\*Live Application:\*\*  

https://sms-spam-detection-amitx2209.streamlit.app/



---



\## 🚀 Project Overview



Spam messages are a major issue in mobile communication systems, often leading to fraud and poor user experience.



This project demonstrates a complete machine learning workflow, including:



• Data preprocessing  

• Feature extraction using TF-IDF  

• Model training and evaluation  

• Model deployment using Streamlit  



Multiple machine learning models were evaluated during experimentation.  

Based on performance and efficiency, \*\*Multinomial Naive Bayes\*\* was selected for deployment.



---



\## ✨ Key Features



• End-to-end machine learning pipeline  

• TF-IDF based feature extraction  

• Multinomial Naive Bayes classifier  

• Interactive Streamlit web interface  

• Prediction confidence visualization  

• Token frequency visualization for explainability  

• Clean, dark-themed user interface  

• Deployment-ready and version-controlled project  



---



\## 🧠 Machine Learning Approach



\### Dataset



• SMS Spam Collection Dataset (UCI Machine Learning Repository)  

• Total messages: 5,572  

• Classes:  

&nbsp; • Spam  

&nbsp; • Ham (Not Spam)  



---



\### Data Preprocessing



• Conversion of text to lowercase  

• Removal of punctuation and special characters  

• Cleaning of text before vectorization  



---



\### Feature Engineering



• TF-IDF vectorization  

• Unigrams and bigrams  

• Vocabulary limited to top 1000 features  



---



\## 🔬 Model Experimentation



During experimentation, multiple classifiers were trained and evaluated:



• Multinomial Naive Bayes  

• Logistic Regression  

• Support Vector Machine (SVM)  

• Random Forest  



Each model was evaluated based on:



• Classification accuracy  

• Consistency across validation sets  

• Computational efficiency  



This comparative analysis enabled informed model selection for real-time deployment.



---



\## 🏆 Deployed Model



\*\*Multinomial Naive Bayes\*\* was selected for deployment due to:



• Low inference time for real-time classification  

• Lightweight and simple architecture  

• Strong performance on short SMS text  

• Effective integration with TF-IDF features  



Although other models achieved competitive accuracy, Multinomial Naive Bayes offered the best balance between performance and efficiency.



---



\## 📊 Model Performance



• Accuracy: \*\*~98%\*\* on held-out test dataset  



---



\## 🌐 Streamlit Web Application



The Streamlit-based web application allows users to:



• Enter a custom SMS message  

• Classify it instantly as Spam or Ham  

• View prediction confidence  

• Explore influential tokens affecting predictions  



\### UI Highlights



• Clean dark-themed interface  

• Confidence bar visualization  

• Sidebar with project information  

• Token frequency charts for interpretability  

• Responsive and lightweight design  



---



\## 🔍 Token Frequency Visualization



To improve model interpretability, the application visualizes:



• Tokens most strongly associated with Spam messages  

• Tokens most strongly associated with Ham messages  



These tokens are derived from learned TF-IDF weights and Naive Bayes feature probabilities.



---



\## 👥 Contributors



This project was developed as a collaborative academic effort.



📄 Full contributor details are available here:  

➡️ \[CONTRIBUTORS.md](CONTRIBUTORS.md)



---



\## 🛠 How to Run the Project Locally



\### Step 1: Clone the Repository



```bash

git clone https://github.com/amitx2209/SMS-Spam-Detection

cd SMS-Spam-Detection



