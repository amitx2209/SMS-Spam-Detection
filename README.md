SMS Spam Detection using Machine Learning



An end-to-end SMS Spam Detection system built using Machine Learning (TF-IDF and Multinomial Naive Bayes) and deployed as an interactive Streamlit web application.



During experimentation, multiple machine learning models were evaluated; however, Multinomial Naive Bayes was selected for deployment due to its efficiency and suitability for short SMS text.



The application classifies SMS messages as Spam or Ham (Not Spam) and provides prediction confidence along with token-level interpretability to help understand model behavior.



PROJECT OVERVIEW



Spam messages are a major problem in mobile communication systems.

This project uses Natural Language Processing (NLP) techniques to automatically detect spam SMS messages with high accuracy.



The system demonstrates the complete machine learning workflow, from data preprocessing and feature extraction to model inference and deployment.



KEY FEATURES



• End-to-end machine learning pipeline

• TF-IDF based feature extraction

• Multinomial Naive Bayes classifier

• Interactive Streamlit web interface

• Prediction confidence visualization

• Token frequency visualization for explainability

• Version-controlled and deployment-ready



MACHINE LEARNING APPROACH



Dataset



• SMS Spam Collection Dataset (UCI Repository)

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



MODEL EXPERIMENTATION



During the experimentation phase, multiple machine learning classifiers were trained and evaluated to identify the most suitable model for SMS spam classification.



The following models were considered:



• Multinomial Naive Bayes

• Logistic Regression

• Support Vector Machine (SVM)

• Random Forest



Each model was evaluated based on classification accuracy, consistency across validation sets, and computational efficiency.

This comparative analysis enabled an informed selection of the model best suited for real-time deployment.



DEPLOYED MODEL



Based on the results obtained during model experimentation, the Multinomial Naive Bayes classifier was selected for deployment in the web application.



The selection was motivated by the following factors:



• Low inference time, enabling real-time SMS classification

• Lightweight and simple model architecture

• Consistent performance on short text messages

• Effective integration with TF-IDF feature representation



Although other models demonstrated competitive accuracy during experimentation, Multinomial Naive Bayes provided the best trade-off between performance and efficiency, making it suitable for deployment in a Streamlit-based web application.



MODEL PERFORMANCE



The model achieves approximately 98% accuracy on a held-out test dataset.



STREAMLIT WEB APPLICATION



The Streamlit-based web application allows users to:



• Enter a custom SMS message

• Instantly classify it as Spam or Ham

• View prediction confidence

• Explore top influential tokens for each class in the sidebar



UI Highlights



• Clean dark-themed interface

• Confidence bar visualization

• Sidebar containing project information and token frequency charts

• Responsive and lightweight design



TOKEN FREQUENCY VISUALIZATION



To improve model interpretability, the application visualizes:



• Tokens most strongly associated with Spam messages

• Tokens most strongly associated with Ham messages



These tokens are derived from the learned TF-IDF and Naive Bayes feature probabilities, helping users understand which words influence predictions.



HOW TO RUN THE PROJECT LOCALLY



Step 1: Clone the Repository



• git clone https://github.com/amitx2209/SMS-Spam-Detection



• cd SMS-Spam-Detection



Step 2: Install Dependencies



• pip install -r requirements.txt



Step 3: Run the Streamlit Application



• python -m streamlit run app.py



The application will be available at:

https://sms-spam-detection-amitx2209.streamlit.app/



Note on Model Selection



Although multiple machine learning models were explored during the experimentation phase, only the selected model is included in the deployed application to ensure fast response time and smooth user experience.



LIMITATIONS



• Model performance depends on historical patterns in the dataset

• Very short or ambiguous messages may be misclassified

• Designed primarily for English-language SMS messages



FUTURE ENHANCEMENTS



• Multilingual spam detection

• Real-time SMS integration

• Model retraining and update pipeline

• Advanced explainability techniques (SHAP, LIME)



DEVELOPERS NAME



• Amit Sharma – Project Lead and Primary Developer

• Priyanka Kumari – Documentation Support

• Praveen Prakash – Dataset Review

• Aatish Raj – Presentation Support

• Jay Prakash Kumar – Project Review

• Sarfarazur Rehman – Literature Survey

• Ziyaur Rehman – Testing Assistance



CONTRIBUTION STATEMENT



The project was developed under a collaborative framework.

The primary development, model design, implementation, and deployment were carried out by the project lead.

Other team members contributed through documentation support, dataset review, literature survey, testing, and presentation preparation.



GitHub Repository:

https://github.com/amitx2209/SMS-Spam-Detection



If you find this project useful, feel free to star the repository.

