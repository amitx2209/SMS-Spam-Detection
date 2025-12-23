SMS Spam Detection



This project is a Python-based machine learning application that can detect whether an SMS message is Spam or Ham (Not Spam). It demonstrates how machine learning can help filter unwanted messages and protect users from spam.



Technologies used:



Python for programming and data manipulation

Pandas for handling datasets

Scikit-learn for machine learning algorithms

TF-IDF Vectorizer for converting text into numerical features

Naive Bayes Classifier for training the spam detection model



Dataset:



The project uses the SMS Spam Collection Dataset, which contains 5,572 SMS messages labeled as spam or ham. This dataset is publicly available and widely used for spam detection projects.



Project workflow:



Data loading and cleaning – load the dataset and remove missing or irrelevant entries.



Text preprocessing – clean the text by removing punctuation, stopwords, and normalizing the messages.



Feature extraction – convert messages into numerical features using TF-IDF.



Model training – train a Naive Bayes classifier to detect spam messages.



Model evaluation – measure accuracy, precision, and recall to check performance.



Predicting new messages – test the model on new SMS messages through a Streamlit web app.



Model performance:



Accuracy: approximately 97%

Spam detection: high precision and recall

The model performs well in distinguishing spam from ham messages.



How to run the project:



Clone the repository:



git clone https://github.com/your-username/SMS-Spam-Detection.git

cd SMS-Spam-Detection





Install required libraries:



pip install pandas scikit-learn streamlit





Run the Streamlit web app:



python -m streamlit run app.py





Enter an SMS message in the browser and click Predict to see whether it is spam or ham.



Contributors:



Amit Sharma – SMS Spam Detection project and Streamlit App

Add other contributors here if needed



Built with Python, scikit-learn, and Streamlit.



Live Demo



You can try the SMS Spam Detection app live here: \[Open the app](https://sms-spam-detection-amitx2209.streamlit.app/)

Try the app online! Enter any SMS message and see if it is classified as Spam or Ham in real-time.





