Email Spam Classifier 📧🚫

📌 Project Overview

This project is an Email Spam Classifier built using Machine Learning (Logistic Regression) and deployed as a Streamlit web application.

Users can enter email text and instantly get a prediction indicating whether the email is Spam or Not Spam, along with a confidence score.

🌐 Live Application

👉 Streamlit App:
https://email-spam-classifier-shravan.streamlit.app/

🎥 Project Demo
 ![Email Spam Classifier Demo](image/email%20spam%20classifier.gif)

🎯 Project Objectives

- Demonstrate text classification using Logistic Regression

- Build a complete ML pipeline from scratch

- Deploy a trained ML model as a live web application

- Provide real-time predictions with confidence scores

🧠 Machine Learning Details

- Algorithm: Logistic Regression

- Text Vectorization: TF-IDF

- Hyperparameter Tuning: GridSearchCV

- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

- Accuracy Achieved: > 90% on test dataset

⚙️ Tech Stack

- Python 3.8+

- Pandas

- NumPy

- Scikit-learn

- Streamlit

📁 Project Structure

```
Email-Spam-Classifier/
│
├── artifacts/
│ ├── model.pkl
| ├── target_encoder.pkl
│ └── preprocessor.pkl
| 
│
├── src/
|    ├──components/
│    |     ├── data_ingestion.py
│    |     ├── data_transformation.py
│    |     └── model_trainer.py
│    ├── utils.py
│    ├── logger.py
│    └── exception.py
├──Notebook/
|     ├──Email Spam Classifier.ipynb
|     └──mail_data.csv
├── main.py
├── app.py
├── setup.py
├── requirements.txt
├── README.md
└── image/
     └── email spam classifier.gif
```

🔄 Workflow

- User enters email text

- Text is preprocessed using TF-IDF

- Logistic Regression model predicts the class

- Result is displayed as Spam / Not Spam with confidence score

🚀 How to Run Locally

1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Application
streamlit run app.py

📊 Model Performance (After Hyperparameter Tuning)

The Logistic Regression model was optimized using GridSearchCV and evaluated on both training and test datasets.

| Metric    | Train Data | Test Data |
| --------- | ---------- | --------- |
| Accuracy  | 1.0000     | 0.9901    |
| Precision | 1.0000     | 0.9897    |
| Recall    | 1.0000     | 0.9990    |
| F1 Score  | 1.0000     | 0.9943    |


✅ Features

- Real-time spam detection

- Confidence score for predictions

- Clean UI built with Streamlit

- Error handling for empty input

- Modular and scalable codebase

📘 Learning Outcomes

- NLP preprocessing techniques

- End-to-end ML pipeline design

- Model persistence using pickle

- Debugging deployment-level ML issues

- Deploying ML models using Streamlit

📌 Conclusion

This project demonstrates a complete Machine Learning lifecycle, from raw data ingestion to a fully deployed web application, making it ideal for learning and showcasing practical ML skills.

👤 Author

Shravan Kumar Pandey
B.Tech (Hons) Data Science

🔗 GitHub: https://github.com/Shravan4598
