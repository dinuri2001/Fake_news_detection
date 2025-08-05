📰 Fake News Detection using Machine Learning

This project is a machine learning-based application that classifies news articles as Fake or Real. It uses Natural Language Processing (NLP) techniques and a Random Forest Classifier to make predictions based on the content of the news.

🚀 Features

Input any news article text and detect if it's real or fake
Built with a trained ML model using the TfidfVectorizer and RandomForestClassifier
Interactive web interface using Streamlit
Supports .pkl model and vectorizer for easy deployment

🧠 Model Details

Vectorizer: TF-IDF (Term Frequency-Inverse Document Frequency)
Classifier: Random Forest
Dataset Used: Kaggle Fake News Dataset (train.csv)

📦 Installation

Clone the repository
git clone https://github.com/dinuri2001/Fake_news_detection.git
cd Fake_news_detection
Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Run the app
streamlit run app.py

Project Structure

Fake_news_detection/
│
├── app.py                  # Streamlit app code
├── rf_model.pkl            # Trained Random Forest model
├── vectorizer.pkl          # TF-IDF Vectorizer
├── train.csv               # Dataset used for training
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

🛠 Technologies Used

Python
Scikit-learn
Pandas, NumPy
Streamlit
Pickle

⚠️ Limitations

Model may overfit to training data if not evaluated properly.
Not robust against out-of-domain or sarcastic/fake-looking news outside the dataset vocabulary.

📌 To Do

Improve prediction accuracy with larger dataset
Add more preprocessing (lemmatization, stemming)
Train with deep learning models like LSTM or BERT
Host on cloud platform (e.g., HuggingFace, Heroku, or Streamlit Cloud)

📬 Contact
Dinuri Gamage
💼 LinkedIn linkedin.com/in/dinuri-gamage
📧 gamagedinuri@gmail.com



