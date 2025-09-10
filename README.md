# 📊 Sentiment Analysis on Product Reviews  

## 🔍 Overview  
This project analyzes customer product reviews and classifies them into **Positive, Neutral, or Negative sentiments**.  
It uses **Natural Language Processing (NLP)** techniques such as **TF-IDF vectorization** and a **Logistic Regression classifier**.  
Finally, the project compares **brand-wise sentiment distribution** with visualization.  

---

## ⚙️ Features  
- Data cleaning & preprocessing of raw text  
- Converting star ratings → sentiment labels  
- TF-IDF based feature extraction  
- Logistic Regression classification model  
- Evaluation using **Accuracy** & **Classification Report**  
- Brand-wise sentiment comparison with **stacked bar charts**  

---

## 🛠️ Tech Stack  
- Python 🐍  
- Pandas  
- scikit-learn  
- Matplotlib  
- Regular Expressions  

---

## 📂 Project Structure  
Sentiment-Analysis-Project/
│── data/
│ └── final_dataset.csv
│── sentiment_analysis.py
│── requirements.txt
│── README.md

---

## 🚀 How to Run  

1. Clone this repository:  
```bash
git clone https://github.com/WAASIFAI/Sentiment-Analysis-Project.git
cd Sentiment-Analysis-Project
pip install -r requirements.txt
python sentiment_analysis.py
📊 Sample Output

Sentiment distribution across dataset

Model evaluation (accuracy & classification report)

Brand-wise sentiment stacked bar chart

✅ Results

Built a Logistic Regression model to classify customer reviews.

Achieved a reasonable accuracy score.

Provided insights into brand-wise customer perception.

📌 Future Improvements

Try advanced models like SVM, Random Forest, or BERT

Aspect-based sentiment analysis (specific features of products)

Deploy as a web app

👨‍💻 Developed by Waasif

📄 requirements.txt
pandas
scikit-learn
matplotlib


Would you like me to also add a **`.gitignore`** snippet here (so that `final_dataset.csv` doesn’t get uploaded accidentally)?
