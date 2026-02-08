# Sentiment Analysis with DistilBERT 🎭

This project implements a high-performance sentiment analysis model using **DistilBERT** to classify text into three categories: **Positive**, **Neutral**, and **Negative**. It includes a professional training notebook and a premium-looking Streamlit web application for real-time inference.

## 🚀 Features
- **State-of-the-Art Model**: Finetuned `distilbert-base-uncased` for accurate sentiment classification.
- **High-Performance Training**:
    - Trained for **10 Epochs** with **Cosine Learning Rate Scheduler**.
    - Uses **Mixed Precision (fp16)** for speed and efficiency.
    - **Early Stopping** to prevent overfitting.
- **Interactive Web App**: A beautiful Streamlit interface for easy testing.
- **Visualizations**:
    - **WordClouds** for Positive and Negative sentiments.
    - **Text Length Distribution** analysis.
    - **Confusion Matrix** & **Training History** plots.

## 📂 Project Structure
- `app.py`: The Streamlit web application for real-time sentiment analysis.
- `English_Sentiment_EndToEnd.ipynb`: The main notebook containing the full pipeline (Data Loading -> Preprocessing -> Training -> Evaluation).
- `requirements.txt`: List of Python dependencies (including `wordcloud`).
- `saved_model/`: Directory containing the trained model artifacts (model weights, tokenizer, config).
- `Data/`: Directory containing dataset files (if applicable).

## 🛠️ Setup & Usage

### 1. Install Dependencies
Ensure you have Python installed (3.8+ recommended). Run the following command to install all necessary libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Web App (Streamlit)
To launch the interactive web application:
```bash
streamlit run app.py
```
This will open the app in your default web browser.

### 3. Run the Training Notebook
To retrain the model with high-performance settings, open `English_Sentiment_EndToEnd.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run all cells.
```bash
jupyter notebook English_Sentiment_EndToEnd.ipynb
```

## 📊 Dataset
We use the [Multiclass Sentiment Analysis Dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset).
- **Labels**: Positive (2), Neutral (1), Negative (0).
- **Size**: >40k samples.

## 🤖 Model Details
- **Architecture**: `distilbert-base-uncased`
- **Batch Size**: 32
- **Epochs**: 10 (with Early Stopping)
- **Optimizer**: AdamW (Fused)
- **Scheduler**: Cosine
- **Learning Rate**: 2e-5

## 📈 Results
The notebook generates a detailed classification report, confusion matrix, and training loss curves to evaluate the model's performance.

---
*Powered by Hugging Face Transformers & Streamlit*
