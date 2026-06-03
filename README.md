# 📰 Fake News Detection System: Transformer Ensemble
### Natural Language Processing | BERT Family | Ensemble Learning

The rapid spread of misinformation is one of the leading challenges in digital media. This project aims to combat fake news by leveraging state-of-the-art **Transformer architectures**. By fine-tuning and ensembling multiple models, the system achieves highly stable and accurate binary classification (Real vs. Fake) of news articles.

---

## 🚀 Key Features & Architecture

### 🧠 The Models
Instead of relying on a single architecture, this project utilizes a trio of powerful Hugging Face transformers:
1. **BERT (`bert-base-uncased`)**: Provides deep bidirectional context understanding.
2. **RoBERTa (`roberta-base`)**: A robustly optimized approach that removes BERT's next-sentence pretraining and trains with larger mini-batches.
3. **DistilBERT (`distilbert-base-uncased`)**: A lighter, faster, distilled version of BERT that retains 97% of its language understanding capabilities.

### 🤝 The Ensemble Strategy
To resolve conflicts and maximize predictive reliability, the models are combined using a custom ensemble pipeline:
* **Majority Voting**: The final class prediction is determined by the consensus of the three models.
* **Average Probability**: In cases of uncertainty, the system averages the continuous probability outputs (`avg_proba`) across all models to break ties and provide a finalized confidence score.

### 🛠️ Data Pipeline
* **Dataset**: Sourced from the [Kaggle Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets), containing over 40,000 samples.
* **Processing**: Implemented automated EDA, data merging, and tokenization using model-specific tokenizers to prepare text for the deep learning layers.

---

## 📊 Performance Insights
The models were rigorously evaluated using Confusion Matrices and standard classification metrics (Accuracy, Precision, Recall, and F1-Score). 

* **DistilBERT** produced the lowest False Positive Rate, making it highly conservative and accurate.
* **RoBERTa** achieved the strongest overall consistency across various linguistic patterns.
* **The Ensemble** successfully mitigated the individual weaknesses of each standalone model, resulting in superior prediction robustness.

---

## 💻 Interactive Web Application
The predictive pipeline is operationalized via an interactive web interface. Users can input raw text from a news article, select individual models to see their isolated predictions, or run the full **Final Ensemble Decision** to get a unified verdict and confidence score.

### **How to Run the Application:**
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Fake-News-Transformer-Ensemble.git](https://github.com/YourUsername/Fake-News-Transformer-Ensemble.git) ```

2. Install the required dependencies:

Bash
```bash
pip install -r requirements.txt
```
3. Launch the web app:


Bash
```bash
python app/app.py
```
## 🛠️ Technical Toolbox

* **Language:** Python 3

* **Deep Learning Framework:** PyTorch

* **NLP & Transformers:** Hugging Face transformers library

* **Machine Learning:** Scikit-Learn

* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn

## Developer Information
* **Developer:** Muhammad Subhan Shahid

* **Affiliation:** National University of Computer and Emerging Sciences (FAST-NU)

* **Program:** BS Artificial Intelligence (BSAI)