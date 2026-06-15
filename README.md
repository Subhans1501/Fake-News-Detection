# Fake News Detection System: Transformer Ensemble
### Natural Language Processing | BERT Family | Ensemble Learning | MLOps

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-F9AB00?logo=huggingface&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

The rapid spread of misinformation is one of the leading challenges in digital media. This project aims to combat fake news by leveraging state-of-the-art **Transformer architectures**. By fine-tuning and ensembling multiple models, the system achieves highly stable and accurate binary classification (Real vs. Fake) of news articles.

---

## Cloud Infrastructure & Open-Source Hosting

To ensure complete reproducibility and decoupling from local hardware restrictions, the curated dataset and all three fine-tuned transformer architectures are hosted publicly on the Hugging Face Hub.

* **Dataset Repository:** [subhan1501/fake-news-classification-dataset](https://huggingface.co/datasets/subhan1501/fake-news-classification-dataset)

### Model Registry
| Model Variant | Base Foundational Weights | Hugging Face Hub Repository Link |
| :--- | :--- | :--- |
| **BERT** | `bert-base-uncased` | [subhan1501/bert-fake-news-detector](https://huggingface.co/subhan1501/bert-fake-news-detector) |
| **RoBERTa** | `roberta-base` | [subhan1501/roberta-fake-news-detector](https://huggingface.co/subhan1501/roberta-fake-news-detector) |
| **DistilBERT** | `distilbert-base-uncased` | [subhan1501/distilbert-fake-news-detector](https://huggingface.co/subhan1501/distilbert-fake-news-detector) |

---

## Key Features & Architecture

### The Models
Instead of relying on a single architecture, this project utilizes a trio of powerful Hugging Face transformer models:
1. **BERT (`bert-base-uncased`)**: Provides deep bidirectional context understanding.
2. **RoBERTa (`roberta-base`)**: A robustly optimized approach that removes BERT's next-sentence pre-training and trains with larger mini-batches.
3. **DistilBERT (`distilbert-base-uncased`)**: A lighter, faster, distilled version of BERT that retains roughly 95% of its core language understanding capabilities while executing significantly faster.

### The Ensemble Strategy
To resolve individual network uncertainty and maximize predictive reliability, the predictions are filtered through a custom ensemble engine:
* **Majority Voting**: The final class prediction is determined by the hard consensus of the three architectures.
* **Average Probability**: The system handles uncertainty by averaging continuous probability distributions (`avg_proba`) across all models to calculate a stabilized system confidence score.

### Data Pipeline
* **Dataset**: Sourced from a multi-domain corpus containing over 40,000 compiled records.
* **Processing**: Implemented automated token fusion (Headline + Text body) and tokenization using model-specific tokenizers to preserve positional embeddings for the deep learning layers.

---

## Performance Insights

The models were evaluated using Confusion Matrices and standard sequence classification metrics (Accuracy, Precision, Recall, and F1-Score). 

* **DistilBERT** produced a highly conservative False Positive Rate, minimizing instances where fake news is improperly verified as real.
* **RoBERTa** achieved the strongest overall semantic consistency across complex and hyper-partisan linguistic patterns.
* **The Ensemble Layer** successfully mitigated the individual tracking weaknesses of each standalone model, resulting in superior prediction robustness and a minimized overall error rate.

---

## Interactive Web Application

The predictive pipeline is operationalized via a responsive Streamlit web interface. The app handles weight management completely on the cloud—automatically pulling fine-tuned states directly from the Hugging Face Hub upon first launch.

### **Installation & Local Setup:**

**1. Clone the repository:**
```bash
git clone [https://github.com/subhans1501/Fake-News-Transformer-Ensemble.git](https://github.com/subhans1501/Fake-News-Transformer-Ensemble.git)
cd Fake-News-Transformer-Ensemble
```
**2. Initialize Virtual Environment & Dependencies:**

```Bash
python -m venv venv
source venv/bin/activate  # On Windows use `.\venv\Scripts\Activate.ps1`
pip install -r requirements.txt
```
** 3. Launch the Application:**

```Bash
python -m streamlit run app.py
```
## Technical Toolbox
* **Language:** Python 3

* **Deep Learning Framework:** PyTorch

* **NLP & Transformers:** Hugging Face transformers, tokenizers

* **Machine Learning & Analytics:** Scikit-Learn, Pandas, NumPy

* **Visualization:** Matplotlib, Seaborn

* **Deployment UI:** Streamlit

## Developer Information
* **Developer:** Muhammad Subhan Shahid

* **Affiliation:** National University of Computer and Emerging Sciences (FAST-NU)

* **Program:** BS Artificial Intelligence (BSAI)