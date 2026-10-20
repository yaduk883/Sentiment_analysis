# Sentiment_analysis
Develop machine learning model and classify emotions in text samples
# Emotion Classification from Text

Classifies text comments into **anger**, **fear** or **joy** using TF-IDF features with two machine learning models: Naive Bayes and Support Vector Machine (SVM).

## Dataset

- File: `nlp_dataset.csv` (5,937 rows, 2 columns: `Comment`, `Emotion`)
- Classes are balanced: anger (2000), joy (2000), fear (1937)
- The notebook downloads it automatically from Google Drive using `gdown`

## Project Workflow

1. **Preprocessing:** lowercasing, removing URLs, mentions, digits and punctuation, tokenization, stopword removal (negations kept), lemmatization
2. **Feature extraction:** TF-IDF with unigrams and bigrams
3. **Models:** Multinomial Naive Bayes and Linear SVM
4. **Evaluation:** accuracy, macro/weighted F1, classification report, confusion matrices

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | 0.9057 | 0.9057 |
| **SVM** | **0.9402** | **0.9402** |

Preprocessing improved the SVM from 91.9% (raw text) to 94.0% (cleaned text).

**Conclusion:** SVM performs best. Linear SVMs work well on high-dimensional sparse text features, and emotions are largely separable by distinctive keywords.

   


## Files

- `sentiment_analysis_final.ipynb`: full code, explanations and results
- `README.md`: project overview

## Limitations

- Only three emotion classes exist.
- Some labels in the dataset appear noisy.

  
## How to Run

1. Open `sentiment_analysis_final.ipynb` in Google Colab

Google Colab link: https://colab.research.google.com/drive/1TEGDJpRux2x7kOfVUAYkcy9YF2Y0fqXz?usp=sharing

