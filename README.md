# Sentiment Analysis NLP API

A professional end-to-end NLP sentiment analysis project that classifies user feedback into three sentiment categories:

- Positive
- Negative
- Neutral

The project is structured for a machine learning portfolio and includes data preparation, preprocessing, model training, evaluation, model persistence, FastAPI inference, and automated tests.

## Problem Statement

Businesses receive large volumes of unstructured text from reviews, tickets, surveys, and social media posts. Manually reading every message is slow and inconsistent. This project automates sentiment classification so teams can quickly identify positive feedback, negative complaints, and neutral statements.

## NLP Workflow

1. Load and inspect labeled review data
2. Clean and normalize raw text
3. Convert text to numeric features using TF-IDF
4. Train multiple baseline classifiers
5. Compare performance using standard metrics
6. Save the best model with Joblib
7. Serve predictions through a FastAPI endpoint

## Dataset

The repository includes `data/reviews.csv` as a ready-to-run starter dataset. For a larger real-world experiment, you can replace it with a public sentiment dataset such as:

- IMDB reviews dataset
- Twitter sentiment dataset
- Amazon reviews dataset

If you use a different dataset, keep the CSV format:

```csv
text,sentiment
I love this product,positive
This was terrible,negative
It is okay,neutral
```

If automatic download is unavailable in your environment, manually add the dataset to `data/reviews.csv`.

## Why TF-IDF?

TF-IDF works well for classical sentiment analysis because it:

- Represents text as weighted numerical features
- Down-weights very common words
- Highlights informative terms
- Works efficiently with small to medium tabular NLP datasets

## Models Compared

- Logistic Regression
- Naive Bayes
- Support Vector Machine

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Project Structure

```text
sentiment-analysis-nlp-api/
├── data/
│   └── reviews.csv
├── notebooks/
│   └── sentiment_analysis_eda.ipynb
├── models/
│   └── sentiment_model.pkl
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
├── api/
│   └── main.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python -m src.train_model
```

This trains all candidate models, evaluates them, and saves the best pipeline to:

```text
models/sentiment_model.pkl
```

## Run the API

```bash
uvicorn api.main:app --reload
```

Open the interactive docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API Example

### Request

```http
POST /predict
Content-Type: application/json

{
  "text": "The movie was excellent"
}
```

### Response

```json
{
  "sentiment": "positive"
}
```

## Testing

Run the API tests with:

```bash
pytest
```

## EDA Notebook

The notebook should include:

- Dataset overview
- Class distribution
- Text length analysis
- Word frequency analysis
- Most common words
- Sentiment distribution visualizations

Recommended libraries:

- Pandas
- Matplotlib
- Seaborn
- WordCloud

## Results and Metrics

After training, the script prints the best model and its evaluation metrics. The exact scores depend on the dataset used. On the starter dataset, the model comparison is intended to demonstrate the full workflow and can be replaced with a larger public corpus for stronger metrics.

## Future Improvements

- Add lemmatization and stemming
- Train on a larger public dataset
- Add class balancing experiments
- Expose confidence scores in the API
- Add Docker support
- Add CI checks and model versioning

## License

MIT License
