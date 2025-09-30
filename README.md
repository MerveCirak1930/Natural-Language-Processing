# NLP - Natural Language Processing Project

This repository hosts a series of Python projects covering various techniques and methods in the field of Natural Language Processing (NLP). Each folder focuses on a specific NLP topic and provides practical implementations.

## Project Structure

The project includes modules covering the following main topics:

1-**Data Cleaning and Preprocessing**: Tokenization, stopword removal, stemming, and lemmatization are applied to a raw text dataset.

2-**Basic Statistics and Bayesian Models**: Probability and likelihood calculations, and Bayesian models.

3-**Language Models**: Hidden Markov Models, Maximum Entropy Models, n-gram models.

4-**Frequency-Based Methods**: BoW, TF-IDF, n-gram models.

5-**Word Embeddings**: The similarity between word pairs in the AnlamVer dataset and their human-annotated similarity and relatedness scores is evaluated using a selected word embedding model (Word2Vec, GloVe, or FastText) and Spearman correlation.

6-**Sentiment Analysis**: Texts from a Turkish blog dataset are cleaned, vectorized using TF-IDF, and classified into mood categories using multiple machine learning models, with Logistic Regression achieving the best performance and allowing user input for real-time prediction.

7-**Clustering and Topic Modeling**: The text documents are cleaned, vectorized using TF-IDF, clustered with multiple algorithms, and topic-modeled using LDA and NMF, then the clustering and topic modeling results are evaluated against true labels using ARI and NMI metrics.

8-**Contextual Meaning and Word Sense Disambiguation**: A polysemous Turkish word in a sentence is disambiguated by comparing its contextual BERT embedding with embeddings of possible sense definitions, selecting the most semantically similar sense.

9-**Dependency Parsing**: Neural network based dependency analysis is performed with the Stanza tool.

10-**Semantic Analysis and Semantic Models**: The semantic similarity between two user-provided sentences is calculated using Sentence-BERT.

11-**Word Cloud**: The frequency of words in a text is statistically analyzed and visualized using a Word Cloud.


# Word Cloud Example

This is a sample Word Cloud generated from our text data:

![Word Cloud](11-Word%20Cloud/word_cloud.png)



## Requirements

To run the project, you will need the following Python libraries:

```bash
   pip install numpy pandas matplotlib seaborn scikit-learn nltk spacy gensim tensorflow pytorch
```

## Usage

Each module contains examples and explanations on how to implement the techniques. For example, the "1-Data Cleaning and Preprocessing" folder contains Python scripts demonstrating data cleaning and preprocessing steps.

## License

This project is licensed under the MIT License.

If you use this project for development or wish to contribute, please submit a pull request or open an issue. Contributions are welcome!
