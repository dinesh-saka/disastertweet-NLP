# Text Classification Models Comparison for disaster or not dataset

## 👋 Introduction

The goal of this project is to develop effective models for classifying tweets as either related to disasters or not. To achieve this, we experiment with different deep learning architectures and techniques, analyzing their accuracy, precision, recall, and F1-score.

## 🥷 Models

### 1. Naive Bayes (Baseline)
This is our baseline model, providing a simple yet effective starting point for comparison.

### 2. Feed-forward Neural Network (Dense Model)
We implement a basic feed-forward neural network to capture complex patterns in the text data.

### 3. Long Short-Term Memory (LSTM) Model (RNN)
The LSTM model is a type of recurrent neural network (RNN) designed to overcome the vanishing gradient problem, making it well-suited for sequential data like text.

### 4. Gated Recurrent Unit (GRU) Model (RNN)
Similar to LSTM, the GRU model is another type of RNN that excels at capturing long-range dependencies in sequential data.

### 5. Bidirectional LSTM Model (RNN)
The bidirectional LSTM model enhances the LSTM architecture by processing input sequences in both forward and backward directions, improving context understanding.

### 6. 1D Convolutional Neural Network (CNN)
We explore the use of CNNs for text classification, leveraging their ability to extract hierarchical features from input sequences.

### 7. TensorFlow Hub Pretrained Feature Extractor (Transfer Learning for NLP)
This model utilizes transfer learning with a TensorFlow Hub pre-trained feature extractor, leveraging pre-trained embeddings to enhance classification performance.

### Ensemble Model
We combine predictions from multiple models using model ensembling/stacking techniques to improve classification accuracy further.

## 💻 Results

We evaluate each model's performance on the validation dataset, considering metrics such as accuracy, precision, recall, and F1-score. Here's a summary of the results:

| Model                              | Accuracy | Precision | Recall | F1-Score |
|------------------------------------|----------|-----------|--------|----------|
| Baseline                           | 79.27%   | 0.79      | 0.79   | 0.79     |
| Feed-forward Neural Network        | 78.74%   | 0.79      | 0.79   | 0.78     |
| LSTM Model                         | 75.72%   | 0.76      | 0.76   | 0.76     |
| GRU Model                          | 77.03%   | 0.77      | 0.77   | 0.77     |
| Bidirectional LSTM Model           | 76.38%   | 0.76      | 0.76   | 0.76     |
| 1D Convolutional Neural Network   | 77.56%   | 0.78      | 0.78   | 0.77     |
| TensorFlow Hub Pretrained Feature Extractor | 81.50% | 0.82      | 0.81   | 0.81     |
| Ensemble Model                     | 78.35%   | 0.78      | 0.78   | 0.78     |

Additionally, we explore model ensembling/stacking to combine the predictions of multiple models, achieving a further improvement in performance.

## 🔎 Conclusion

Our experiments demonstrate the effectiveness of various deep learning architectures for text classification tasks. By comparing the performance of different models, we provide insights into selecting suitable approaches for similar tasks in natural language processing (NLP).
