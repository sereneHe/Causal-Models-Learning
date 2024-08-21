# Introduction
These are results for Learning Joint Multiple Dynamical Systems via Non-Commutative Polynomial Optimization(NCPOP). The package contains various functionalities related to causal learning and evaluation, including: 
* Data generation and processing: data simulation, data reading operators, and data pre-processing operators（such as prior injection and variable selection).
* Causal structure learning: causal structure learning methods, including classic and recently developed methods, especially gradient-based ones that can handle large problems.
* Evaluation metrics: various commonly used metrics for causal structure learning, including F1, SHD, FDR, TPR, FDR, NNZ, etc.

Args:
W_true: ground truth graph
W_est: predicted graph
W_und: predicted undirected edges in CPDAG, asymmetric

Returns in dict:
fdr: (reverse + false positive) / prediction positive
tpr: (true positive) / condition positive
fpr: (reverse + false positive) / condition negative
shd: undirected extra + undirected missing + reverse
nnz: prediction positive


# Definitions and applications:
## 1. FDR (False Discovery Rate)

- **Definition**: The false discovery rate is the proportion of instances identified as positive that are actually negative.
- **Formula**:
  ```math
  \text{FDR} = \frac{\text{FP}}{\text{TP} + \text{FP}}
  ```
  where FP (False Positives) is the number of negative instances incorrectly classified as positive, and TP (True Positives) is the number of correctly identified positive instances.
- **Application**: FDR is mainly used in multiple hypothesis testing, especially in genomics and data analysis.

## 2. TPR (True Positive Rate) - Recall

- **Definition**: The true positive rate is the proportion of actual positive instances that are correctly identified as positive.
- **Formula**:
  ```math
  \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  ```
  where FN (False Negatives) is the number of positive instances incorrectly classified as negative.
- **Application**: TPR, also known as recall, is commonly used to measure the sensitivity of a classifier in detecting positive instances.

## 3. FPR (False Positive Rate)

- **Definition**: The false positive rate is the proportion of actual negative instances that are incorrectly identified as positive.
- **Formula**:
  ```math
  \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
  ```
  where TN (True Negatives) is the number of correctly identified negative instances.
- **Application**: FPR is used to evaluate the ability of a classifier to avoid false alarms, particularly important in imbalanced datasets.

## 4. SHD (Structural Hamming Distance)

- **Definition**: SHD is used to compare the differences between two graph structures (e.g., Bayesian networks or Markov networks). It represents the number of edge operations (additions, deletions, or reversals) required to transform one graph into another.
- **Application**: SHD is often used to evaluate the difference between a learned graph structure and the true graph structure.

## 5. NNZ (Number of Non-Zero Elements)

- **Definition**: NNZ refers to the number of non-zero elements in a matrix, commonly used to indicate the sparsity of a matrix.
- **Application**: In optimization problems, NNZ is used to measure the sparsity of a solution, especially in regularization methods like L1 regularization.

## 6. Precision

- **Definition**: Precision is the proportion of instances identified as positive that are actually positive.
- **Formula**:
  ```math
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  ```
- **Application**: Precision is used to evaluate the accuracy of a classifier, particularly when the focus is on minimizing false positives.

## 7. Recall

- **Definition**: Recall (also known as TPR, see above) is the proportion of actual positive instances that are correctly identified as positive.
- **Formula**:
  ```math
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  ```
- **Application**: Recall is used to measure a classifier's ability to detect positive instances, especially in applications where missing positive cases is costly.

## Summary

- **FDR** measures the proportion of false discoveries, often used in multiple testing scenarios.
- **TPR (Recall)** measures the detection rate of positive instances.
- **FPR** measures the rate of false positives.
- **SHD** measures the difference between graph structures.
- **NNZ** measures the sparsity of a matrix.
- **Precision** measures the accuracy of positive predictions.
- **Recall (TPR)** measures the ability to detect positive instances.

These metrics together help provide a comprehensive evaluation of model performance and the effectiveness of learning algorithms.
```

This is how the explanation would look in a GitHub `README.md` file, using Markdown for headings, lists, and inline math where appropriate.

Referred from:
- https://github.com/xunzheng/notears/blob/master/notears/utils.py
      
# Our results discription
We generated nonlinear(or linear)synthetic data based on various nodes(6, 9, 12) and edges (10, 15, 20), tested F1 score for time window length between 5-35 sequentially, and demonstrated the results on heatmaps Linear_Synthetic_Heatmap.pdf and Nonlinear_Synthetic_Heatmap.pdf. In comparison with artificial data generated from nonlinear and linear SEMs, the performances of ANM-NCPOP on linear data is better than nonlinear data. More precisely, the F1 score for data built on linear SEM is 0.4-0.6, but the F1 score for data built on nonlinear SEM is only 0.2-0.4.

