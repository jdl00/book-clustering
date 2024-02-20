# **Book Recommendation System**

A project leveraging MultiHead Attention and unsupervised techniques to explore book recommendations.

## **Overview**

This project investigates the development of a book recommendation system. The core approach is based on content-based similarity, primarily utilizing MultiHead Attention to analyze book descriptions. Clustering or other unsupervised techniques might be included for grouping books with similar themes and characteristics.

## **Dataset**

* **Source:** The book description dataset originates from [GoodReads 100k books](https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k)
* **Composition:** The dataset encompasses descriptions for approximately 100k books. Additionally, it includes the following attributes (if available):
    * Title
    * Author
    * Genre
    * Description

## **Dependencies**

### **Python:** Version 3.7 or later is recommended.
###  **Libraries:**
    * transformers (Hugging Face)
    * pandas
    * NumPy
    * torch
    * tqdm


## **Architecture**

*All layers are located within model.py*

1. [Embeddings](#embeddings)
2. [MultiHead Attention](#multihead-attention)
3. [Embeddings](#k-means-clustering)

### **Embeddings**

#### **Purpose**:

 This module generates meaningful vector representations of book descriptions. It leverages techniques inspired by BERT-like models.

#### **Components**:
- **Word Embeddings**: Maps individual words in a description to dense vectors.

- **Positional Embeddings**: Encodes information about the order of words within the description.

- **LayerNorm**: Normalizes the embeddings to improve training stability.

- **Dropout**: A regularization technique to prevent overfitting.


### **MultiHead Attention**

#### **Purpose**:

This attention mechanism refines the book description embeddings from the BERTEmbeddings module. It emphasizes the most relevant and interconnected aspects of the descriptions.

#### **Mechanism**:

Multihead attention learns to focus on different "heads" (subspaces) within the embeddings, promoting a more nuanced understanding of relationships within the text.

### K Means Clustering

#### **Purpose**:

This module performs clustering on the processed book representations. It aims to group books with similar descriptions and themes.

#### **Algorithm**:

K-Means is an iterative clustering algorithm that assigns data points (book representations) to clusters based on their distance to pre-determined cluster centers (centroids).

