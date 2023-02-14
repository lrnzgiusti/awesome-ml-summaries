## Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning
#### Qimai Li, Zhichao Han, Xiao-Ming Wu
###### 22 Jan 2018

<p align="center">
  <img width="500" src="assets/oversmooth1.png">
</p>

**Abstract**: 

Many interesting problems in machine learning are being revisited with new deep learning tools. For graph-based semisupervised learning, a recent important development is graph convolutional networks (GCNs), which nicely integrate local vertex features and graph topology in the convolutional layers. Although the GCN model compares favorably with other state-of-the-art methods, its mechanisms are not clear and it still requires a considerable amount of labeled data for validation and model selection. In this paper, we develop deeper insights into the GCN model and address its fundamental limits. First, we show that the graph convolution of the GCN model is actually a special form of Laplacian smoothing, which is the key reason why GCNs work, but it also brings potential concerns of over-smoothing with many convolutional layers. Second, to overcome the limits of the GCN model with shallow architectures, we propose both co-training and self-training approaches to train GCNs. Our approaches significantly improve GCNs in learning with very few labels, and exempt them from requiring additional labels for validation. Extensive experiments on benchmarks have verified our theory and proposals.

**Summary**:

The paper provides deeper insights into Graph Convolutional Networks (GCNs) for semi-supervised learning by demonstrating that graph convolution in GCNs is a special form of Laplacian smoothing, and addressing the limits of shallow GCN architectures with co-training and self-training approaches. The authors show that these approaches significantly improve GCNs in learning with very few labels, and exempt them from requiring additional labels for validation.

**Key insights and lessons learned**:
* The graph convolution in GCNs is a special form of Laplacian smoothing, which is the key to their effectiveness, but also raises concerns of over-smoothing with multiple convolutional layers.
* Shallow GCN architectures have limitations that can be overcome with co-training and self-training approaches.
* These approaches significantly improve GCN performance in learning with very few labels and don't require additional labeled data for validation.

**Questions for the authors**:
* How do the co-training and self-training approaches compare with other state-of-the-art graph-based semi-supervised learning methods?
* Can the proposed methods be applied to other graph-based learning tasks, such as node classification and graph classification?
* What is the computational cost of implementing the co-training and self-training approaches, compared to traditional GCN models?
* How do the performance results change with different graph structures and different numbers of labeled data points?
* What are the potential future directions for improving GCN-based semi-supervised learning?

**Suggestions for related topics or future research directions**:
* Improving the robustness and generalization of GCN-based models.
* Combining GCN-based models with other graph-based learning approaches, such as graph attention networks and random walk-based methods.
* Exploring the application of GCN-based models to real-world problems, such as social network analysis and recommendation systems.
* Studying the interpretability of GCN-based models and developing methods to visualize the learned graph representations.
* Investigating the impact of different graph structures and graph representations on the performance of GCN-based models.

--- 

## 
#### 
###### 

<p align="center">
  <img width="500" src="assets/.png">
</p>

**Abstract**: 


**Summary**:


--- 

## 
#### 
###### 

<p align="center">
  <img width="500" src="assets/.png">
</p>

**Abstract**: 


**Summary**:


--- 

## 
#### 
###### 

<p align="center">
  <img width="500" src="assets/.png">
</p>

**Abstract**: 


**Summary**:


--- 

## 
#### 
###### 

<p align="center">
  <img width="500" src="assets/.png">
</p>

**Abstract**: 


**Summary**:


--- 
