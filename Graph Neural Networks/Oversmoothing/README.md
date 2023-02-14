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

## DeepGCNs: Can GCNs Go as Deep as CNNs?
#### Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem
###### 7 Apr 2019

<p align="center">
  <img width="500" src="assets/oversmooth2.png">
</p>

**Abstract**: 

Convolutional Neural Networks (CNNs) achieve impressive performance in a wide variety of fields. Their success benefited from a massive boost when very deep CNN models were able to be reliably trained. Despite their merits, CNNs fail to properly address problems with non-Euclidean data. To overcome this challenge, Graph Convolutional Networks (GCNs) build graphs to represent non-Euclidean data, borrow concepts from CNNs, and apply them in training. GCNs show promising results, but they are usually limited to very shallow models due to the vanishing gradient problem. As a result, most state-of-the-art GCN models are no deeper than 3 or 4 layers. In this work, we present new ways to successfully train very deep GCNs. We do this by borrowing concepts from CNNs, specifically residual/dense connections and dilated convolutions, and adapting them to GCN architectures. Extensive experiments show the positive effect of these deep GCN frameworks. Finally, we use these new concepts to build a very deep 56-layer GCN, and show how it significantly boosts performance (+3.7% mIoU over state-of-the-art) in the task of point cloud semantic segmentation. We believe that the community can greatly benefit from this work, as it opens up many opportunities for advancing GCN-based research.

**Summary**:

The paper "DeepGCNs: Can GCNs Go as Deep as CNNs?" by Li et al. explores the limitations of Graph Convolutional Networks (GCNs) in training deep models, and presents new methods to overcome this challenge. By incorporating concepts from Convolutional Neural Networks (CNNs), such as residual/dense connections and dilated convolutions, the authors successfully train very deep GCNs with 56 layers, achieving improved performance in the task of point cloud semantic segmentation.

**Key insights and lessons learned**:
* GCNs face challenges in training deep models due to the vanishing gradient problem.
* Incorporating concepts from CNNs can help overcome these limitations and lead to deeper GCN models.
* Deep GCNs can achieve improved performance compared to shallow GCN models.

**Questions for the authors**:
* How does the choice of residual/dense connections and dilated convolutions affect the performance of deep GCNs?
* How does the proposed method compare to other methods for addressing the vanishing gradient problem in GCNs?
* Can the proposed method be applied to other tasks and domains beyond point cloud semantic segmentation?

**Future research directions**:
* Exploration of other techniques for addressing the vanishing gradient problem in GCNs.
* Extension of the proposed method to other types of non-Euclidean data beyond point clouds.
* Investigation of the scalability and efficiency of the proposed method for larger and more complex graph structures.

--- 

## Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View
#### Deli Chen, Yankai Lin, Wei Li, Peng Li, Jie Zhou, Xu Sun
###### 7 Sep 2019

<p align="center">
  <img width="500" src="assets/oversmooth3.png">
</p>

**Abstract**: 

Graph Neural Networks (GNNs) have achieved promising performance on a wide range of graph-based tasks. Despite their success, one severe limitation of GNNs is the over-smoothing issue (indistinguishable representations of nodes in different classes). In this work, we present a systematic and quantitative study on the over-smoothing issue of GNNs. First, we introduce two quantitative metrics, MAD and MADGap, to measure the smoothness and over-smoothness of the graph nodes representations, respectively. Then, we verify that smoothing is the nature of GNNs and the critical factor leading to over-smoothness is the low information-to-noise ratio of the message received by the nodes, which is partially determined by the graph topology. Finally, we propose two methods to alleviate the over-smoothing issue from the topological view: (1) MADReg which adds a MADGap-based regularizer to the training objective;(2) AdaGraph which optimizes the graph topology based on the model predictions. Extensive experiments on 7 widely-used graph datasets with 10 typical GNN models show that the two proposed methods are effective for relieving the over-smoothing issue, thus improving the performance of various GNN models.

**Summary**:

The paper "Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View" by Deli Chen et al. introduces two metrics to measure the over-smoothing issue in Graph Neural Networks (GNNs) and presents two methods to alleviate this issue by adding a regularizer and optimizing the graph topology. The authors verify that the critical factor causing over-smoothing is the low information-to-noise ratio of the messages received by the nodes, which is partially determined by the graph topology.

**Key insights and lessons learned**:
* Over-smoothing is a limitation of GNNs, where nodes in different classes are indistinguishable.
* The authors introduce two metrics, MAD and MADGap, to measure the smoothness and over-smoothing of the graph node representations, respectively.
* The critical factor causing over-smoothing is the low information-to-noise ratio of the messages received by the nodes, which is partially determined by the graph topology.

**Questions for the authors**:
* How would the proposed methods impact the performance of other types of graph-based models, such as GCN or GraphSAGE?
* What are the computational costs of using the proposed methods, particularly the AdaGraph method?
* How do the proposed methods handle the case where the graph topology is dynamic or changes over time?
* Can the proposed methods be extended to work with different types of graph-based datasets, such as weighted or attributed graphs?
* What is the effect of increasing the size of the training dataset on the performance of the proposed methods?

**Suggestions for future research**:
* Examining the impact of the proposed methods on graph-based tasks in domains other than those used in the experiments.
* Exploring alternative approaches for optimizing the graph topology.
* Evaluating the performance of the proposed methods on large-scale graph datasets.
* Investigating the combination of the proposed methods with other techniques for addressing the over-smoothing issue in GNNs.
* Testing the generalization ability of the proposed methods on unseen graphs.

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
