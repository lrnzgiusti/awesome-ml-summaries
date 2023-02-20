## Reconciling modern machine learning practice and the bias-variance trade-off
#### Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal
###### 28 Dec 2018 (v1)
**Abstract**:
Breakthroughs in machine learning are rapidly changing science and society, yet our fundamental understanding of this technology has lagged far behind. Indeed, one of the central tenets of the field, the bias-variance trade-off, appears to be at odds with the observed behavior of methods used in the modern machine learning practice. The bias-variance trade-off implies that a model should balance under-fitting and over-fitting: rich enough to express underlying structure in data, simple enough to avoid fitting spurious patterns. However, in the modern practice, very rich models such as neural networks are trained to exactly fit (i.e., interpolate) the data. Classically, such models would be considered over-fit, and yet they often obtain high accuracy on test data. This apparent contradiction has raised questions about the mathematical foundations of machine learning and their relevance to practitioners.
**Summary**:
The paper “Reconciling modern machine learning practice and the bias-variance trade-off” by Belkin et al. proposes a unified performance curve, the “double descent” curve, which reconciles the traditional bias-variance trade-off with the behavior of modern machine learning methods that often use very rich models such as neural networks trained to interpolate the data.
**Key insights and lessons learned**:
* The bias-variance trade-off has been a fundamental concept in machine learning, but it does not fully explain the behavior of modern machine learning methods.
* Modern machine learning methods often use very rich models that interpolate the data, which would be considered over-fit by classical standards, but still achieve high accuracy on test data.
* The “double descent” curve reconciles the traditional bias-variance trade-off with the behavior of modern machine learning methods, showing how increasing model capacity beyond the point of interpolation can lead to improved performance.
* The “double descent” curve provides a more nuanced understanding of the performance of machine learning models and suggests that the choice of model capacity depends on the task, the dataset, and the amount of available data.
**Questions for the authors**:
* What motivated you to investigate the behavior of modern machine learning methods and the bias-variance trade-off?
* How do you see the “double descent” curve impacting the practice of machine learning?
* Do you think the “double descent” curve can help guide the development of new machine learning models or the selection of existing models for specific tasks?
**Suggestions for future research**:
* Investigate the “double descent” curve on a wider range of machine learning tasks and datasets to further validate its applicability.
* Examine the impact of other factors, such as data quality and model interpretability, on the choice of model capacity and the behavior of the “double descent” curve.
* Explore the implications of the “double descent” curve for deep learning and other emerging areas of machine learning research.
---

## To understand deep learning we need to understand kernel learning
#### Mikhail Belkin, Siyuan Ma, Soumik Mandal
###### 5 Feb 2018 (v1)



**Abstract**:
Generalization performance of classifiers in deep learning has recently become a subject of intense study. Deep models, typically over-parametrized, tend to fit the training data exactly. Despite this “overfitting”, they perform well on test data, a phenomenon not yet fully understood.

**Summary**:

Certain key phenomena of deep learning are manifested similarly in kernel methods in the modern “overfitted” regime. The combination of the experimental and theoretical results presented in this paper indicates a need for new theoretical ideas for understanding properties of classical kernel methods. We argue that progress on understanding deep learning will be difficult until more tractable “shallow” kernel methods are better understood.* The paper “To understand deep learning we need to understand kernel learning” by Mikhail Belkin, Siyuan Ma, and Soumik Mandal argues that overfitting is not unique to deep learning and that kernel machines trained with zero classification or near-zero regression error perform well on test data, even with high levels of noise; the paper then provides a lower bound on the norm of zero loss solutions for smooth kernels and shows that they increase nearly exponentially with data size, which is difficult to reconcile with existing generalization bounds, and that non-smooth Laplacian kernels easily fit random labels, while fitting noisy data requires many more epochs for smooth Gaussian kernels.
* Key insights and lessons learned from the paper are that understanding kernel methods is essential to understanding deep learning, that strong performance of overfitted classifiers is not unique to deep learning, that existing generalization bounds are difficult to reconcile with lower bounds on the norm of zero loss solutions for smooth kernels, and that Laplacian kernels parallel ReLU neural networks in their ability to fit random labels.
* Some questions that I would like to ask the authors of this paper are: how can their findings inform the development of new generalization bounds that are better able to account for the norm of zero loss solutions for smooth kernels? Can the insights gained from studying kernel methods shed light on the ways in which deep learning models are able to achieve strong performance on test data despite overfitting the training data? What are some possible applications of the insights gained from this paper in fields such as computer vision, natural language processing, and speech recognition?
* Some suggestions for related topics or future research directions based on the content of the paper are: exploring the relationship between kernel methods and deep learning in more depth, developing new algorithms that are able to leverage the insights gained from this paper to improve the generalization performance of deep learning models, investigating the role of noise in the training data in determining the generalization performance of deep learning models, and examining the implications of these findings for the development of explainable AI.

---


## On the linearity of large non-linear models: when and why the tangent kernel is constant
#### Chaoyue Liu, Libin Zhu, Mikhail Belkin
###### 2 Oct 2020 (v1)
**Abstract**:
The goal of this work is to shed light on the remarkable phenomenon of transition to linearity of certain neural networks as their width approaches infinity. We show that the transition to linearity of the model and, equivalently, constancy of the (neural) tangent kernel (NTK) result from the scaling properties of the norm of the Hessian matrix of the network as a function of the network width. We present a general framework for understanding the constancy of the tangent kernel via Hessian scaling applicable to the standard classes of neural networks. Our analysis provides a new perspective on the phenomenon of constant tangent kernel, which is different from the widely accepted “lazy training”. Furthermore, we show that the transition to linearity is not a general property of wide neural networks and does not hold when the last layer of the network is non-linear. It is also not necessary for successful optimization by gradient descent.
**Summary**:
The paper “On the linearity of large non-linear models: when and why the tangent kernel is constant” by Liu, Zhu, and Belkin aims to explain the phenomenon of transition to linearity of certain neural networks as their width approaches infinity by showing that it results from the scaling properties of the norm of the Hessian matrix of the network as a function of the network width, and presents a general framework for understanding the constancy of the tangent kernel via Hessian scaling applicable to standard classes of neural networks.
**Key insights and lessons learned from the paper are**:
* The transition to linearity of the model and the constancy of the neural tangent kernel (NTK) result from the scaling properties of the norm of the Hessian matrix of the network as a function of the network width.
* The constancy of the tangent kernel can be explained by the Hessian scaling framework, which is different from the widely accepted “lazy training” perspective.
* The transition to linearity is not a general property of wide neural networks and does not hold when the last layer of the network is non-linear.
* The transition to linearity is not necessary for successful optimization by gradient descent.
**Some questions that could be asked to the authors are**:
* What motivated you to investigate the phenomenon of transition to linearity of neural networks, and why do you think it is important to understand it?
* How did you come up with the idea of using the Hessian scaling framework to explain the constancy of the tangent kernel, and what challenges did you face in developing this framework?
* What are some potential applications or implications of your findings for the field of machine learning?
* In what ways do you think your work can be extended or improved, and what are some promising directions for future research on this topic?
* Are there any limitations or assumptions of your study that you think are important to acknowledge, and how might they affect the generalizability of your results?
**Some suggestions for related topics or future research directions based on the content of the paper are**:
* Investigating the Hessian properties of other types of neural networks beyond the standard classes considered in the paper, and exploring how they relate to the constancy of the tangent kernel and the transition to linearity.
* Examining how the transition to linearity and the constancy of the tangent kernel can affect the generalization performance of neural networks, and developing theoretical bounds or empirical methods to quantify this effect.
* Exploring how the Hessian scaling framework can be used to guide the design or optimization of neural networks, for example by identifying optimal architectures or hyperparameters based on their Hessian properties.
* Studying the connection between the Hessian properties of neural networks and other aspects of deep learning, such as adversarial robustness, transfer learning, or unsupervised representation learning.
* Investigating the potential implications of the transition to linearity and the constancy of the tangent kernel for other areas of science or engineering, such as physics, optimization, or control theory.
--- 


## Generalization bounds for deep learning
#### Guillermo Valle-Pérez, Ard A. Louis
###### 7 Dec 2020 (v1)
**Abstract**:
Generalization in deep learning has been the topic of much recent theoretical and empirical research. Here we introduce desiderata for techniques that predict generalization errors for deep learning models in supervised learning. Such predictions should 1) scale correctly with data complexity; 2) scale correctly with training set size; 3) capture differences between architectures; 4) capture differences between optimization algorithms; 5) be quantitatively not too far from the true error (in particular, be non-vacuous); 6) be efficiently computable; and 7) be rigorous. We focus on generalization error upper bounds, and introduce a categorisation of bounds depending on assumptions on the algorithm and data. We review a wide range of existing approaches, from classical VC dimension to recent PAC-Bayesian bounds, commenting on how well they perform against the desiderata.
**Summary**:
The paper “Generalization bounds for deep learning” by Guillermo Valle-Pérez and Ard A. Louis introduces criteria for predicting generalization errors in deep learning models and reviews a range of existing approaches that meet these criteria, including classical VC dimension and recent PAC-Bayesian bounds, before proposing a marginal-likelihood PAC-Bayesian bound that is optimal up to a multiplicative constant in the asymptotic limit of large training sets and typically found in practice for deep learning problems.
* Key insights and lessons learned from the paper include the importance of developing techniques that accurately predict generalization errors for deep learning models and the need to ensure that such predictions are scalable, architecture- and algorithm-dependent, non-vacuous, efficiently computable, and rigorous. The paper demonstrates that existing approaches, including classical VC dimension and recent PAC-Bayesian bounds, perform well against these desiderata, but propose a novel approach, the marginal-likelihood PAC-Bayesian bound, that is optimal up to a multiplicative constant in the asymptotic limit of large training sets.
* Questions for the authors might include: How might the proposed marginal-likelihood PAC-Bayesian bound be modified to address cases in which the learning curve does not follow a power law? To what extent do the desiderata identified in the paper apply to unsupervised deep learning models? How might the proposed framework for generalization error upper bounds be extended to deep reinforcement learning models?
* Suggestions for related topics or future research directions might include: Developing techniques for predicting generalization errors in deep learning models that can account for additional factors such as noise in the data or variation in network connectivity; exploring the extent to which the desiderata identified in the paper apply to deep learning models trained on non-image data; developing techniques for predicting generalization errors in deep reinforcement learning models that are scalable, architecture- and algorithm-dependent, non-vacuous, efficient, and rigorous.
---

## On the linearity of large non-linear models: when and why the tangent kernel is constant
#### Chaoyue Liu, Libin Zhu, Mikhail Belkin
###### 2 Oct 2020 (v1)
**Abstract**:
The goal of this work is to shed light on the remarkable phenomenon of transition to linearity of certain neural networks as their width approaches infinity. We show that the transition to linearity of the model and, equivalently, constancy of the (neural) tangent kernel (NTK) result from the scaling properties of the norm of the Hessian matrix of the network as a function of the network width. We present a general framework for understanding the constancy of the tangent kernel via Hessian scaling applicable to the standard classes of neural networks. Our analysis provides a new perspective on the phenomenon of constant tangent kernel, which is different from the widely accepted “lazy training”. Furthermore, we show that the transition to linearity is not a general property of wide neural networks and does not hold when the last layer of the network is non-linear. It is also not necessary for successful optimization by gradient descent.
**Summary**:
The paper “On the linearity of large non-linear models: when and why the tangent kernel is constant” by Liu, Zhu, and Belkin aims to explain the phenomenon of transition to linearity of certain neural networks as their width approaches infinity by showing that it results from the scaling properties of the norm of the Hessian matrix of the network as a function of the network width, and presents a general framework for understanding the constancy of the tangent kernel via Hessian scaling applicable to standard classes of neural networks.
**Key insights and lessons learned from the paper are**:
* The transition to linearity of the model and the constancy of the neural tangent kernel (NTK) result from the scaling properties of the norm of the Hessian matrix of the network as a function of the network width.
* The constancy of the tangent kernel can be explained by the Hessian scaling framework, which is different from the widely accepted “lazy training” perspective.
* The transition to linearity is not a general property of wide neural networks and does not hold when the last layer of the network is non-linear.
* The transition to linearity is not necessary for successful optimization by gradient descent.
**Some questions that could be asked to the authors are**:
* What motivated you to investigate the phenomenon of transition to linearity of neural networks, and why do you think it is important to understand it?
* How did you come up with the idea of using the Hessian scaling framework to explain the constancy of the tangent kernel, and what challenges did you face in developing this framework?
* What are some potential applications or implications of your findings for the field of machine learning?
* In what ways do you think your work can be extended or improved, and what are some promising directions for future research on this topic?
* Are there any limitations or assumptions of your study that you think are important to acknowledge, and how might they affect the generalizability of your results?
**Some suggestions for related topics or future research directions based on the content of the paper are**:
* Investigating the Hessian properties of other types of neural networks beyond the standard classes considered in the paper, and exploring how they relate to the constancy of the tangent kernel and the transition to linearity.
* Examining how the transition to linearity and the constancy of the tangent kernel can affect the generalization performance of neural networks, and developing theoretical bounds or empirical methods to quantify this effect.
* Exploring how the Hessian scaling framework can be used to guide the design or optimization of neural networks, for example by identifying optimal architectures or hyperparameters based on their Hessian properties.
* Studying the connection between the Hessian properties of neural networks and other aspects of deep learning, such as adversarial robustness, transfer learning, or unsupervised representation learning.
* Investigating the potential implications of the transition to linearity and the constancy of the tangent kernel for other areas of science or engineering, such as physics, optimization, or control theory.
---