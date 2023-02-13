## Importance Weighted Autoencoders
#### Yuri Burda, Roger Grosse, Ruslan Salakhutdinov
###### 1 Sep 2015

**Abstract**:
The variational autoencoder (VAE; Kingma, Welling (2014)) is a recently proposed generative model pairing a top-down generative network with a bottom-up recognition network which approximates posterior inference. It typically makes strong assumptions about posterior inference, for instance that the posterior distribution is approximately factorial, and that its parameters can be approximated with nonlinear regression from the observations. As we show empirically, the VAE objective can lead to overly simplified representations which fail to use the network's entire modeling capacity. We present the importance weighted autoencoder (IWAE), a generative model with the same architecture as the VAE, but which uses a strictly tighter log-likelihood lower bound derived from importance weighting. In the IWAE, the recognition network uses multiple samples to approximate the posterior, giving it increased flexibility to model complex posteriors which do not fit the VAE modeling assumptions. We show empirically that IWAEs learn richer latent space representations than VAEs, leading to improved test log-likelihood on density estimation benchmarks.

**Summary**:
The paper introduces the Importance Weighted Autoencoder (IWAE), which is a generative model with the same architecture as a Variational Autoencoder (VAE), but with a tighter log-likelihood lower bound derived from importance weighting. The IWAE allows for increased flexibility in approximating the posterior distribution and leads to a richer latent space representation, leading to improved results on density estimation benchmarks.

**Key Insights and Lessons Learned**:

* The VAE objective can lead to overly simplified representations and fail to use the entire modeling capacity.
* The IWAE uses multiple samples to approximate the posterior, leading to increased flexibility in modeling complex posteriors.
* The IWAE learns richer latent space representations and results in improved test log-likelihood on density estimation benchmarks.

**Questions for the Authors**:

* Can you explain the limitations of VAEs that led you to propose IWAEs as an alternative?
* How do the results of IWAEs compare to other generative models in terms of both computational efficiency and accuracy?
* How would the results change if the IWAE architecture were applied to different types of data or tasks?
* Can you discuss any potential implications of the IWAE for real-world applications in areas such as computer vision or natural language processing?
* Have you considered using IWAEs in combination with other generative models or techniques, and if so, what kind of results have you seen?

**Future Research Directions**:

* Applying IWAEs to different types of data, such as images or text, to see if the improved latent space representation leads to improved performance on downstream tasks.
* Investigating the impact of different hyperparameters, such as the number of importance samples, on the performance of IWAEs.
* Exploring the use of IWAEs in combination with other generative models or techniques.
* Evaluating the scalability of IWAEs on large-scale datasets.
* Investigating the potential for using IWAEs for unsupervised representation learning.


--- 


## Auto-Encoding Variational Bayes
#### Diederik P Kingma, Max Welling
###### 20 Dec 2013


<p align="center">
  <img width="500" src="assets/vae.png">
</p>


**Abstract**:

How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions are two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results.

**Summary**:

The paper "Auto-Encoding Variational Bayes" introduces a stochastic variational inference and learning algorithm for directed probabilistic models with continuous latent variables and intractable posterior distributions. The authors provide a reparameterization of the variational lower bound and show how to fit an approximate inference model to the intractable posterior.

**Key Insights and Lessons Learned**:

* A reparameterization of the variational lower bound can be optimized using standard stochastic gradient methods.
* Fitting an approximate inference model to the intractable posterior can make posterior inference more efficient.
* The authors show theoretical and experimental results to support their proposed method.

**Questions for the authors**:

* Can you provide further explanation of the conditions under which the proposed method will work in the intractable case?
* How does this method compare to other existing methods in terms of computational efficiency and accuracy?
* Have you considered any extensions of this method to handle non-i.i.d datasets?
* How does this method handle high dimensional continuous latent variables?
Can you discuss any limitations or limitations of the method?

**Future research directions**:

* Application of the proposed method to other types of probabilistic models.
* Extension of the method to handle non-continuous latent variables.
* Comparison of the method to other Bayesian deep learning methods.
* Investigation of how to handle cases where the intractable posterior distributions are not i.i.d.
* Exploration of how to handle high dimensional continuous latent variables. 



--- 


## Variational Auto-encoded Deep Gaussian Processes
#### Zhenwen Dai, Andreas Damianou, Javier González, Neil Lawrence
###### 19 Nov 2015


<p align="center">
  <img width="500" src="assets/vaegp.png">
</p>



**Abstract**:

We develop a scalable deep non-parametric generative model by augmenting deep Gaussian processes with a recognition model. Inference is performed in a novel scalable variational framework where the variational posterior distributions are reparametrized through a multilayer perceptron. The key aspect of this reformulation is that it prevents the proliferation of variational parameters which otherwise grow linearly in proportion to the sample size. We derive a new formulation of the variational lower bound that allows us to distribute most of the computation in a way that enables to handle datasets of the size of mainstream deep learning tasks. We show the efficacy of the method on a variety of challenges including deep unsupervised learning and deep Bayesian optimization

**Summary**:

The paper "Variational Auto-encoded Deep Gaussian Processes" by Zhenwen Dai, Andreas Damianou, Javier González, and Neil Lawrence introduces a scalable deep non-parametric generative model that combines deep Gaussian processes with a recognition model. Inference is performed in a scalable variational framework, where the variational posterior distributions are reparameterized through a multilayer perceptron, resulting in a new formulation of the variational lower bound. The method is demonstrated to be effective in various deep unsupervised learning and deep Bayesian optimization tasks.

**Key Insights and Lessons**:

* A scalable deep non-parametric generative model is introduced.
* Inference is performed in a scalable variational framework.
* The variational posterior distributions are reparameterized through a multilayer perceptron.
* A new formulation of the variational lower bound is derived.
* The method is demonstrated to be effective in deep unsupervised learning and deep Bayesian optimization tasks.

**Questions for the authors**:

* How does the new formulation of the variational lower bound compare to existing methods in terms of computational efficiency?
* What is the scalability of the method compared to other deep generative models?
* Can you discuss the limitations of the method and how they can be addressed in future work?
* How does the method handle multi-modal data distributions?
* How does the performance of the method compare to other deep generative models in terms of sample quality?


**Future Research Directions**:

* Extension of the method to handle multi-modal data distributions.
* Comparison of the method with other deep generative models in terms of sample quality.
* Investigation of the scalability of the method compared to other deep generative models.
* Exploration of the limitations of the method and how they can be addressed in future work.
* Application of the method in various fields, such as computer vision and speech recognition


--- 


## Neural Variational Inference and Learning in Belief Networks
#### Andriy Mnih, Karol Gregor
###### 31 Jan 2014


**Abstract**:


Highly expressive directed latent variable models, such as sigmoid belief networks, are difficult to train on large datasets because exact inference in them is intractable and none of the approximate inference methods that have been applied to them scale well. We propose a fast non-iterative approximate inference method that uses a feedforward network to implement efficient exact sampling from the variational posterior. The model and this inference network are trained jointly by maximizing a variational lower bound on the log-likelihood. Although the naive estimator of the inference model gradient is too high-variance to be useful, we make it practical by applying several straightforward model-independent variance reduction techniques. Applying our approach to training sigmoid belief networks and deep autoregressive networks, we show that it outperforms the wake-sleep algorithm on MNIST and achieves state-of-the-art results on the Reuters RCV1 document dataset.


**Summary**:

The paper "Neural Variational Inference and Learning in Belief Networks" by Andriy Mnih and Karol Gregor proposes a fast non-iterative approximate inference method for highly expressive directed latent variable models, such as sigmoid belief networks. The method uses a feedforward network to implement efficient exact sampling from the variational posterior and the model and inference network are trained jointly. The authors show that the proposed method outperforms the wake-sleep algorithm on MNIST and achieves state-of-the-art results on the Reuters RCV1 document dataset.

**Key Insights and Lessons Learned**:

* Exact inference in highly expressive directed latent variable models is intractable and existing approximate methods do not scale well.
* A feedforward network can be used to implement efficient exact sampling from the variational posterior.
* The model and inference network can be trained jointly by maximizing a variational lower bound on the log-likelihood.
* Variance reduction techniques can make the naive estimator of the inference model gradient practical.
* The proposed method outperforms the wake-sleep algorithm on MNIST and achieves state-of-the-art results on the Reuters RCV1 document dataset.


**Questions for the authors**:

* How does the proposed method compare to other variational inference methods?
* Can the method be extended to handle models with more complex structure?
* How sensitive is the performance of the method to the choice of variance reduction techniques?
* How does the method handle missing data?
* Have the results been verified on other datasets?

**Related Topics/Future Research Directions**:

* Extension of the method to handle models with more complex structure.
* Comparison with other variational inference methods.
* Application of the method to other domains, such as natural language processing.
* Exploration of other variance reduction techniques.
* Extension of the method to handle missing data.



--- 


## Variational inference for Monte Carlo objectives
#### Andriy Mnih, Danilo J. Rezende
###### 22 Feb 2016

<p align="center">
  <img width="500" src="assets/vaemonte.png">
</p>


**Abstract**:

Recent progress in deep latent variable models has largely been driven by the development of flexible and scalable variational inference methods. Variational training of this type involves maximizing a lower bound on the log-likelihood, using samples from the variational posterior to compute the required gradients. Recently, Burda et al. (2016) have derived a tighter lower bound using a multi-sample importance sampling estimate of the likelihood and showed that optimizing it yields models that use more of their capacity and achieve higher likelihoods. This development showed the importance of such multi-sample objectives and explained the success of several related approaches.
We extend the multi-sample approach to discrete latent variables and analyze the difficulty encountered when estimating the gradients involved. We then develop the first unbiased gradient estimator designed for importance-sampled objectives and evaluate it at training generative and structured output prediction models. The resulting estimator, which is based on low-variance per-sample learning signals, is both simpler and more effective than the NVIL estimator proposed for the single-sample variational objective, and is competitive with the currently used biased estimators.


**Summary**:

The paper presents a new method for variational training of deep latent variable models using a multi-sample importance sampling estimate of the likelihood. The authors derive a tight lower bound and develop an unbiased gradient estimator based on low-variance per-sample learning signals. They evaluate this estimator on generative and structured output prediction models and show that it is both simple and effective, competitive with currently used biased estimators.

**Key insights and lessons learned from the paper**:
* Variational training of deep latent variable models involves maximizing a lower bound on the log-likelihood
* The use of multi-sample objectives can lead to models that use more of their capacity and achieve higher likelihoods
* The authors develop a new unbiased gradient estimator designed for importance-sampled objectives

**Questions to ask the authors**:
* What motivated the development of a new gradient estimator specifically for importance-sampled objectives?
* Can the method be extended to other types of deep latent variable models besides generative and structured output prediction models?
* How does the performance of the proposed estimator compare with other existing methods on large-scale datasets?
* How does the proposed method deal with issues such as high variance in the importance weights?
* Can the method be used in other areas besides deep latent variable models, such as reinforcement learning or Bayesian neural networks?

**Suggestions for related topics or future research directions**:
* Extension of the method to handle non-Gaussian variational posteriors and likelihoods
* Applications of the method in reinforcement learning and Bayesian neural networks
* Analysis of the scalability of the method to large datasets
* Exploration of different strategies for choosing importance sampling distributions
* Comparison of the method with other gradient-based variational inference methods, such as the score function estimator.
