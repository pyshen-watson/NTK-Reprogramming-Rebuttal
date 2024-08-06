
# [NIPS2024] NTK Rebuttal
## Author Rebuttal
Dear Area Chairs and Senior Area Chairs,

We truly appreciate your dedication and time spent coordinating our reviews. In our work, the reviewers'  concerns are mainly about "whether our NTK analysis can be extended to other large-scale models and large-scale dataset", "reasonability of untrainable output mapping assumption" and "confusion of contribution 3 and conclusion".

For "whether our NTK analysis can be extended to other large-scale models and large-scale datasets", we have performed additional experiments on ResNet/VGG trained on ImageNet-10 to respond to reviewers' concern. The experimental results further support our theoretical predictions. 


For "reasonability of untrainable output mapping assumption", we would like to say our work is the first trial to analyze model reprogramming (MR), so we choose the simplest setting (untrainable output mapping). Beside, even in the simplest setting, there are still many MR applications satisfying our assumptions. Hence, we believe the assumption in our paper is reasonable.

For "confusion of contribution 3 and conclusion", we want to clarify that the inconsistency between experimental results of VP and  our theoretical predictions could be attributed to the insufficiently large width, which deviates our assumption.We have conducted additional experiments on models with larger width. The corresponding experimental results support our hypothesis, and hence our theory is still valid for VP. We apologize for the inaccurate words in contribution 3. which is misleading. 

In summary, we have responded to all reviewer's concerns and additional experiments further support our NTK analysis.

Best regards,
Authors







## Reviewer 1

### Weakness-1
The proposed theory using NTK does not comply with the many successful MR applications [1-6] in computer vision tasks, which questions whether the assumptions made by the authors are generic or not. When it comes to reprogramming in computer vision, VPs are a defacto choice of input transformation. The provided explanation is not convincing.

>To our knowledge, this paper is the first formal analysis of MR; we start by considering the simplest setting. Our analysis does not apply to all kinds of MR. Nevertheless, many MR applications such as BAR (Tsai et al., 2020) and V2S (Yang et al., 2021) still satisfy our theory's assumptions. Besides, it is possible to extend our theory to more general cases ($b$ is trainable). When $b$ is trainable, NTK Theory is still applicable. We can still analyze MR through the eigenspectrum of NTK, but unfortunately $\hat\Theta_T(x, x')$ can no longer be expressed as Eq. (16); $\hat\Theta_T(x, x')$ will be in a more complicated form, which deserves to be investigated in the future. 

### Weakness.2
The authors failed to explain why VP fails, and why it results in a less pronounced increase in minimum eigenvalues compared to FC layers. Even the VP+FC result in Table 6 could be influenced by the stability of FC. This questions the experimental validation of the paper.

>In fact, in lines 316~330, we have provided a potential explanation on why VP fails. In particular, different input transformation layers (FC, VP, and VP+FC) have different cirterions of ``suffiicent large width of target model.'' Thus, VP fails due to the width of the target model (input dimension of the source model) is insufficiently large. Here, the width of the target model needs to be sufficiently large to ensure that the NTK theory is valid. We conduct additional experiments to validate the above argument, where we change the source dataset from CIFAR-10 (image size: 32x32x3) to ImageNet-10 (image size: 112x112x3), which increases the input dimension of the source model.

    TBD Experiments on source dataset: ImageNet-10 (Wei-Chen)

>The experimental results show that our theoretical prediction is valid for the target model with larger width (larger source model input dimension), which supports our potential explanation of why VP fails. 

### Weakness.3
The paper's presentation of results requires improvement. The authors have condensed their findings to accommodate extensive theoretical content, which has compromised readability. Visualizations and tables need enhancement; for instance, the figures lack detail, possibly they were rushed in their creation.

>In our revised paper, we will rewrite and reorganize our findings to improve the overall readability. We will also revise the figures and tables, ensuring that they are informative. 


### Weakness.4
Authors should make a similar effort in providing experimental analysis as they did for theoretical exploration.

>We have conducted more experiments under different losses, different model structures, and different source datasets. The results below are conducted on DNN for CIFAR10 with cross-encropy loss. 
>
>The results below are conducted on DNN for CIFAR10 with mean square error. 
>The results below are conducted on CNN (width w=32) for CIFAR10 with mean square error. 
>The results below are conducted on CNN (width w=32) for CIFAR10 with mean square error. 
>The results below are conducted on CNN (width w=512) for CIFAR10 with mean square error. 
>The results below are conducted on CNN (width w=512) for CIFAR10 with mean square error. 
>All the experiments further support our theoretical prediction.

### Question.1
Could Neural Tangent Kernel (NTK) theory be employed to elucidate the behavior of Model Reprogramming (MR) with larger vision models such as ResNet, as discussed in?

>Yes, NTK theory can be employed on other larger vision models like ResNet, VGG, transformer, etc. We conduct experiments with ResNet/VGG as source model and ImageNet-10 as source dataset. The  results are shown below.

    TBD: ResNet/VGG Eigenvalue, Prompt (Wei-Chen)
    
>The results show that the minimum eigenvalue of the source model increases as we make the source model deeper. On the other hand, the target loss decreases as we make the source model deeper. Hence, the above results validate our theoretical prediction.

### Question.2
Q. I am particularly intrigued as current implementations of reprogramming predominantly utilize transformer-based source models. Is NTK theory applicable to architectures employing self-attention mechanisms and contrastive loss, such as CLIP?

>NTK theory is applicable to transformer-based source models. [1] has proved that any modern model satisfying "Simple gradient independence assumption (GIA) Check" has NTK. As for CLIP, NTK theory is not immediately applicable, due to the type of contrastive loss (cosine similarity). However, developing $\ell_2$-norm-like contrastive loss to make NTK theory applicable on CLIP would be an interesting research direction.
>
>[1] Greg Yang. Tensor programs II: Neural tangent kernel for any architecture. arXiv preprint arXiv:2006.14548, 2020.

## Reviewer 2

### Weakness.1 
Q. In the numerical results section 5.2, there is no notion of the width of the layers. This is while the NTK analysis hold only when the width is large enough.

>The width of the neural network is defined as the number of neurons of each hidden layer.

### Weakness.2
Q. In section 5.2 please clarify how did you compute the NTK. Did you use predefined libraries for computing them or the empirical NTK methods? please clearly mention your method and refer to proper literature.

>In our experiment, at first, we randomly sampled 10 images per class from the source dataset. Then, we computed the NTK matrix and corresponding minimum eigenvalue through this [library](https://github.com/google/neural-tangents). We repeated this procedure 3 times, and used the average of these 3 minimum eigenvalue of NTK matrix as our experimental results, as shown in Table 1. 

### Weakness.3, 4, 5, 6
Q. Figure 1 is very uninformative. Please illustrate some mathematical concepts from the paper in this figure. My suggestion is to illustrate the dimensions of input/output layers as well as the distribution of source and target model. A figure indicating f_S, f_T,c_T,c_S, a, b, d_T, etc will not only enrich this figure but also will be helpful in understanding the paper. some shapes like the ones on the bottom right are not meaningful nor the arrows are in place.

>In the revised paper, we will follow your suggestions to revise Figures 1, 2, 3, and to ensure all abbreviations are well defined. We will also fix typos and grammatical errors. 
   

### Weakness.7, Question.2
Q. On page 2, the contribution noted in 3 is not comprehensible. Your analysis fails for visual classification but when a fully connected (FC) layer is used the the inference holds? isn’t the NTK analysis independent of the type of layers used to build the target model? relating it to the scale of VP and FC is also not clear. Either way please clarify contribution 3.

>In our contribution 3, we want to convey the following things.

>Corollary 1 claims that the loss of target model evaluated on target distribution will decrease as we make the source model deeper. However, the results in Table 4 and Table 6 suggest that such a theoretical prediction holds when we choose FC as input transformation layer, but fails when we choose VP as input transformation layer. We hypothesize that this failure is due to the width of the target model (input dimension of the source model) is not large enough. In NTK theory, the model width needs to be sufficiently large, such that the target model's behavior can be characterized by NTK. In our experiments, the model width is $32\times 32\times 3$, which is sufficiently large for the target model with FC as the input transformation layer. So, the results about FC support our theoretical prediction. On the contrary, $32\times 32\times 3$ is not large enough for the target model with VP as the input transformation layer, so our theoretical prediction fails on the experimental results about VP.

### Weakness.8
Q. Equation (16) is not accurate, I do not see any dependency on x’ whereas it should be dependent on x’. Moreover, please clarify how you derive it.

>A. There is a typo in Eq. (16); it should be:
\begin{equation}
b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x))
\nabla_{\theta_A} a(x)
\{b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x'))
\nabla_{\theta_A} a(x')\}^T.
\end{equation}
>The derivation of Eq. (16) will be included in the revised paper. 

### Weakness.9
Q. Do the new Kernels you defined in (17)-(18) represent a meaningful concept? if yes, you could possibly name them while introducing them. For instance, can we say is the NTK w.r.t to the input layer and same for others?

>Yes, they are meaningful. We will name them while introducing them in the revised paper.

### Weakness.10
Q. The regression analysis in figure 3, is based on very sparse points. Is there any reason, you could not include more number of runs and fit a line to them? Nonetheless VP+FC on CNN network seem to be consistent with your analysis while VP is not. Is there a reason for that?

>A. The reason that we did not include more runs is the insufficient computing resource. We have conducted more experiments; the results are shown below.

    TBD CNN/DNN CE/MSE MEV (Wei-Chen)
    
>From the results, VP+FC appears to require less width in the target model to ensure NTK theory holds compared to VP, but more width compared to FC. This explains why FC on both DNN and CNN is consistent with our analysis, why VP on both DNN and CNN is inconsistent with our analysis, and why VP+FC on CNN is consistent with our analysis, whereas VP+FC on DNN is not.

### Weakness.11
Q. Given the importance of sensitivity of NTK analysis to width of the networks, I expected a set of figures dedicated to effect increasing and decreasing width on consistency of your analysis, which I saw only briefly discussed on page 9. All your figure/ tables are representing depth of networks, I believe you can add some informative figures discussing width.

>A. We have conducted experiments for different widths (input dimension of source model). We train the source models on ImageNet10 with the input size of 112, 56, 28, and train the target models with VP as the input transformation layer. The results are shown below.

    TBD ImageNet-10 112, 56, 28 (Wei-Chen)

### Question.1
Q. Fixing the output transformation layer “b” in the assumption 2 seems to be very limiting. If it is not trainable, then it might as well be considered as a layer inside the source model so why do you need to introduce it separately? My understanding is that this assumption made the derivations of NTK more feasible. However, wouldn’t it be feasible to fix “b”, do the analysis for “a” and then similarly fix “a” and do the NTK analysis for b? if this was your reasoning or not, either way please elaborate more on assumption 2 and clarify in the paper why it is not a limiting assumption.

>Our assumption 2 requires three things: (1) Source model $f_S$ has large width and hence satisfies NTK theory. (2) Output mapping $b$ is a linear matrix. (3) Output mapping $b$ is untrainable.

>Our assumption is not a limiting assumption. For (1), we notice that the source model should be a big model in real-world MR application scenarios. So, the source model $f_S$ naturally has large width. Besides, many modern NN structures like ResNet, transformer, etc have NTK. [1] proved that any modern NN satisfying "Simple gradient independence assumption (GIA) check" has NTK. Hence, assuming source model $f_S$ satisfying NTK theory is reasonable. For (2) and (3), we require that $b$ is an untrainable linear matrix because such a setting is more feasible to do NTK analysis and many MR applications such as BAR (Tsai et al., 2020) and V2S (Yang et al., 2021) satisfy this requirement ($b$ is untrainable linear matrix). 

>Indeed, “fix output transformation layer $a$ and do the NTK analysis for output mapping $b$” is also feasible. However there is relatively less research under this setting. That is why we set "fix $b$ and do NTK analysis for $a$"" rather than "fix $b$ and do NTK analysis for $a$".

[1] Yang, G. (2020). Tensor programs ii: Neural tangent kernel for any architecture. arXiv preprint arXiv:2006.14548.  

### Question.3
Q. What does notion of percentage “%” mean for minimum eigen value?(Fig. 2)

>A. It indicates that the unit of the minimum eigenvalue is 1e-2. 


### Question.4
Q. What is the width of source model and a and b layers in your simulation? please mention it in numerical results section and how they satisfy the large width assumption on NTK.

>The width of the neural network is defined as the number of neurons of the hidden layers. For $b$, since $b$ is a single linear matrix in our setting, $b$ does not have width. For $a$, when $a$ is VP or FC, $a$ only has one input layer and one output layer, so $a$ also does not have width. However, when $a$ is VP+FC, $a$ has one hidden layer, which has $32\times 32\times 3$ (input dimension of the source model) neurons. So, the width of $a$ is $32\times 32\times 3$.

>For the source model $f_S$, the width of $f_S$ depends on the model structure of $f_S$. The width will be $2048$ when $f_S$ is DNN. Especially, when $f_S$ is CNN, the width is considered as $channel \times filter$. In our experiments, CNN is considered to have $512$ channels and $(3, 3)$ filter. So, the width will be $512\times 3^2$.

>We follow [1] to set the width of the source model. [1] suggested that the training dynamics of CNN and DNN can be well approximated by NTK theory. So, our source model $f_S$ satisfies has a large width.

[1] J. Lee, L. Xiao, S. Schoenholz, Y. Bahri, R. Novak, J. Sohl-Dickstein, J. Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. NeurIPS, 2019.

### Question.5
Q. “In-context learning” is another finetuning method for repurposing pretrained models for other tasks. Is MR a special case of In-context learning and is your analysis applicable to it? please clarify this in the introduction and if relevant in-context learning too should be discussed in the introduction.

> MR is not a special case of In-context learning (ICL). ICL is usually used in large language models (LLMs) while MR is mostly used in classifiers. Besides, the prompts in ICL are derived by a training-free manner (or provided by the user) while the noise in MR needs to be trained with a specific target dataset. 

## Reviewer 3

### Weakness.1
Q. Model Reprogramming (MR) [1] generally refers to the reuse of models from well-trained tasks to resource-limited tasks. In traditional settings, upstream tasks typically involve a large number of classes and are trained on extensive data, while downstream tasks may focus on specific domains. However, the theory and experiments in this paper, such as MR between cifar10, stl10, or SVHN, resemble transfer learning in small-scale classification tasks across different domains. Can the findings in this paper be generalized to traditional MR settings (e.g., where models are pre-trained on large-scale data and a small part of the pre-trained model's training data may even include classes/domains same as the downstream task)?

>A. Yes, our findings can be generalized to traditional MR settings because we do not assume the relation between the source and target task samples/classes/domains. 

### Weakness.2
Q. Following the previous question, could experiments be added under traditional MR settings [1-4] (e.g., Using CLIP or ResNet pre-trained on ImageNet, ViT as the pre-trained model)?

>A. We have conducted experiments on ResNet and VGG according to the traditional MR settings. The results are shown below. 

    TBD-Resnet (Wei-Chen)

>As for other transformer-based big pre-trained models (source model), we do not have enough computing resource to compute NTK matrix. However, we would like to say that transformers also can compute NTK [1] and our theory is also applicable for transformer-based pre-trained model (source model).  

[1] G. Yang. Tensor programs II: Neural tangent kernel for any architecture. arXiv preprint arXiv:2006.14548, 2020.

### Weakness.3
Q. In Assumption 2, the output label mapping is set as an untrainable linear mapping b. This seems unreasonable. Output label mapping, as shown in [1-2], cannot only be trained but also significantly impacts reprogramming performance. Please provide a reasonable explanation.

>We are the first to formally analyze MR, and so consider the simplest setting. Besides, it is possible to extend our theory to more general cases ($b$ is trainable). When $b$ is trainable, NTK Theory is still applicable [1]. We can still analyze MR through the eigenspectrum of NTK, but unfortunately $\hat\Theta_T(x, x')$ can no longer be expressed as Eq. (16); $\hat\Theta_T(x, x')$ will be in a more complicated form, which deserves to be investigated in the future. Moreover, many MR applications such as BAR (Tsai et al., 2020) and V2S (Yang et al., 2021) satisfy our assumptions ($b$ is chosen in advance). Hence, our theoretical results can be applied immediately in many applications.
  
[1] Yang, G. (2020). Tensor programs ii: Neural tangent kernel for any architecture. arXiv preprint arXiv:2006.14548.

### Weakness.4
Q. This paper's experimental setup of input reprogramming includes FC and VP. Besides adding training parameters around the images [1-2]. There is also a commonly used reprogramming method that adds trainable watermarks to images after resizing the input images [3-4]. Can this method be considered a form of FC described in the paper? Can the proof in the paper support the performance differences between these two commonly used VP methods?

>Yes, the watermark-like input transformation layer can be considered as a special case of FC.
>In Corollary 1, $\hat\Theta_A (X_T, X_T)$ reflects performance for different input transformation layers $a$ (FC and VP). On the other hand, as for our experimental results in section 5, we find that the FC supports our theoretical prediction but VP fails. We believe that the failure of VP is due to the different criterions of "sufficiently large width of target model (input dimension of the source model)". VP may need a larger input dimension to ensure NTK theory holds.

>We have conducted experiments with a large source dataset (ImageNet-10 with image size $112\times 112\times 3$) to verify our conjecture mentioned above.

    TBD ImageNet-10 (Wei-Chen)

>The experimental results further validate our conjecture. 


### Weakness.5
Q. In the conclusion of Section 6, it is claimed that "Our theoretical results can explain the phenomenon that the success of MR usually depends on the success of the source model" [2], but it is also mentioned that this does not hold for VP (padding parameters) reprogramming, but the method in [2] just uses padding VP. Is this contradictory?

>"Our theoretical results can explain that the success of MR usually depends on the success of the source model" means that our theory prediction aligns with the experimental results in [2]. 

>"This does not hold for VP'' means that our theory fails to predict the experimental results conducted in our paper (section 5).

>There is no contradiction because the experimental setting of [2] and our experimental setting are different. [2] considers ImageNet-10 as the source dataset, while we use CIFAR-10. ImageNet-10 has a larger image size compared to CIFAR-10. We believe this is the reason that the theory prediction fails on VP. We think that different target model structures (VP, FC, VP+FC) have different criterions of "sufficiently large width of target model". For VP, the images in CIFAR-10 are not sufficiently large to ensure NTK theory holds, so the experimental results do not align with our theoretical prediction. We have conducted experiments to verify our conjecture. Please check the experiment in the response of Weakness.5.



## Reviewer 4


### Weakness.1
Q. NTK Limitations: While the NTK framework offers valuable insights, its reliance on the infinite-width assumption significantly deviates from real-world neural network architectures. In practice, one has to approximate the empirical NTK. This raises questions about the correctness of theoretical results built on the idealized infinite-width setting.


>We would like to say that the difference between NTK theory and practical finite NN is studied in many papers. [1] had provided comprehensive experiments to record the difference between NTK (idealized infinite-widthe setting) and pratical finite NN. As shown in figure 1 and figure S6 in [1], NN indeed follows NTK theory when the width is sufficiently large.

[1]Lee, J., Xiao, L., Schoenholz, S., Bahri, Y., Novak, R., Sohl-Dickstein, J., & Pennington, J. (2019). Wide neural networks of any depth evolve as linear models under gradient descent. Advances in neural information processing systems, 32.


### Weakness.2

Q. Objective Function Mismatch. The analysis operates on L2 objective (Equation 1) but the context focuses on a classification task. This inconsistency should be clarified.

>A. We will clarify this inconsistency in section 5. 

>Besides, we have done experiments with MSE and CE. Please check the following. 
    
    TBD MSE/CE Wei-Chen
    
>Both MSE and CE (Cross Entropy) share similar tendencies in our experiments. We will put MSE experiments in main content, and put CE experiments into appendix as ablation studies.


### Weakness.3

Q. Content Redundancy. Much of the "Theoretical Framework" (Section 3) overlaps with existing literature, such as [1]. The relationships between prediction, generalization, and eigenvalues are well-established concepts, and devoting substantial space (around two pages) to reiterating them seems unnecessary. Additionally, a clear connection between the derived upper bound (Equation 13) and the subsequent MR analysis in Section 4 remains unclear.

>A. We will refine our content in section 3. Our original intention is to provide a thorough introduction to help readers who are not familiar with NTK realize why we analyze the eigenspectrum of the NTK matrix here. Maybe it is too redundant here. We will move the content of section 3.2 to the appendix. We will also add a citation of [1] in section 3.1 to support why we use eigenvalue spectrum to analyze MR in our paper.

>As for generalization gap (Equation 13), our MR analysis in section 4 is based on empirical risk. We do not consider the generalization gap in our MR analysis because the generalization gap may behave like a constant. In our theorem 2 (Equation 13), we know that
\begin{align}
    \text{Generalization  Gap}
    &\leq 
    \frac{2 \rho B \sqrt{T}}{N_T} 
    \sum_{(x_t, y_t)\in D_T \atop (x_t', y_t')\in D_T} |\Theta_T (x_t, x'_y)|
    + 3 L_{\mathfrak{D}_T} \Gamma_{\mathfrak{D}_T} \sqrt{\frac{\log{\frac{2}{\delta}}}{2N_T}}.
\end{align}
In most cases, $\Gamma_{\mathfrak{D}_T}$, $L_{\mathfrak{D}_T}$ are very large real numbers. It is possible that 
$$
\frac{2 \rho B \sqrt{T}}{N_T} 
    \sum_{(x_t, y_t)\in D_T \atop (x_t', y_t')\in D_T} |\Theta_T (x_t, x'_y)|
    << 
    3 L_{\mathfrak{D}_T} \Gamma_{\mathfrak{D}_T} \sqrt{\frac{\log{\frac{2}{\delta}}}{2N_T}}.
$$
Hence, the influence on the generalization gap caused by NTK could be infeasible. That is why we only use the eigenspectrum of NTK to analyze MR.

### Weakness.4

Q. Unrealistic assumption 2. The authors assume an untrainable output label mapping, which contradicts existing MR practices documented in works like [2, 3]. While the label mapping function might be non-parametric, it can still be updated during training. In fact, these updates have a significant impact on MR performance [2].

>A. We would like to say that this paper is the first attempt to analyze MR, so we consider the simplest setting to study MR's theory.
    
>Besides, it is possible to extend our theory to more general cases ($b$ is trainable). When $b$ is trainable, NTK Theory is still applicable. We still can use analysis MR through the eigenspectrum of NTK. But, $\hat\Theta_T(x, x')$ can not be simply expressed as,
\begin{equation}
b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x))
\nabla_{\theta_A} a(x)
\{b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x'))
\nabla_{\theta_A} a(x')\}^T,
\end{equation}
which increases the difficulty of analysis. 



>Moreover, there are still many MR applications satisfying our theory's assumptions ($b$ is chosen in advance). (need citation here) Thus, we believe our assumption 2 is not an unrealistic assumption.


    TBD need more citations (Ming-Yu)


### Weakness.5

Q. Validity Concerns. Proposition 1 establishes the equivalence between the eigenvalue spectra of $\hat{\Theta}_T(x,x')$ and $\Theta_T(x,x')I_{c_T}$. However, this equivalence might not hold due to the potential output size mismatch, $c_S \neq c_T$, and the trainable nature of label mappings in practice.


>A. We would like to say that Proposition 1 still holds even when $c_S \neq c_T$ or output mapping $b$ is trainable. The only difference when we consider trainable $b$ is that it is more sophisticated to express the value of $\hat{\Theta}_T(x,x')$ . However, $\hat{\Theta}_T(x,x')$ and $\Theta_T(x,x')I_{c_T}$ still possess equivalent eigenvalue spectra no matter what.


### Weakness.6

Q. Unexplored Insights. Assumption 3 introduces a relation between the source and target data distributions ($D_s$ and $D_T$) and input transformation. While this assumption could imply a constraint on the minimum eigenvalues of the source and target kernels and data variability, the authors don't integrate this finding into their method nor validate it empirically.

>We conducted experiments to verify assumption 3,
$$
\lambda_{\text{min}}[\hat\Theta^A_S(x_t, x_t)] \geq \lambda_{\text{min}}[K_S], \forall (x_t, y_t)\in D_T
$$
under simple source model structer $f_S (x) = W x$, where $W\in \mathbb{R}^{c_S \times d_S}$ where $d_S$ is the source model input dimension while $c_S$ is the source model output dimension.

>In this case, the NTK of the source model will be $k(s, s') = \langle \Phi(s), \Phi(s')\rangle = s^T s'$, where $s, s'\in\mathbb{R}^{d_S}$. Thus, we have
$$
\hat\Theta^A_S(x, x') = \nabla_a \Phi(a(x)) [\nabla_a \Phi(a(x'))]^T = I_{d_S}
$$
where $I_{d_S}$ is a $d_S \times d_S$ identity matrix. This implies $\lambda_{\text{min}}[\hat\Theta^A_S(x_t, x_t)]=1$. On the other hand, the exprimental results shows that $\lambda_{\text{min}}[K_S] = 0.003857 < 1$. Thus, assumption 3 holds.

>As for more general source model structure, we could not provide corresponding experiments because $\Phi(x)$ is unknown and even not unique in general. So, it is difficult to verify assumption 3.

### Weakness.7

Q. Limited Empirical Setting. The chosen empirical setting deviates from common practices in MR research. Typically, studies leverage pre-trained models like ResNet on ImageNet or CLIP as the source model. Both the small-scale transfer task and the employed DNN/CNN architecture (with less than 10 hidden layers) may limit the demonstration of the "effectiveness" of the source model due to its restricted capacity.


>A. We have conducted more large-scale experiments (ResNet/VGG trained on ImageNet-10) for your review. Please check the following.

    TBD ResNet & VGG on ImageNet-10(Wei Chen)

>All experimental results further support our theoretical prediction.


### Weakness.8

Q. Missing details and mistakes. The authors do not clearly explain the meaning/terminology involved in some important steps. For example, the concepts of Lipschitz continuity and gradient boundedness within Assumption 1 need explicit definitions. Equation 13 appears to be a typo.

>A. 1. We will provide more description to introduce the concept of some terminologies. Take assumption 1 for example, the gradient boundedness
$$
\|\nabla_{(x, y)} g(x, y)\|_2 \leq L_{\mathfrak{D}_T},~\forall(x, y)\sim\mathfrak{D}_T
$$
indicates the sensitivity of the loss surface $g(x, y)$ on the distribution $\mathfrak{D} _T$. When $L_{\mathfrak{D}_T}$ is large, the loss surface $g(x, y)$ would be steep; while $L_{\mathfrak{D}_T}$ is close to zero, the loss surface $g(x, y)$ will be flat.
 
 
>2.Thanks for your detailed review! We will fix these typos. For example, Equation 13 should be
\begin{align}
    &\mathbb{E}_{(x_t,y_t)\sim\mathfrak{D}_T}[g(x_t, y_t)] - \sum_{(x_i, y_i)\in D_T} \frac{g(x_i, y_i)}{N_T} \nonumber\\
    &\leq 
    \frac{2 \rho B \sqrt{T}}{N_T} 
    \sum_{(x_t, y_t)\in D_T \atop (x_t', y_t')\in D_T} |\Theta_T (x_t, x'_t)|
    + 3 L_{\mathfrak{D}_T} \Gamma_{\mathfrak{D}_T} \sqrt{\frac{\log{\frac{2}{\delta}}}{2N_T}},~\forall g\in\mathcal{G}.
\end{align}
and Equation 16 should be
\begin{equation}
b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x))
\nabla_{\theta_A} a(x)
\{b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x'))
\nabla_{\theta_A} a(x')\}^T.
\end{equation}

