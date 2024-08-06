
# [NIPS2024] NTK Rebuttal
## Author Rebuttal

Dear Area Chairs and Senior Area Chairs,

We truly appreciate your dedication and the time spent coordinating our reviews. The reviewers' concerns are mainly about the following points: "whether our NTK analysis can be extended to other large-scale models and large-scale datasets," "the reasonability of the untrainable output mapping assumption," and "confusing between contribution 3 and the conclusion."

Regarding "whether our NTK analysis can be extended to other large-scale models and large-scale datasets," we have conducted additional experiments on ResNet/VGG trained on ImageNet-10 to address the reviewers' concerns. The experimental results further support our theoretical predictions.

Concerning the "reasonability of the untrainable output mapping assumption," we would like to mention that our work is the first attempt to analyze model reprogramming (MR), which is why we chose the simplest setting (untrainable output mapping). Even in this simplest setting, many MR applications satisfy our assumptions. Therefore, we believe the assumption in our paper is reasonable.

For the "confusion between contribution 3 and the conclusion," we want to clarify that the inconsistency between the experimental results of VP and our theoretical predictions could be attributed to the insufficiently large width, which deviates from our assumption. We have conducted additional experiments on models with larger width. The corresponding experimental results support our hypothesis, hence our theory is still valid for VP. We apologize for the inaccurate wording in contribution 3, which is misleading.

In summary, we have addressed all the reviewers' concerns, and additional experiments further support our NTK analysis.

Best regards,
Authors





## Reviewer 1

We appreciate your comments and have a point-to-point response below. Please consider raising the score if you are satisfied with our explanations, and feel free to raise more questions otherwise. 

### W1
>To our knowledge, this paper is the first formal analysis of MR; we start by considering the simplest setting. Our analysis does not apply to all kinds of MR. Nevertheless, many MR applications, such as BAR (Tsai et al., 2020) and V2S (Yang et al., 2021), still satisfy the assumptions of our theory. Moreover, it is possible to extend our theory to more general cases where $b$ is trainable. When $b$ is trainable, NTK theory remains applicable. We can still analyze MR through the eigenspectrum of NTK; however, $\hat{\Theta}_T(x, x')$ can no longer be expressed as in Eq. (16). Instead, $\hat{\Theta}_T(x, x')$ will take on a more complicated form, which merits further investigation in the future.

### W2
>In fact, in lines 316–330, we have provided a potential explanation for why VP fails. Specifically, different input transformation layers (FC, VP, and VP+FC) have different criteria for a "sufficiently large width of the target model." Thus, VP fails because the width of the target model (input dimension of the source model) is insufficiently large. Here, the width of the target model needs to be sufficiently large to ensure that the NTK theory is valid. We conducted additional experiments to validate this argument, in which we changed the source dataset from CIFAR-10 (image size: 32x32x3) to ImageNet-10 (image size: 112x112x3), thereby increasing the input dimension of the source model.

    TBD Experiments on source dataset: ImageNet-10 (Wei-Chen)
![ResNet](https://hackmd.io/_uploads/Sk-Raee5C.png)
![VGG](https://hackmd.io/_uploads/rJX4kbecC.png)

[[Imgur-ResNet]](https://imgur.com/gSjVTRD)
[[Imgur-VGG]](https://imgur.com/gSjVTRD)

>The experimental results show that our theoretical prediction is valid for the target model with a larger width (larger input dimension of the source model), which supports our potential explanation of why VP fails.

### W3
>In our revised paper, we will rewrite and reorganize our findings to improve overall readability. We will also revise the figures and tables to ensure they are informative.



### W4
>We have conducted additional experiments using different losses, different model structures, and different source datasets. The results below are based on a DNN for CIFAR-10 with cross-entropy loss.
>
>The results below are conducted on DNN for CIFAR10 with mean square error. 
>The results below are conducted on CNN (width w=32) for CIFAR10 with mean square error. 
>The results below are conducted on CNN (width w=32) for CIFAR10 with mean square error. 
>The results below are conducted on CNN (width w=512) for CIFAR10 with mean square error. 
>The results below are conducted on CNN (width w=512) for CIFAR10 with mean square error. 
>All the experiments further support our theoretical prediction.

### Q1
>Yes, NTK theory can be employed on other larger vision models like ResNet, VGG, transformers, etc. We conducted experiments using ResNet/VGG as the source model and ImageNet-10 as the source dataset. The results are shown below.

    TBD: ResNet/VGG Eigenvalue, Prompt (Wei-Chen)
![ResNet](https://hackmd.io/_uploads/Sk-Raee5C.png)
![VGG](https://hackmd.io/_uploads/rJX4kbecC.png)

[[Imgur-ResNet]](https://imgur.com/gSjVTRD)
[[Imgur-VGG]](https://imgur.com/gSjVTRD)

    
>The results show that the minimum eigenvalue of the source model increases as we make the source model deeper. Conversely, the target loss decreases as we make the source model deeper. Hence, the above results validate our theoretical prediction.

### Q2
>NTK theory is applicable to transformer-based source models. (Tensor Programs II: Neural Tangent Kernel for Any Architecture, arXiv, 2020) has proved that any modern model satisfying the "Simple Gradient Independence Assumption (GIA) Check" has an NTK. As for CLIP, NTK theory is not immediately applicable due to the type of contrastive loss (cosine similarity). However, developing an $\ell_2$-norm-like contrastive loss to make NTK theory applicable to CLIP would be an interesting research direction.

## Reviewer 2

We appreciate your comments and have a point-to-point response below. Please consider raising the score if you are satisfied with our explanations, and feel free to raise more questions otherwise. 

### W1 
>The width of the neural network is defined as the number of neurons in each hidden layer.

### W2
>In our experiment, we first randomly sampled 10 images per class from the source dataset. Then, we computed the NTK matrix and corresponding minimum eigenvalue using this [library](https://github.com/google/neural-tangents). We repeated this procedure 3 times and used the average of these 3 minimum eigenvalues of the NTK matrix as our experimental results, as shown in Table 1.

### W3, 4, 5, 6
>In the revised paper, we will follow your suggestions to revise Figures 1, 2, and 3, and ensure all abbreviations are well defined. We will also fix typos and grammatical errors.

### W7, Q2
>In our contribution 3, we want to convey the following things.
>Corollary 1 claims that the loss of the target model evaluated on the target distribution will decrease as we make the source model deeper. However, the results in Table 4 and Table 6 suggest that this theoretical prediction holds when we choose FC as the input transformation layer but fails when we choose VP as the input transformation layer. We hypothesize that this failure is due to the width of the target model (input dimension of the source model) not being large enough. In NTK theory, the model width needs to be sufficiently large so that the target model's behavior can be characterized by the NTK. In our experiments, the model width is $32\times 32\times 3$, which is sufficiently large for the target model with FC as the input transformation layer. Thus, the results for FC support our theoretical prediction. On the contrary, $32\times 32\times 3$ is not large enough for the target model with VP as the input transformation layer, so our theoretical prediction fails on the experimental results for VP.


### W8
>There is a typo in Eq. (16); it should be:
\begin{equation}
b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x))
\nabla_{\theta_A} a(x)
\{b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x'))
\nabla_{\theta_A} a(x')\}^T.
\end{equation}
>The derivation of Eq. (16) will be included in the revised paper. 

### W9
>Yes, they are meaningful. We will name them while introducing them in the revised paper.

### W10
>The reason we did not include more runs is the insufficient computing resources. However, we have conducted additional experiments, and the results are shown below.

    TBD CNN/DNN CE/MSE MEV (Wei-Chen)
    
![CNN](https://hackmd.io/_uploads/BkmqV-xc0.png)
![DNN](https://hackmd.io/_uploads/B1n5Ebl5R.png)


[[Imgur-CNN]](https://imgur.com/oF8anzv)
[[Imgur-DNN]](https://imgur.com/Eu9KO7a)
    
>From the results, VP+FC appears to require less width in the target model to ensure NTK theory holds compared to VP but more width compared to FC. This explains why FC on both DNN and CNN is consistent with our analysis, why VP on both DNN and CNN is inconsistent with our analysis, and why VP+FC on CNN is consistent with our analysis, whereas VP+FC on DNN is not.

### W11
>We have conducted experiments for different widths (input dimensions of the source model). We trained the source models on ImageNet10 with input sizes of 112, 56, and 28, and trained the target models with VP as the input transformation layer. The results are shown below.

    TBD ImageNet-10 112, 56, 28 (Wei-Chen)

![SIZE-1](https://hackmd.io/_uploads/HJWhLbgcC.png)
![SIZE-2](https://hackmd.io/_uploads/rkr2UbxcA.png)

[[Imgur SizeExp1]](https://imgur.com/pwvvz8E)
[[Imgur SizeExp2]](https://imgur.com/yt8jcxS)

### Q1
>Our assumption 2 requires three things: (1) Source model $f_S$ has large width and hence satisfies NTK theory. (2) Output mapping $b$ is a linear matrix. (3) Output mapping $b$ is untrainable.

>Our assumption is not limiting. For (1), we notice that the source model should be a big model in real-world MR application scenarios. So, the source model $f_S$ naturally has a large width. Besides, many modern NN structures like ResNet, transformers, etc., have NTK properties. (Tensor Programs II: Neural Tangent Kernel for Any Architecture, arXiv'22) proved that any modern NN satisfying the "Simple Gradient Independence Assumption (GIA) check" has NTK. Hence, assuming the source model $f_S$ satisfies NTK theory is reasonable. For (2) and (3), we require that $b$ is an untrainable linear matrix because such a setting is more feasible for NTK analysis, and many MR applications such as BAR (Tsai et al., 2020) and V2S (Yang et al., 2021) satisfy this requirement ($b$ is an untrainable linear matrix).

>Indeed, “fix output transformation layer $a$ and do the NTK analysis for output mapping $b$” is also feasible. However there is relatively less research under this setting. That is why we set "fix $b$ and do NTK analysis for $a$"" rather than "fix $b$ and do NTK analysis for $a$".

### Q3
>It indicates that the unit of the minimum eigenvalue is 1e-2. 


### Q4
>The width of the neural network is defined as the number of neurons of the hidden layers. For $b$, since $b$ is a single linear matrix in our setting, $b$ does not have width. For $a$, when $a$ is VP or FC, $a$ only has one input layer and one output layer, so $a$ also does not have width. However, when $a$ is VP+FC, $a$ has one hidden layer, which has $32\times 32\times 3$ (input dimension of the source model) neurons. So, the width of $a$ is $32\times 32\times 3$.

>For the source model $f_S$, the width of $f_S$ depends on the model structure of $f_S$. The width will be $2048$ when $f_S$ is DNN. Especially, when $f_S$ is CNN, the width is considered as $channel \times filter$. In our experiments, CNN is considered to have $512$ channels and $(3, 3)$ filter. So, the width will be $512\times 3^2$.

>We follow (Lee et al., 2019) to set the width of the source model. Lee et al. (2019) suggested that the training dynamics of CNN and DNN can be well approximated by NTK theory. So, our source model $f_S$ satisfies has a large width.

### Q5
> MR is not a special case of In-context learning (ICL). ICL is usually used in large language models (LLMs) while MR is mostly used in classifiers. Besides, the prompts in ICL are derived by a training-free manner (or provided by the user) while the noise in MR needs to be trained with a specific target dataset. 

## Reviewer 3

We appreciate your comments and have a point-to-point response below. Please consider raising the score if you are satisfied with our explanations, and feel free to raise more questions otherwise. 

### W1
>Yes, our findings can be generalized to traditional MR settings because we do not assume the relation between the source and target task samples/classes/domains. 

### W2
>We have conducted experiments on ResNet and VGG according to the traditional MR settings. The results are shown below. 

    TBD-Resnet/ VGG (Wei-Chen)
    
![ResNet](https://hackmd.io/_uploads/Sk-Raee5C.png)
![VGG](https://hackmd.io/_uploads/rJX4kbecC.png)

[[Imgur-ResNet]](https://imgur.com/gSjVTRD)
[[Imgur-VGG]](https://imgur.com/gSjVTRD)

>As for other transformer-based big pre-trained models (source model), we do not have enough computing resource to compute NTK matrix. However, we would like to say that transformers also can compute NTK (Tensor programs II: Neural tangent kernel for any architecture, arxiv'22) and our theory is also applicable for transformer-based pre-trained model (source model).  

### W3
>We are the first to formally analyze MR, and so consider the simplest setting. Besides, it is possible to extend our theory to more general cases ($b$ is trainable). When $b$ is trainable, NTK Theory is still applicable (Tensor programs II: Neural tangent kernel for any architecture, arxiv'22). We can still analyze MR through the eigenspectrum of NTK, but unfortunately $\hat\Theta_T(x, x')$ can no longer be expressed as Eq. (16); $\hat\Theta_T(x, x')$ will be in a more complicated form, which deserves to be investigated in the future. Moreover, many MR applications such as BAR (Tsai et al., 2020) and V2S (Yang et al., 2021) satisfy our assumptions ($b$ is chosen in advance). Hence, our theoretical results can be applied immediately in many applications.

### W4
>Yes, the watermark-like input transformation layer can be considered as a special case of FC.
>In Corollary 1, $\hat\Theta_A (X_T, X_T)$ reflects performance for different input transformation layers $a$ (FC and VP). On the other hand, as for our experimental results in section 5, we find that the FC supports our theoretical prediction but VP fails. We believe that the failure of VP is due to the different criterions of "sufficiently large width of target model (input dimension of the source model)". VP may need a larger input dimension to ensure NTK theory holds.

>We have conducted experiments with a large source dataset (ImageNet-10 with image size $112\times 112\times 3$) to verify our conjecture mentioned above.

    TBD ImageNet-10 (Wei-Chen)

![ResNet](https://hackmd.io/_uploads/Sk-Raee5C.png)
![VGG](https://hackmd.io/_uploads/rJX4kbecC.png)

[[Imgur-ResNet]](https://imgur.com/gSjVTRD)
[[Imgur-VGG]](https://imgur.com/gSjVTRD)
>The experimental results further validate our conjecture. 


### W5
>"Our theoretical results can explain that the success of MR usually depends on the success of the source model" means that our theory prediction aligns with the experimental results in [2]. 

>"This does not hold for VP'' means that our theory fails to predict the experimental results conducted in our paper (section 5).

>There is no contradiction because the experimental setting of [2] and our experimental setting are different. [2] considers ImageNet-10 as the source dataset, while we use CIFAR-10. ImageNet-10 has a larger image size compared to CIFAR-10. We believe this is the reason that the theory prediction fails on VP. We think that different target model structures (VP, FC, VP+FC) have different criterions of "sufficiently large width of target model". For VP, the images in CIFAR-10 are not sufficiently large to ensure NTK theory holds, so the experimental results do not align with our theoretical prediction. We have conducted experiments to verify our conjecture. Please check the experiment in the response of Weakness.5.

## Reviewer 4

We appreciate your comments and have a point-to-point response below. Please consider raising the score if you are satisfied with our explanations, and feel free to raise more questions otherwise. 

### W1
>The difference between NTK theory and practical finite NN has been studied extensively. Lee et al. (2019) had provided comprehensive experiments to record the difference between NTK (idealized infinite-widthe setting) and pratical finite NN. As shown in Figure 1 and Figure S6 in (Lee et al., 2019), NN indeed follows NTK theory when the width is sufficiently large.

### W2
>We have done extra experiments with MSE and CE.
    
    TBD MSE Wei-Chen
![CNN](https://hackmd.io/_uploads/BkmqV-xc0.png)
![DNN](https://hackmd.io/_uploads/B1n5Ebl5R.png)


[[Imgur-CNN]](https://imgur.com/oF8anzv)
[[Imgur-DNN]](https://imgur.com/Eu9KO7a)
>Both MSE and CE share similar tendencies in our experiments.

### W3
>We will move part of the content in Section 3.2 to the Appendix. As for generalization gap (Eq. (13)), our MR analysis in Section 4 is based on empirical risk. We do not consider the generalization gap in our MR analysis because the generalization gap may behave like a constant. In Theorem 2, we know that
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

### W4
>To our knowledge, this paper is the first formal analysis of MR; we start by considering the simplest setting. Our analysis does not apply to all kinds of MR. Nevertheless, many MR applications such as BAR (Tsai et al., 2020) and V2S (Yang et al., 2021) still satisfy our theory's assumptions. Besides, it is possible to extend our theory to more general cases ($b$ is trainable). When $b$ is trainable, NTK Theory is still applicable. We can still analyze MR through the eigenspectrum of NTK, but unfortunately $\hat\Theta_T(x, x')$ can no longer be expressed as Eq. (16); $\hat\Theta_T(x, x')$ will be in a more complicated form, which deserves to be investigated in the future. 


### W5
>Proposition 1 still holds even when $c_S \neq c_T$ or output mapping $b$ is trainable. The only difference when we consider trainable $b$ is that it is more sophisticated to express the value of $\hat{\Theta}_T(x,x')$. However, $\hat{\Theta}_T(x,x')$ and $\Theta_T(x,x')I_{c_T}$ still possess equivalent eigenvalue spectra in any case.


### W6
>We conduct experiments to empirically validate Assumption 3,
$$
\lambda_{\text{min}}[\hat\Theta^A_S(x_t, x_t)] \geq \lambda_{\text{min}}[K_S], \forall (x_t, y_t)\in D_T
$$
under simple source model structure $f_S (x) = W x$, where $W\in \mathbb{R}^{c_S \times d_S}$ and $d_S$ is the source model input dimension while $c_S$ is the source model output dimension.

>In this case, the NTK of the source model will be $k(s, s') = \langle \Phi(s), \Phi(s')\rangle = s^T s'$, where $s, s'\in\mathbb{R}^{d_S}$. Thus, we have
$$
\hat\Theta^A_S(x, x') = \nabla_a \Phi(a(x)) [\nabla_a \Phi(a(x'))]^T = I_{d_S},
$$
where $I_{d_S}$ is a $d_S \times d_S$ identity matrix. This implies $\lambda_{\text{min}}[\hat\Theta^A_S(x_t, x_t)]=1$. On the other hand, the experimental results shows that $\lambda_{\text{min}}[K_S] = 0.003857 < 1$. Thus, Assumption 3 holds.

>As for more general source model structure, we could not provide corresponding experiments because $\Phi(x)$ is unknown and even not unique in general.

### W7
>We have conducted extra large-scale experiments (ResNet/VGG trained on ImageNet-10). 

    TBD ResNet & VGG on ImageNet-10(Wei Chen)
![ResNet](https://hackmd.io/_uploads/Sk-Raee5C.png)
![VGG](https://hackmd.io/_uploads/rJX4kbecC.png)

[[Imgur-ResNet]](https://imgur.com/gSjVTRD)
[[Imgur-VGG]](https://imgur.com/gSjVTRD)
>All experimental results further support our theoretical prediction.


### W8
>1. We will provide more descriptions to introduce the concept of some terminologies. Take assumption 1 for example, the gradient boundedness
$$
\|\nabla_{(x, y)} g(x, y)\|_2 \leq L_{\mathfrak{D}_T},~\forall(x, y)\sim\mathfrak{D}_T
$$
indicates the sensitivity of the loss surface $g(x, y)$ on the distribution $\mathfrak{D} _T$. When $L_{\mathfrak{D}_T}$ is large, the loss surface $g(x, y)$ would be steep; while $L_{\mathfrak{D}_T}$ is close to zero, the loss surface $g(x, y)$ will be flat.
 
 
>2. Thanks for your detailed review! We will fix these typos. For example, Eq. (13) should be
\begin{align}
    &\mathbb{E}_{(x_t,y_t)\sim\mathfrak{D}_T}[g(x_t, y_t)] - \sum_{(x_i, y_i)\in D_T} \frac{g(x_i, y_i)}{N_T} \nonumber\\
    &\leq 
    \frac{2 \rho B \sqrt{T}}{N_T} 
    \sum_{(x_t, y_t)\in D_T \atop (x_t', y_t')\in D_T} |\Theta_T (x_t, x'_t)|
    + 3 L_{\mathfrak{D}_T} \Gamma_{\mathfrak{D}_T} \sqrt{\frac{\log{\frac{2}{\delta}}}{2N_T}},~\forall g\in\mathcal{G}.
\end{align}
and Eq. (16) should be
\begin{equation}
b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x))
\nabla_{\theta_A} a(x)
\{b Y_S^T [K_S + \sigma_S I]^{-1} \Phi(X_S)^T \nabla_{a} \Phi(a(x'))
\nabla_{\theta_A} a(x')\}^T.
\end{equation}

