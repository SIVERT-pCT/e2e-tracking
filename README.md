# Exploring End-to-end Differentiable Neural Charged Particle Tracking - A Loss Landscape Perspective
[Tobias Kortus](https://www.scicomp.uni-kl.de/team/kortus/), [Ralf Keidel](https://www.scicomp.uni-kl.de/team/keidel/), [Nicolas R. Gauger](https://www.scicomp.uni-kl.de/team/gauger/), on behalf of *the Bergen pCT* collaboration

The repository contains the [PyTorch](https://pytorch.org/) code for "Exploring End-to-end Differentiable Neural Charged Particle Tracking -- A Loss Landscape Perspective". 

>   Measurement and analysis of high energetic particles for scientific, medical or industrial applications is a complex procedure, requiring the design of sophisticated detector and data processing systems. The development of adaptive and differentiable software pipelines using a combination of conventional and machine learning algorithms is therefore getting ever more important to optimize and operate the system efficiently while maintaining end-to-end (E2E) differentiability. We propose for the application of charged particle tracking an E2E differentiable decision-focused learning scheme using graph neural networks with combinatorial components solving a linear assignment problem for each detector layer. We demonstrate empirically that including differentiable variations of discrete assignment operations allows for efficient network optimization, working better or on par with approaches that lack E2E differentiability. In additional studies, we dive deeper into the optimization process and provide further insights from a loss landscape perspective. We demonstrate that while both methods converge into similar performing, globally well-connected regions, they suffer under substantial predictive instability across initialization and optimization methods, which can have unpredictable consequences on the performance of downstream tasks such as image reconstruction. We also point out a dependency between the interpolation factor of the gradient estimator and the prediction stability of the model, suggesting the choice of sufficiently small values. Given the strong global connectivity of learned solutions and the excellent training performance, we argue that E2E differentiability provides, besides the general availability of gradient information, an important tool for robust particle tracking to mitigate prediction instabilities by favoring solutions that perform well on downstream tasks.
<!--TODO: UPDATE BADGES-->

<a href="https://sivert.info"><img src="https://img.shields.io/website?style=flat-square&logo=appveyor?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=sivert.info&url=https://sivert.info" height=22.5></a>  

</br>

<div class="warning" style='background-color:#d1ecf1; color: #0c5460; border-left: solid #bee5eb 4px; border-radius: 4px; padding:0.7em;'>
<span>
<p style='margin-top:1em; text-align:center'>
<p style='margin-left:1em;'>
<b>NOTE:</b> All source code, models and datasets will be released after paper acceptance.
</p></span>
</div>