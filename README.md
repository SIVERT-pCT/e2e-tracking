**Status:** Archive (code is provided as-is, no updates expected)

# Exploring End-to-end Differentiable Neural Charged Particle Tracking - A Loss Landscape Perspective
[Tobias Kortus](https://www.scicomp.uni-kl.de/team/kortus/), [Ralf Keidel](https://www.scicomp.uni-kl.de/team/keidel/), [Nicolas R. Gauger](https://www.scicomp.uni-kl.de/team/gauger/), on behalf of *the Bergen pCT* collaboration

The repository contains the [PyTorch](https://pytorch.org/) code for "Exploring End-to-end Differentiable Neural Charged Particle Tracking -- A Loss Landscape Perspective". The basic code structure and multiple code modules are taken from [1]. The source code for calculating hessian eigenvalues and eigenvectors is adapted from the PyHESSIAN library [3, 4]. The baseline results for the track follower are generated using the Digital Tracking Calorimeter Toolkit [2]

> Measurement and analysis of high energetic particles for scientific, medical or industrial applications is a complex procedure, requiring the design of sophisticated detector and data processing systems. The development of adaptive and differentiable software pipelines using a combination of traditional and machine learning algorithms is therefore getting increasingly more important to optimize and operate the system efficiently while maintaining end-to-end~(E2E) differentiability. We propose for the application of charged particle tracking an E2E differentiable decision-focused learning scheme using graph neural networks with combinatorial components solving a linear assignment problem for each detector layer. We demonstrate empirically that including differentiable variations of discrete assignment operations allows for efficient network optimization, working better or on par with approaches that lack E2E differentiability. In additional studies, we dive deeper into the optimization process and provide further insights into two-step and end-to-end schemes from a loss landscape perspective. We demonstrate that while two-step and end-to-end methods converges into similar performing, globally well-connected regions, both suffer under substantial predictive instability across random initialization and optimization methods, which can have unpredictable consequences on the performance of downstream tasks such as image reconstruction. We also point out a dependency between the interpolation factor of the blackbox gradient estimator used and the prediction stability of the model, suggesting the choice of sufficiently small values. Given the strong global connectivity of learned solutions and the excellent training performance, we argue that E2E differentiability provides, besides the general availability of gradient information, an important tool for robust charged particle tracking to incorporate additional functional requirements that allow to mitigate prediction instabilities by favoring solutions that perform well on downstream tasks.

<!--TODO: UPDATE BADGES-->
<a href="https://sivert.info"><img src="https://img.shields.io/website?style=flat-square&logo=appveyor?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=sivert.info&url=https://sivert.info" height=22.5></a>


![img](figures/tracks.png)

## Installation

```powershell
pip install -r requirements.txt
```

## Data and Models

For reproducibility we provide the user with the exact Monte-Carlo simulations used throughout the paper. All simulations can be downloaded from [Zenodo](https://zenodo.org/record/7426388) and extracted using the following command (the downloaded archive should be copied into the `data/` directory):

```powershell
tar -xf data.tar.gz --strip-components 1
```
> Note: Please note that the following instructions are provided for Linux operating systems. Some commands may vary for different operating systems.

Similarly we provide the pretrained weights, checkpoints and precomputed loss landscapes of all evaluated network variants used throughout the paper. The data can be extracted in a similar fashion using:

```powershell
tar -xf models.tar.gz --strip-components 1
```

## Training the models

All experiments with the corresponding hyperparameters parameters, performed in the paper, are documented as `.json` files. An experiment, with the provided models, can be re-run using the following commands:

```powershell
python train.py  -e experiments/****.json  -d ****
```

- `-e`: Experiment definition file. Either one of the predefined in `experiments/default`/ `experiments/ablation` or a custom definition following the json structure of the existing experiments.
- `-d`: Computation device that should be used by pytorch (cpu, cuda:0-N)

> Note: If you wish to retrain a model, the respective files should be deleted from the corresponding model directory.


## Generating Similairites and Mode Connectivity Results

```powershell
python slurm_iterate_combinations.py -t mode #Mode connectivity

python slurm_iterate_combinations.py -t cka #Fun. & repr. similarities
```

## Running Reporting Scripts

All code used for analyzing the trained models and loss landscape analysis are archived in the `reporting/` directory. A list of all scripts as well as a summary of the data generated is provided in the following script cells.

```powershell
python reporting/generate_filter_plots.py #Figure 8
```

```powershell
python reporting/generate_loss_landscape_plots.py #Figure 5 & 6
```

```powershell
python reporting/generate_mode_connectivity_results.py # Table 2 & 3
```

```powershell
python reporting/generate_performance_results.py # Results for Table 1 & Figure 4
```

```powershell
python reporting/generate_similarity_plots.py #Figure 7
```


## Referencing this Work

If you find this repository useful for your research, please cite the following work.

```
TODO
```

## References

[1] **Towards Neural Charged Particle Tracking in Digital Tracking Calorimeters with Reinforcement Learning**, Tobias Kortus, Ralf Keidel, Nicolas R. Gauger, [Source code]: https://github.com/SIVERT-pCT/rl-tracking

[2] **Digital Tracking Calorimeter Toolkit**, Helge E.S. Pettersen, [Sorce code]: https://github.com/HelgeEgil/DigitalTrackingCalorimeterToolkit

[3] **PyHessian**, Z. Yao, A. Gholami, K Keutzer, M. Mahoney, [Source Code] https://github.com/amirgholami/

[4] Z. Yao, A. Gholami, K Keutzer, M. Mahoney. PyHessian: Neural Networks Through the Lens of the Hessian, Spotlight at ICML workshop on Beyond First-Order Optimization Methods in Machine Learning, 2020