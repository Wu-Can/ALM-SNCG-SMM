# Support matrix machine: exploring sample sparsity, low rank, and adaptive sieving in high-performance computing

This archive is distributed under the [General Public License v2.0](LICENSE).

This repository contains a snapshot of the software used for the research presented in: [Support matrix machine: exploring sample sparsity, low rank, and adaptive sieving in high-performance computing](https://arxiv.org/abs/2412.08023) by Can Wu, Donghui Li and Defeng Sun.

## 1 Description

The goal of this MATLAB software is to solve the [support matrix machine](https://proceedings.mlr.press/v37/luo15.html) (SMM) model using a  semismooth Newton-CG based augmented Lagrangian method. Performance comparisons include:
- Inexact semi-proximal ADMM (isPADMM)
- Symmetric Gauss-Seidel based inexact semi-proximal ADMM (sGS-isPADMM)
- Publically available [Fast ADMM with restart](http://bcmi.sjtu.edu.cn/~luoluo/code/smm.zip)

####  1.1 Optimization Problems and Solvers

- **1) Solvers for SMM model with a fixed value of C**
  - **ALM-SNCG**: Semismooth Newton-CG based augmented Lagrangian method
  - **isPADMM**: Inexact semi-proximal alternating direction method of multipliers 
  - **sGS-isPADMM**: Symmetric Gauss-Seidel based isPADMM
  - **F-ADMM**: Fast alternating direction method of multipliers with restart rule
- **2) Solvers for SMM models with a sequence of C**
  - **AS+ALM**: Adaptive sieving strategy combined with **ALM-SNCG**
  - **Warm+ALM**: Warm-strated **ALM-SNCG**

#### 1.2 Data Sources

- **1) Four real-world datasets**  
  - [EEG Alcoholism](http://kdd.ics.uci.edu/databases/eeg/eeg.html): Classification of alcoholic vs. non-alcoholic subjects via EEG signals
  - INRIA Person: Human detection in images. You can download the dataset (e.g., using Google Chrome) from:
    `ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar`
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): Binary classification (dog vs. truck)
  - [MNIST](https://github.com/cvdfoundation/mnist.git): Handwritten digit recognition (0 vs. non-0)

- **2) Synthetic data**  
Generated randomly following the process in [Luo et al. (2015)](https://proceedings.mlr.press/v37/luo15.html).

#### 1.3 Repository Structure

- **1) Core solvers**
  - `mainfun`: Main functions (ALM-SNCG, isPADMM, sGS-isPADMM, and AS+ALM)
  - `subfun`: Subfunctions called by the above main functions
  - `FADMM`: Fast ADMM implementation with its subfunctions
- **2) Numerical experiments**  
The scripts for replicating all numerical results in the paper 
  - `Test0_figures_1_2`: Sample sparsity and low-rank characterization
  - `Test1_fixed_C`: Fixed-parameter solver comparisons: ALM-SNCG vs. isPADMM vs. sGS-isPADMM vs. F-ADMM
  - `Test2_path_Cvec`: Parameter sequence solver comparisons: AS+ALM vs. Warm+ALM
- **3) Data resources** 
  - `Data`: All synthetic and real datasets with precomputed high-precision objective values
  
## 2. Computational Environment and Usage
  
All reported results were obtained using MATLAB R2022b on a desktop computer (8-core, Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz, 64G RAM). Replicate experiments as follows:

***Step 1***. Extract the repository and launch MATLAB from its root directory  
***Step 2***. In the MATLAB command window, type:

```
    >> startup 
```

***Step 3***. Generate synthetic datasets: Run 
*Test_generate_random_data.m* in the folder `\Data\Random_data`

***Step 4***. Create benchmark solutions: Run *Generate_ALMSNCG_relkkt_1e_8.m* in the folder `\Data`.

***Step 5***. Execute experiment scripts according to the following table:

<table style="border-top: 1px solid white; border-bottom: 1px solid white">
  <tr>
    <th>Results</th>
    <th>Scripts</th>
    <th>Folders</th>
  </tr>
  <tr>
    <td>Table 3</td>
    <td><i>Generate_Table3_real_relobj</i></td>
    <td><code>\Test1_fixed_C</code></td>
  </tr>
    <tr>
    <td>Table 4</td>
    <td><i>Generate_Table4_random_relobj</i></td>
    <td><code>\Test1_fixed_C</code></td>
  </tr>
    <tr>
    <td>Table 5</td>
    <td><i>Generate_Table5_real_relobj</i></td>
    <td><code>\Test1_fixed_C</code></td>
  </tr>
    <tr>
    <td>Table 6</td>
    <td><i>Generate_Table6_random_relobj</i></td>
    <td><code>\Test2_path_Cvec</code></td>
  </tr>
  <tr>
    <td>Table 7</td>
    <td><i>Generate_Table7_real_relobj</i></td>
    <td><code>\Test2_path_Cvec</code></td>
  </tr>
  <tr>
    <td>Figure 1</td>
    <td><i>Figure1_SM_ASM</i></td>
    <td><code>\Test0_figures_1_2</code></td>
  </tr>
  <tr>
    <td>Figure 2</td>
    <td><i>Figure2_low_rank_W</i></td>
    <td><code>\Test0_figures_1_2</code></td>
  </tr> 
    <tr>
    <td>Figure 3</td>
    <td><i>Test_figure_AS_Percentage_random</i></td>
    <td><code>\Test2_path_Cvec\Result_solution_path_figure_random</code></td>
  </tr>
    <tr>
    <td>Figure 4</td>
    <td><i>Test_figure_AS_Percentage_real</i></td>
    <td><code>\Test2_path_Cvec\Result_solution_path_figure_real</code></td>
  </tr>
  </table>

## 3. Results Replication

Execute the corresponding scripts in the following folders to reproduce all paper results:
- `\Test0_figures_1_2`: Sample sparsity and low-rank characterization
- `\Test1_fixed_C`: Fixed-parameter solver comparisons
- `\Test2_path_Cvec`: Parameter sequence solver evaluations

  
## 4. Remark

As mentioned in our paper, objective values computed by **ALM-SNCG** (tolerance 1e-8) serve as accuracy benchmarks when comparing algorithms with different termination criteria. Precomputed high-accuracy objective values are available in `\Data\Real_data`, enabling omission of Step 4.














