[//]: # (# PMBP)

[//]: # ([Wenbo Shao]&#40;https://scholar.google.com/citations?user=nJgFCn0AAAAJ&hl=zh-CN&oi=ao&#41;)
[//]: # (Wenbo Shao, Boqi Li, Wenhao Yu, Jiahui Xu, Hong Wang)

[//]: # ()
[//]: # (- School of Vehicle and Mobility, Tsinghua University )

[//]: # (- Department of Civil and Environmental Engineering, University of Michigan )

[//]: # (- School of Mechanical Engineering, Beijing Institute of Technology)

<p align="center">
<img src="src/paper_1.png" width="500" alt="curve">
</p>

# 1. Abstract
Accurate trajectory prediction is vital for various applications, including autonomous vehicles. However, the complexity and limited transparency of many prediction algorithms often result in black-box models, making it challenging to understand their limitations and anticipate potential failures. This further raises potential risks for systems based on these prediction models. This study introduces the performance monitor for black-box trajectory prediction model (PMBP) to address this challenge. The PMBP estimates the performance of black-box trajectory prediction models online, enabling informed decision-making. The study explores various methods' applicability to the PMBP, including anomaly detection, machine learning, deep learning, and ensemble, with specific monitors designed for each method to provide online output representing prediction performance. Comprehensive experiments validate the PMBP's effectiveness, comparing different monitoring methods.
Results show that the PMBP effectively achieves promising monitoring performance, particularly excelling in deep learning-based monitoring. It achieves improvement scores of 0.81 and 0.79 for average prediction error and final prediction error monitoring, respectively, outperforming previous white-box and gray-box methods. Furthermore, the PMBP's applicability is validated on different datasets and prediction models, while ablation studies confirm the effectiveness of the proposed mechanism. Hybrid prediction and autonomous driving planning experiments further show the PMBP's value from an application perspective. 

# 2. Method Overview

Offline training (gray background) and online monitoring (green background) phases for PMBP

<p align="center">
<img src="src/paper_2.png" width="500">
</p>

**The comprehensive design and evaluation framework of PMBP.** Various methods belonging to four type of approaches are designed and compared.
<p align="center">
<img src="src/S1.png" width="900">
</p>

## 2.1. Anomaly Detection-based Monitoring
[//]: # (| <img src="./src/paper_4.png" width="300px"> | <img src="./src/AD_table.png" width="300px"> |)
[//]: # (![paper_4.png]&#40;src%2Fpaper_4.png&#41;)
![AD.png](src%2FAD.png)

## 2.2. Machine Learning-based Monitoring

<p align="center">
<img src="src/ML.png" width="400">
</p>

## 2.3. Deep Learning-based Monitoring

![paper_5.png](src%2Fpaper_5.png)


## 2.4. Ensemble-based Monitoring
<p align="center">
<img src="src/paper_6.png" width="500">
</p>

# 3. Evaluation

## 3.1. Calculation of AUCOC & IS
<p align="center">
 <img src="src/img_2.png" width = "500" alt="img_2" />
</p>

<p align="center">
 <img src="src/img_1.png" width = "400" alt="img_2" />
</p>


## 3.2. Comparison of PMBP with other methods
![img.png](src/comparison.png)

Cutoff Curve for PMBP (MLP-based model with error estimation)

<p align="center">
<img src="src/cutoff_curve.png" width="500" alt="curve">
</p>

Temporal Analysis

<p align="center">
<img src="src/Temporal Analysis.png" width="500" alt="curve">
</p>


## 3.3. Visualization

[//]: # (| <img src="./src/vis_pic_01.png" width="300px"> | <img src="./src/vis_pic_02.png" width="300px"> | <img src="./src/vis_pic_03.png" width="300px"> |)
<img src="./src/vis_pic_01.png">

More visualization results:
![img.png](src/more_vis_1.png)

![img_1.png](src/more_vis_2.png)


## 3.4. Hybrid Prediction
<p align="center">
<img src="src/Hybrid Prediction .png" width="800" alt="curve">
</p>

## 3.5. Autonoumous Driving Planning

| <video muted controls width=380> <source src="./src/Planning_05.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/Planning_06.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./src/Planning_02.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/Planning_03.mp4"  type="video/mp4"> </video> |




## 3.6. Citation
```
@article{shao2023likely,
  title={When Is It Likely to Fail? Performance Monitor for Black-Box Trajectory Prediction Model},
  author={Shao, Wenbo and Li, Boqi and Yu, Wenhao and Xu, Jiahui and Wang, Hong},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2023},
  publisher={IEEE}
}
```

## 3.7. Contact
If you have any questions, feel free to contact us (swb19@mails.tsinghua.edu.cn).