# When Is It Likely to Fail? Performance Monitor for Black-Box Trajectory Prediction Model
[Zhiyu Huang](https://scholar.google.com/citations?user=nJgFCn0AAAAJ&hl=zh-CN&oi=ao)

Tsinghua University

## Abstract
Accurate trajectory prediction is vital for various applications, including autonomous vehicles. However, the complexity and limited transparency of many prediction algorithms often result in black-box models, making it challenging to understand their limitations and anticipate potential failures. This further raises potential risks for systems based on these prediction models. This study introduces the performance monitor for black-box trajectory prediction model (PMBP) to address this challenge. The PMBP estimates the performance of black-box trajectory prediction models online, enabling informed decision-making. The study explores various methods' applicability to the PMBP, including anomaly detection, machine learning, deep learning, and ensemble, with specific monitors designed for each method to provide online output representing prediction performance. Comprehensive experiments validate the PMBP's effectiveness, comparing different monitoring methods.
Results show that the PMBP effectively achieves promising monitoring performance, particularly excelling in deep learning-based monitoring. It achieves improvement scores of 0.81 and 0.79 for average prediction error and final prediction error monitoring, respectively, outperforming previous white-box and gray-box methods. Furthermore, the PMBP's applicability is validated on different datasets and prediction models, while ablation studies confirm the effectiveness of the proposed mechanism. Hybrid prediction and autonomous driving planning experiments further show the PMBP's value from an application perspective. 

## Method Overview

The proposed framework draws inspiration from hierarchical game-theoretic modeling of agent interactions. The framework encodes the historical states of agents and maps as background information via a Transformer-based encoder. A level-0 agent's future trajectories are decoded independently, based on the initial modality query. At level-k, an agent responds to all other agents at level-(k-1). The level-0 decoder uses modality embedding and agent history encodings as query inputs to independently decode the future trajectories and scores for level-0 agents. The level-k decoder incorporates a self-attention module to model the future interactions at level-(k-1) and appends this information to the scene context encoding.

<img src="./src/S1.png">

## Citation
```
@article{shao2023likely,
  title={When Is It Likely to Fail? Performance Monitor for Black-Box Trajectory Prediction Model},
  author={Shao, Wenbo and Li, Boqi and Yu, Wenhao and Xu, Jiahui and Wang, Hong},
  journal={Authorea Preprints},
  year={2023},
  publisher={Authorea}
}
```

## Contact
If you have any questions, feel free to contact us (swb19@mails.tsinghua.edu.cn).
