# PRLight-and-PLight

# 自适应交通信号控制
本项目使用基于模型的多智能体强化学习解决自适应交通信号控制中的快速自适应问题。以下内容介绍了项目的环境要求以及使用方式。

本项目为论文 [Enhancing traffic signal control through model-based reinforcement learning and policy reuse](https://www.sciencedirect.com/science/article/pii/S0957417425033706) 的官方开源实现，如有引用请使用以下 BibTeX：
```bibtex
@article{LI2026129755,
title = {Enhancing traffic signal control through model-based reinforcement learning and policy reuse},
journal = {Expert Systems with Applications},
volume = {298},
pages = {129755},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.129755},
author = {Yihong Li and Chengwei Zhang and Furui Zhan and Wanting Liu and Kailing Zhou and Longji Zheng},
keywords = {Traffic signal control, Learning efficiency, Transfer learning, Model-based reinforcement learning},
abstract = {Multi-agent reinforcement learning (MARL) has shown significant potential in traffic signal control (TSC). However, current MARL-based methods often suffer from insufficient generalization due to the fixed traffic patterns and conditions of the road network used during training. This limitation results in poor adaptability to new traffic scenarios, leading to high retraining costs and complex deployment. To address this challenge, we propose two algorithms: PLight and PRLight. PLight employs a model-based reinforcement learning approach, pretraining control policies, and environment models using predefined source-domain traffic scenarios. The environmental model predicts state transitions, facilitating the comparison of environmental characteristics. PRLight further enhances adaptability by adaptively selecting pre-trained PLight agents based on the similarity between the source and target domains to accelerate the learning process in the target domain. We evaluated the algorithms through two transfer settings: (1) adaptability to different traffic scenarios within the same road network, and (2) generalization across different road networks. The results show that PRLight significantly reduces the adaptation time compared to learning from scratch in new TSC scenarios, achieving optimal performance using similarities between available and target scenarios.}
}
