# 基于Gzip和kNN的文本分类
该教程源于ACL2023的一篇Findings论文《"Low-Resource" Text Classification: A Parameter-Free Classification Method with Compressors》。

论文地址：https://aclanthology.org/2023.findings-acl.426.pdf

官方代码：https://github.com/bazingagin/npc_gzip

作为一种DNN的轻量级替代方法，它使用无损压缩器（如gzip）和kNN相结合的方法实现文本分类的任务。该方法利用压缩长度来近似柯氏复杂性（Kolmogorov complexity），基于归一化压缩距离（Normalized Compression Distance, NCD）度量文本之间的相似度。
