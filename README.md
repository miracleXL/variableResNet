# 一种基于残差的模型训练优化方法

残差网络可以确保随网络深度增加而不降低模型的准确性，但在训练层数较多的深度学习模型时，会占用大量的设备资源并消耗大量的时间。考虑到残差的特性，先训练浅层模型快速收敛，在训练过程中逐渐新的层则可以节约大量资源，同时也能保证模型性能不会下降。  
**基于TensorFlow的代码存在不可控bug，目前仅基于PyTorch的代码可正常运行**，建议参考应用于ABIDE数据集的代码

## 文档待完成
