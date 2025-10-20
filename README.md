<h1 align='center'>LGflow</h1>
<h2 align='center'>一个只依赖numpy的类pytorch计算库</h2>
<p align='center'>可以达到pytorch 80%效率，以及100%的精度</p>


---

本代码库，可用于学习深度学习原理。

1. 实现了部分运算的求导
2. 实现了基于链式法则的反向梯度传播
3. 实现了简单的优化算法, 如：随机梯度下降算法（SGD）
4. 实现了部分损失函数, 如：多分类交叉熵（CrossEntropy）
5. 实现了类pytorch的Tensor, Parameter, Module类
6. 实现了部分参数初始化功能, 如：kaiming_uniform_

---

## 一. 安装

- 创建一个新的conda环境(推荐，可选)
    ```shell
    # 创建环境
    conda create -n lgflow_env python=3.8
    
    # 激活环境
    conda activate lgflow_env
    ```
  
- 安装依赖项(项目只依赖numpy)
    ```shell
    pip install numpy
    ```
  
- 使用LGflow训练MNIST数据集
    ```shell
    python train.py
    ```

## 二. 例子

1. LGflow训练一个3层网络的多层感知机(MLP)用于MNIST手写数字识别：[LGflow_train_mnist.ipynb](LGflow_train_mnist.ipynb)
2. 对于相同参数与相同输入，LGflow与torch在链式求导与权重更新方面，可以保持完全一致：[LGflow_vs_torch.ipynb](LGflow_vs_torch.ipynb)
3. 一个例子1的torch实现，用于对比LGflow与pytorch性能：[pytorch_train_mnist.ipynb](pytorch_train_mnist.ipynb)