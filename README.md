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

## 三. 与torch的差异，以及需要注意的地方

1. LGflow的梯度自动计算步骤：Tensor类调用backward函数，通过递归依次传递grad给每个参数，并调用Math类的backward函数执行具体的梯度计算。
2. LGflow即使是一个2x3的矩阵也可以调用backward计算梯度，计算时会初始化每个元素梯度为1。
3. LGflow中的Module没有backward函数，所有的梯度计算功能，都基于Math类。
4. LGflow中的Module只有两个功能：参数注册；递归获取parameters，供优化器更新。

## 四. 部分计算的实现过程

### 4.1 链式法则
对于 $y=f(x)$， $x=g(t)$

$$\frac {\partial y}{\partial t} = \frac {\partial y}{\partial x} \frac{\partial x}{\partial t}$$

在LGflow中，链式法则主要存在于两部分中：

1. Math类，定义了运算操作的前向计算与反向梯度计算
2. Tensor类，backward通过递归依次将梯度进行回传，并调用Math_op类的backward函数执行具体的梯度计算与更新。

*在LGflow中，Module类没有backward方法，所有梯度运算都继承自math类，比如relu(), log()等*

如Math_op.py中DIV_WITH_TENSOR运算: $f(x, y) = x / y$

$$\frac {\partial}{\partial x}f(x, y) = grad \frac {1}{y}$$
$$\frac {\partial}{\partial y}f(x, y) = grad \frac {-x}{y^2}$$

```python
# 张量/张量
class DIV_WITH_TENSOR(Math):
    def forward(self, from_tensors):
        return results(from_tensors[0].data / from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        data0 = from_tensors[0].data
        data1 = from_tensors[1].data
        grad0 = grad / data1
        grad1 = -grad * data0 / (data1 ** 2)

        def reduce_grad(grad, target_shape):
            # 处理shape
            ...
        
        grad0 = reduce_grad(grad0, data0.shape)
        grad1 = reduce_grad(grad1, data1.shape)
        return [grad0, grad1]
```

```python
class Tensor:
    def __init__(self, data, from_tensors=None, grad_fn=None, grad=None, dtype=float32, requires_grad=False):
        ...
    
    ...
    
    # 除
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            results = Math_op.DIV_WITH_TENSOR().forward([self, other])
            return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)
        results = Math_op.DIV_WITH_CONST().forward([self, other])
        return Tensor(data=results.data, from_tensors=results.from_tensors, grad_fn=results.grad_fn)

    
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones(self.shape)
        else:
            grad = grad
        if not self.requires_grad:
            self.grad = None

        if self.grad_fn is not None:
            grads = self.grad_fn.backward(self.from_tensors, grad)
            for tensor, grad in zip(self.from_tensors, grads):
                if isinstance(tensor, Tensor):
                    if tensor.grad is not None and tensor.grad_fn is None:
                        tensor.grad = tensor.grad + grad
                    else:
                        tensor.grad = grad
                    tensor.backward(tensor.grad)
```

### 4.2 Linear、Relu、以及CrossEntropy的实现

Math运算类可以完美支持梯度计算，所以在LGflow中，大部分的计算过程只需要写前向传播就好。

```python
class Linear(Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weights = Parameter(randn(shape=(out_features, in_features), requires_grad=True))
        self.bias = Parameter(zeros(out_features,requires_grad=True)) if use_bias else None
        self.reset_parameters()

    def forward(self, x:Tensor):
        return linear(x, self.weights, self.bias)

    def reset_parameters(self):
        kaiming_uniform_(self.weights)

    def __str__(self):
        return "LG_flow.nn.Linear(in_features={}, out_features={}, use_bias={})".format(self.in_features, self.out_features, self.use_bias)

def linear(input_tensor:Tensor, weight:Tensor, bias:Tensor=None)->Tensor:
    output = input_tensor.matmul(weight.T())
    if bias is not None:
        output = output+bias
    return output
```

```python
class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()
```

```python
class CrossEntropyLoss(Module):
    def __init__(self, reduction='sum'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return cross_entropy(input, target, self.reduction)

def softmax(input: Tensor, axis:int=1)->Tensor:
    x_max = input.max(axis=axis, keepdims=True)
    x_std = input - x_max
    x_exp = x_std.exp()
    output = x_exp / x_exp.sum(axis=axis, keepdims=True)
    return output

def cross_entropy(input:Tensor, target: Tensor, reduction: str = "mean")->Tensor:
    x_sortmax = softmax(input, axis=1)
    x_log = x_sortmax.log()
    output = - target * x_log
    output = output.sum(axis=1)
    if reduction == "mean":
        output = output.mean()
    elif reduction == "sum":
        output = output.sum()
    elif reduction == "none":
        output = output
    else:
        raise ValueError("Invalid reduction option")
    return output
```


### 4.3 SGD

LGflow的SGD实现代码非常简单，只包含参数更新功能，因为功能都分散在了其他模块中。

1. 批数据采样，是在Dataloader中实现的，Dataloader是一个支持shuffle和batch的迭代器。
2. 待更新参数的获取是通过Module类的parameters()方法获取，是一个字典类型（与torch不同）。
3. 参数的梯度，通过backward()计算。

```python
class SGD:
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for k, w in self.params.items():
            if  w is not None and w.grad is not None:
                w.grad = None

    def step(self):
        for k, w in self.params.items():
            if w is not None:
                w.data =  w.data - w.grad * self.lr
```

### 4.4 kaiming_uniform_

维持输入于输出的方差一致性，避免梯度消失或爆炸。

```python
def kaiming_uniform_(tensor: Tensor, a: float = 0, mode='fan_in'):
    assert tensor.data.ndim == 2, "Only support 2-D tensor"
    fan = tensor.data.shape[1] if mode == 'fan_in' else tensor.data.shape[0]
    gain = math.sqrt(2.0 / (1 + a ** 2)) # for relu or leaky relu
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    tensor.data = np.random.uniform(low=-bound, high=bound, size=tensor.data.shape)
```

代码较短，但是比较难理解，下面是具体的推导过程：

#### 1). bound = math.sqrt(3) * std的计算过程

bound是uniform的上下限，std是标准差。

对于uniform均匀分布，概率分布函数f(x)满足下述公式，其中a，b分别为最小值与最大值：

$$f(x)=\begin{cases} \frac{1}{(b-a)}& \text{$a<=x<=b$} \\0 \end{cases}$$

uniform的期望与方差计算公式如下：

$$
\begin{multline}
\begin{split}
E(x) 
&= \int_a^b x \frac {1}{b-a} \,dx \\ 
&= \frac {x^2}{2(b-a)} \bigg|_a^b \\
&= \frac {b^2-a^2}{2(b-a)} \\
&= \frac {b+a}{2}
\end{split}
\end{multline}
$$

$$
\begin{multline}
\begin{split}
E(x^2)
&= \int_a^b x^2 \frac {1}{b-a}\,dx \\
&= \frac {x^3}{3(b-a)} \Big|_a^b \\
&= \frac {b^3-a^3}{3(b-a)} \\
&= \frac {(b-a)(b^2+ab+a^2)}{3(b-a)} \\ 
&= \frac {b^2+ab+a^2}{3}
\end{split}
\end{multline}
$$

$$
\begin{multline}
\begin{split}
D(x)
&=E(x^2) - E(x)^2 \\
&= \frac {b^2+ab+a^2}{3} - \frac {(b+a)^2}{4} \\
&= \frac {b^2+ab+a^2}{3} - \frac {a^2+2ab+b^2}{4} \\
&= \frac {4b^2+4ab+4a^2-3a^2-6ab-3b^2}{12} \\
&= \frac {b^2-2ab+a^2}{12} \\
&= \frac {(a-b)^2}{12}
\end{split}
\end{multline}$$

对于分布于[-a, a]的uniform分布：

$$D(x) = \frac {4a^2}{12} = \frac {a^2}{3}$$

则，分布上下限参数a可表示为：

$$a = \sqrt[2] {3D(x)} = \sqrt[2] {3}std$$

#### 2). gain = math.sqrt(2.0 / 1 + a ** 2)的计算过程

gain是经过激活函数后的增益系数，定义是输入标准差与输出标准差的比值。

对于一个f(x)表示的激活函数（这里主要是relu()与leakyrelu()，relu()是a=0的leakyrelu()）

$$
f(x) = 
\begin{cases} 
x & x>=0 \\
ax & x<0
\end{cases}$$

$$
\begin{multline}
\begin{split}
Var(f(x))
&= E(f(x^2)) - E(f(x))^2 \\
&= E(x^2|x>=0)P(x>=0) + E(x^2|x<0)P(x<0) \\
\end{split}
\end{multline}
$$ 

$$
\begin{multline}
\begin{split}
Var(f(x)) 
&= E(x^2|x>=0)P(x>=0) + E(x^2|x<0)P(x<0) \\
&= E(x^2) * 0.5 + a^2E(x^2) * 0.5 \\
&= \frac {1+a^2}{2}E(x^2)
\end{split}
\end{multline}
$$

$$
\begin{multline}
\begin{split}
\frac {Var(f(x))}{Var(x)} 
&= \frac {\frac {1+a^2}{2}E(x^2)}{E(x^2) - E(x)^2} \\
&= \frac {\frac {1+a^2}{2}E(x^2)}{E(x^2)} \\
&= \frac {1+a^2}{2}
\end{split}
\end{multline}
$$

因为要保持激活函数的输入与输出具有方差一致性，所以需要对输出乘以一个增益倍数gain。

$$gain = \sqrt\frac {2}{1+a^2}$$

（gain是输入标准差与输出标准差的比值，所以需要开根号）
