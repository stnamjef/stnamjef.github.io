---
layout: default
title: Back propagation in CNN
parent: Machine Learning
nav_order: 1
mathjax: true
---

# **Back propagation in CNN**



> Notation
>
> - $J$: 손실 함수(loss function)
> - $w^l_{ji}$: $l$ 번째 층의 $j$ 번째 노드와 $l-1$ 번째 층의 $i$ 번째 노드를 연결하는 가중치
> - $z^l_j$: $l$ 번째 층의 $j$ 번째 노드의 가중합(weighted sum)
> - $a^l_j$: $l$ 번째 층의 $j$ 번째 노드의 활성값(activation value)
> - $\delta^l_j$: 완전연결 층에서 $J$의 $z^l_j$에 대한 변화율, $\partial J/ \partial z^l_j$
> - $\delta^l_{ij}$: 컨볼루션 및 풀링 층에서 $J$의 $z^l_{ij}$에 대한 변화율, $\partial J/ \partial z^1_{ij}$
> - $\tau(x)$: 활성화 함수
> - 벡터는 굵은 소문자, 행렬을 굵은 대문자로 나타냄



아래는 Yann LeCun이 제안한 LeNet-5 [1]입니다. 이 구조를 예로 들어서 오차 역전파 과정을 설명하도록 하겠습니다. 

![lenet-5](\img\cnn\LeNet-5.png)



## **1. Output layer**

<br/>

![back_prop_output](\img\cnn\back_prop_output.png)

<br/>

먼저 출력 층을 살펴보겠습니다. 위 그림은 LeNet-5의 C5, F6, OUTPUT 층을 왼쪽부터 차례대로 나타낸 것입니다. 가장 오른쪽의 Class label은 ground truth label을 one-hot-vector로 표현한 것입니다. 예를 들어서 숫자 0을 나타내는 이미지가 CNN에 입력으로 주어졌다면, 이때의 ground truth label은 0이고 $(1, 0, ..., 0)^T$ 로 나타낼 수 있습니다. 한편 OUTPUT 층에서 활성화 함수로 쓰일 수 있는 함수는 여러 가지이나, 본 글에서는 하이퍼탄젠트(tanh)와 소프트맥스(softmax)로 한정해서 살펴보도록 하겠습니다. 또한 손실 함수는 MSE와 corss entropy를 사용한다고 하겠습니다.

<br/>

### **(1) tanh + mse**

<br/>

- derivative of $J$ with respect to $a^l_j$

<br/>


$$
\begin{align}
\frac{\partial J}{\partial a^l_j} &= \frac{\partial(0.5||\textbf{y} - \textbf{a}^l||^2_2)}{\partial a^l_j} \\\\&= \frac{\partial(0.5 \sum_k{(y_k - a^l_k)^2})}{\partial a^l_j} \\\\&= -(y_j - a^l_j)
\end{align}
$$


<br/>

위 수식의 두 번째 줄은 첫 번째 줄의 L2 norm을 시그마를 이용해 다시 나타낸 것입니다. 예를 들어서 $\textbf{y}$가 $(1, 2)^T$ 이고 $\textbf{a}^l$이 $(2, 1)^T$라 하면 두 점 사이의 거리는 $\sqrt{(1-2)^2 + (2-1)^2}$ 로 나타낼 수 있습니다. 위의 식은 단순히 이를 일반화 한 것입니다. 계속해서 현재 우리가 구하려고 하는 것은 OUTPUT 층의 **어떤 한 노드의 출력값** $a^l_j$에 대한 손실 함수 $J$ 의 변화율입니다. 즉 OUTPUT 층의 $j$번째 노드 외에는 신경쓰지 않아도 되기 때문에 다른 노드는 미분 값이 모두 0이 되어 사라집니다. 이에 따라 최종적으로 세 번째 줄과 같이 나타낼 수 있게 됩니다.

<br/>

- derivative of $J$ with respect to $z^l_j$

<br/>


$$
\begin{align}
\delta^l_j &= \frac{\partial J}{\partial z^l_j}\\\\&= \frac{\partial J}{\partial a^l_j} \times \frac{\partial a^l_j}{\partial z^l_j} \\\\&= \frac{\partial J}{\partial a^l_j} \times tanh'(z^l_j) \\\\&= \frac{\partial J}{\partial a^l_j} \times (1 - tanh(x)^2)
\end{align}
$$


<br/>

계속해서 OUTPUT 층의 가중합 $z^l_j$에 대한 손실함수의 변화율을 살펴보겠습니다. 먼저 알아야 하는 사실은 $a^l_j$가 $z^l_j$에 대한 함수라는 것입니다. 예를 들어 $J = f(a), a = g(z)$라면, $\partial J / \partial z = f'(g(z))g'(z)$로 나타낼 수 있습니다. 이를 생각하면 두 번째 줄에서 세 번째로 넘어가는 과정을 쉽게 이해할 수 있습니다. 이때 함수 $g$는 위의 활성화 함수에 대응한다고 생각하면 됩니다. 하이퍼탄젠트 함수의 미분은 결과만 나타내고 설명은 생략하도록 하겠습니다.

<br/>

- derivative of $J$ with respect to $w^l_{ji}$

<br/>


$$
\begin{align}
\Delta{w^l_{ji}} &= \frac{\partial J}{\partial w^l_{ji}} \\\\&= \frac{\partial J}{\partial a^l_j} \times \frac{\partial a^l_j}{\partial z^l_j} \times \frac{\partial z^l_j}{\partial w^l_{ji}} \\\\&= \delta^l_j \times a^{l - 1}_j \\\\
\end{align}
$$


<br/>

다음은 가중치 $w^l_{ji}$에 대한 변화율입니다. 위에서 설명한 것과 같이 체인룰을 적용해 미분을 여러 단계로 쪼개서 계산을 하면 됩니다. 한편 최종적으로 도출한 결과를 자세히 살펴보면 **어떤 가중치에 대한 손실함수의 변화율은 구할 때는 그 가중치와 연결된 노드만을 고려하면 된다**는 사실을 알 수 있습니다. 즉 이 변화율은 $w^l_{ji}$와 앞으로 연결된 노드로 흘러들어온 미분값 $\delta^l_j$와 뒤로 연결된 노드의 활성값  $a^{l-1}_i$의 곱으로  구할 수 있습니다. 

<br/>

- derivative of $J$ with respect to $\textbf{W}^l$(matrix notation)

<br/>


$$
\Delta \textbf{W}^l = \boldsymbol{\delta}^l \times (\textbf{a}^{l - 1})^T
$$



<br/>

### **(2) softmax + cross entropy**

<br/>

![back_prop_output](\img\cnn\back_prop_output.png)

<br/>

위는 앞서 제시한 그림과 같은 것입니다. 이번에는 같은 구조에서 OUTPUT 층의 활성화 함수가 softmax이고 손실 함수가 cross entropy 일 때 오차 역전파가 어떻게 진행되는지 알아보겠습니다.

<br/>

- cross entropy loss with softmax

<br/>


$$
\begin{align}J &= -\sum^k_j p(y_j) \times log \ q(z^l_j) \\\\&= -log \ q(z^l_y)\\\\&= -log \Big(\frac{exp(z^l_y)}{\sum_j exp(z^l_j)}\Big) \\\\&= -z^l_y + log \Big(\sum_j exp(z^l_j) \Big)\end{align}
$$



<br/>

$q(z^l_j)$는 마지막 층의 가중합 $z^l_j$에 대한 소프트맥스 함숫값으로, $q(z^l_j) = softmax(z^l_j)$로 표현할 수 있습니다. 소프트맥스는 그것의 함숫값이 [0, 1]에 위치하고, 모든 함숫값의 합이 1이 되는 특징이 있습니다. 이에 따라 $q(z^l_j)$를 확률변수 $z^l_j$에 대한 확률분포라고 할 수도 있습니다. 같은 맥락에서 $p(y_j)$는 레이블(label) $y_j$에 대한 확률분포를 나타냅니다. 일반적으로 레이블은 one-hot vector로 나타내기 때문에 정답 레이블이 되는 노드는 1의 확률을 그 외의 노드는 0의 확률을 갖습니다. 예를 들어 총 5개의 레이블 중에서 정답이 5인 벡터 $\textbf y$는 $(0,0,0,0,1)^T$ 로 나타낼 수 있습니다. 이러한 경우 당연히 $p(y_y) = p(y_5) = 1$이고 그 외의 값은 0이 됩니다. 이에 따라 **정답 레이블이 아닌 항($p(y_j)$가 0인 항)은 오차 계산시 사라지기 때문에 고려하지 않아도 됩니다.** 여기서 알 수 있는 재밌는 사실은 소프트맥스를 출력 층의 활성화 함수로 사용할 경우, 크로스 엔트로피로 나타낸 손실 함수가 로그우도로 나타낸 손실 함수와 같다는 것입니다. 위의 수식 두 번째 줄에서 이를 확인할 수 있습니다.

<br/>

- derivative of $J$ with repect to $z^l_j$

$if \quad y = j$


$$
\begin{align}
\frac{\partial J}{\partial z^l_j} &= \frac{\partial\Big(-z^l_y+log\Big(\sum_i 														exp(z^l_i)\Big)\Big)}{\partial z^l_j} \\\\
&= -1 + \frac{exp(z^l_j)}{\sum_i exp(z^l_i)} \\\\
&= -1 + softmax(z^l_j)
\end{align}
$$


$if \quad y \ne j$


$$
\begin{align}
\frac{\partial J}{\partial z^l_j} &= \frac{\partial\Big(-z^l_y+log\Big(\sum_i 														exp(z^l_i)\Big)\Big)}{\partial z^l_j} \\\\
&= 0 + \frac{exp(z^l_j)}{\sum_i exp(z^l_i)} \\\\
&= softmax(z^l_j)
\end{align}
$$


- derivative of softmax function(소프트맥스 함수 자체의 미분)

$if \quad i = j$ 


$$
\begin{align}
\frac{\partial softmax(z^l_i)}{\partial z^l_j} &= 
\frac{\partial \Big(\frac{exp(z^l_i)}{\sum^k_{j=1}exp(z^l_j)}\Big)}{\partial z^l_j}\\\\
&= \frac{exp(z^l_i)\sum_j exp(z^l_j) - exp(z^l_i)exp(z^l_j)}
		{\Big(\sum_j exp(z^l_j)\Big)^2} \\\\
&= \frac{exp(z^l_i)}{\sum_j exp(z^l_j)} \times \frac{\sum_j exp(z^l_j) - exp(z^l_j)}
													{\sum_j exp(z^l_j)} \\\\
&= softmax(z^l_i) \times (1 - softmax(z^l_j))\\
\end{align}
$$


$if \quad i \ne j$


$$
\begin{align}
\frac{\partial softmax(z^l_i)}{\partial z^l_j} &= 
\frac{\partial \Big(\frac{exp(z^l_i)}{\sum^k_{j=1}exp(z^l_j)}\Big)}{\partial z^l_j}\\\\
&= \frac{0 - exp(z^l_i)exp(z^l_j)}{\Big( \sum_j exp(z^l_j) \Big)^2} \\\\
&= \frac{exp(z^l_i)}{\sum_j exp(z^l_j)} \times \frac{exp(z^l_j)}{\sum_j exp(z^l_j)} \\\\
&= -softmax(z^l_i) \times softmax(z^l_j)
\end{align}
$$



### **(3) Fully connected layer**

<br/>

![back_prop_fc](/img/cnn/back_prop_fc.png)

<br/>

- derivative of $J$ with respect to $z^{l-1}_i$



$$
\begin{align}
\delta^{l-1}_i &= \frac{\partial J}{\partial z^{l-1}_i} \\\\
&= \sum_j \Big(\frac{\partial J}{\partial a^l_j} \times 
		  	   \frac{\partial a^l_j}{\partial z^l_j} \times
		  	   \frac{\partial z^l_j}{\partial a^{l-1}_i} \Big) \times
		  	   \frac{\partial a^{l-1}_i}{\partial z^{l-1}_i} \\\\
&= \sum_j \Big(\delta^l_j \times w^l_{ji} \Big) \times tanh'(z^{l-1}_i)
\end{align}
$$




- derivative of $J$ with respect to $\textbf{z}^{l-1}$(matrix notation)



$$
\boldsymbol{\delta}^{l-1} = \Big((\textbf{w}^l)^T \times \boldsymbol{\delta}^l \Big)
\odot tanh'(z^{l-1}_i)
$$

### **(4) Pooling layer <- Fully connected layer**

<br/>

![back_prop_pool](/img/cnn/back_prop_pool.png)

<br/>

미니배치 단위로 훈련이 이루어진다고 했을 때, 풀링 층에 입력되는 데이터의 차원은 3차원("C x H x W")이다. 반면 완전연결 층에 입력되는 데이터는 1차원("# of nodes")이다. 이에 따라 순전파 과정에서 여러 장의 이미지(C x H x W)를 1차원(# of nodes)으로 변형하는 작업이 필요한데 일반적으로 이를 flattening이라 표현한다. 위의 그림에서 주황색으로 표시된 것은 일반적인 풀링 층의 모습이고, 오른편의 첫 번째 층은 flattening을 한 후의 모습을 나타낸 것이다. 이때 데이터를 표현하는 차원은 다르지만 그 값은 같다는 점에 유의해야 한다. 

반대로 역전파 과정에서는 1차원 데이터를 다시 3차원으로 복원시키는 작업이 필요하다.  이때  **$\boldsymbol \delta^{l-1}$을 먼저 계산한 뒤 3차원으로 복원시키는 것이 구현하는 데 있어서 더 편리했기 때문에**, $\boldsymbol \delta^{l-1}$이 1차원 그레이디언트라 가정하고 아래와 같이 정리하였다. 이에 따라 정리한 수식이 일반적인 완전연결 층에서의 오차 역전파와 같은 것을 확인 할 수 있다.

- derivative of $J$ with respect to $z^{l-1}_i$



$$
\begin{align}
\delta^{l-1}_i &= \frac{\partial J}{\partial z^{l-1}_i} \\\\
&= \sum_j \delta^l_j \times w^l_{ji}
\end{align}
$$




- derivative of  $J$ with respect to $\textbf{z}^{l-1}$



$$
\textbf{z}^{l-1} = (\textbf{W}^{l})^T \times \boldsymbol{\delta}^l
$$



### **(5) Convolutional layer <- Pooling layer**

<br/>

![back_prop_conv](/img/cnn/back_prop_conv.png)

<br/>

$l$ 층은 평균 풀링(average pooling) 층이다. 풀링 층에서는 커널을 필요로 하지 않지만, 이해를 돕기 위해 그림에 나타냈다. 또한 M, N은 각각 입력 이미지의 세로, 가로 크기를 나타내고 m, n은 커널 및 출력 이미지의 세로, 가로 크기를 나타낸다. 보통의 경우 풀링 층에서 활성화 함수를 적용하지 않기 때문에 출력 값이 $a^l_{ij}$ 이 아닌  $z^l_{ij}$로 표현하였다.

- average pooling operation

아래에 정리한 식은 복잡하지만 평균 풀링 연산 자체는 간단한 개념이다. 컨볼루션 연산과 비슷하게 일정 크기의 커널이 입력 이미지 위를 훑고 지나가면서 자신과 겹치는 부분에 해당하는 이미지의 평균값을 구한다. 위의 그림을 예로 들면, 커널(가운데 행렬)을 입력 이미지 위에 겹치고, 겹친 영역에 해당하는 모든 값의 평균을 구하는 것이 평균 풀링 연산이다. 

$$
z^l_{ij} = \frac{1}{m \times n} \sum^{m-1}_{x=0}\sum^{n-1}_{y=0} a^{l-1}_{x+i, \ y+j}
$$


- derivative $J$ with respect to $z^{l-1}_{pq}$



$$
\begin{align}
\delta^{l-1}_{pq} &= \frac{\partial J}{\partial z^{l-1}_{pq}} \\\\
&= \delta^l_{ij} \times \frac{\partial z^l_{ij}}{\partial a^{l-1}_{pq}} \times 
						\frac{\partial a^{l-1}_{pq}}{\partial z^{l-1}_{pq}} \\\\
&= \delta^l_{ij} \times \frac{\partial  \frac{1}{m \times n} \sum^{m-1}_{x=0}\sum^{n-1}_{y=0} a^{l-1}_{x+i, \ y+j}}
												{\partial a^{l-1}_{pq}} \times \tau'(z^{l-1}_{pq}) \\\\
&= \frac{\delta^l_{ij}}{m \times n} \times \tau'(z^{l-1}_{pq})
\end{align}
$$



### **(6) Convolutional layer**

<br/>

![back_prop_pool2](/img/cnn/back_prop_pool2.png)

<br/>

- convolution operation


$$
\begin{align}
z^l_{ij} &= \sum^{m-1}_{x=0} \sum^{n-1}_{y=0} w^l_{xy} a^{l-1}_{i+x, \ j+y} + b^l \\\\
a^l_{ij} &= \tau(z^l_{ij})
\end{align}
$$


- derivative of $J$ with respect to $w^l_{xy}$ 


$$
\begin{align}
\frac{\partial J}{\partial w^l_{xy}} 
&= \frac{\partial J}{\partial z^l_{ij}} \times \frac{\partial z^l_{ij}}{\partial w^l_{xy}} \\\\
&= \sum^{N-m}_{i=0} \sum^{N-m}_{j=0} \delta^l_{ij} \times a^{l-1}_{i+x, \ j+y} \\\\
&= \sum^{N-m}_{i=0} \sum^{N-m}_{j=0} a^{l-1}_{i+x, \ j+y} \times \delta^l_{ij}
\end{align}
$$


컨볼루션 층의 **가중치에 대한 손실 함수의 변화량은 이전 층($l-1$)의 출력 이미지를 현재 층($l$)의 델타 이미지로 컨볼루션하는 것**과 같다.



- derivative of $J$ with respect to $b^l$


$$
\begin {align}
\frac{\partial J}{\partial b^l} &= \frac{\partial J}{\partial z^l_{ij}} \times \frac{\partial z^l_{ij}}{\partial b^l} \\\\
&= \sum^{N-m}_{i=0} \sum^{N-m}_{j=0} \delta^l_{ij} \times 1 \\\\
&= \sum^{N-m}_{i=0} \sum^{N-m}_{j=0} \delta^l_{ij}
\end{align}
$$


컨볼루션 층의 **바이어스에 대한 손실 함수의 변화량은 현재 층($l$)의 델타 이미지를 모두 더한 것**과 같다.