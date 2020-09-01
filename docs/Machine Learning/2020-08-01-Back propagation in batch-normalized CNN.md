---
layout: default
title: Back propagtion in batch-normalized CNN
parent: Machine Learning
nav_order: 2
mathjax: true
---

# **Back propagtion in batch-normalized CNN**

<br/>

이 글은 Kevin Zakka의 [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)을 읽고 그 내용을 CNN에 적용하능하도록 수식을 도출하여 정리한 것입니다. 또한 C++로 구현한 배치 정규화는 [여기](https://github.com/stnamjef/SimpleNN)에서 확인할 수 있습니다. 

<br/>

> Notation
>
> - $b$: 배치(batch) 인덱스
> - $c$: 채널(channel) 인덱스
> - $x^{(b)}_{c,i,j}$: b번 배치 c번 채널 입력 이미지의 $(i, j)$번 원소
> - $\hat{\textbf X}^{(b)}_c$: b번 배치 c번 채널의 정규화 이미지
> - $\textbf Y^{(b)}_c$: b번 배치 c번 채널의 시프트 이미지
> - $\boldsymbol \delta^{(b)}_c$: $\partial J / \partial \textbf Y^{(b)}_c$, 손실 함수의 현재 층 출력값에 대한 야코비 행렬
> - $\odot$: 아다마르 곱(Hadamard product)
> - $\oplus$: 행렬의 각 원소에 스칼라를 더하는 연산
> - $\ominus$: 행렬의 각 원소에서 스칼라를 빼는 연산
> - $sum()$: 행렬의 모든 원소를 더하는 함수
> - 행렬을 굵은 대문자, 벡터는 굵은 소문자로 표기함



## **1. Forward propagation**

<br/>

오차 역전파 과정을 살피기에 앞서 순전파 과정을 간략하게 살펴보도록 하자.

<br/>

![forward_propagation](/img/batch normalization/forward_propagation.png)

<br/>

순전파 과정에서는 먼저 각 특성맵(feature maps)의 평균과 분산을 구한다. 이때 평균과 분산은 전체 배치에 대하여 구하되, 각 채널별로 계산하여야 한다. 즉 각각의 채널에 대해 평균과 분산을 따로 구한다. 예를 들어 빨간색으로 표시한 R 채널의 평균은 배치 내의 모든 R 채널 특성맵의 원소를 더한 후 그것의 수로 나누어 구한다. 자세한 수식은 아래에 따로 작성하였다. 계속해서 정규화된 이미지를 $\gamma$와 $\beta$로 시프트하면 순전파 과정이 종료된다. 이때 $\gamma$와 $\beta$도 각 채널에 대하여 한 쌍씩 생성한다.


$$
\begin{align}
\mu_c &= \frac{1}{mpq}\sum^m_{b=1} \sum^p_{i=1} \sum^q_{j=1} x^{(b)}_{c,i,j} \\\\
&= \frac{1}{mpq} sum \Big( \sum^m_{b=1} \textbf X^{(b)}_c \Big) \\\\
\sigma^2_c &= \frac{1}{mpq} \sum^m_{b=1} \sum^p_{i=1} \sum^q_{j=1} (x^{(b)}_{c,i,j} - \mu_c)^2 \\\\
&= \frac{1}{mpq} sum \Big( \sum^m_{b=1} (\textbf X^{(b)}_c \ominus \mu_c)^2 \Big) \\\\
\hat{\textbf X}^{(b)}_c &= \frac{\textbf X^{(b)}_c \ominus \mu_c}{\sqrt{\sigma^2_c + \epsilon}} \\\\
\textbf Y^{(b)}_c &= \gamma_c \hat{\textbf X}^{(b)}_c \oplus \beta_c
\end{align}
$$



## **2. Backward propagation**







### **(1) derivative of $J$ with respect to $\gamma_c$**

<br/>

![back_prop_1](/img/batch normalization/back_prop_1.png)

<br/>

위 그림은 처음에 제시한 순전파 그림에서 $\boldsymbol \gamma, \boldsymbol \beta$를 이용해 시프트하는 과정을 떼어서 역전파에 맞게 용어만 수정한 것이다. 이때 $\boldsymbol \delta^{(b)}_c$는 델타 이미지로 현재 층의 출력 이미지에 대한 손실함수의 변화율, $\partial J / \partial \textbf Y^{(b)}_c$을 의미한다. 결론부터 말하자면 $J$의 $\gamma_c$에 대한 변화율은 같은 배치 인덱스$\cdot$채널 인덱스를 갖는 델타 이미지와 정규화 이미지 간 아다마르 곱(Hadamard product)을 시행하고 그 원소들의 값을 모두 더한 것이 된다. 예를 들어 첫 번째 채널의 감마에 대한 손실 함수의 변화율인 $\partial J / \partial \gamma_1$은 먼저 다음 세 개의 쌍 $(\boldsymbol \delta^{(1)}_1, \textbf X^{(1)}_1 )$, $(\boldsymbol \delta^{(2)}_1, \textbf X^{(2)}_1 )$, $(\boldsymbol \delta^{(3)}_1, \textbf X^{(3)}_1 )$에 대해 원소 간 곱셈을 한다. 그렇게 하면 세 개의 행렬이 나올 것이고, 이들을 모두 합(행렬 간 덧셈)하는 것이 아래의 수식에서 $\sum$가 하는 역할이다. 마지막으로 $sum()$에 의해 이 행렬의 모든 원소가 하나의 스칼라 값으로 더해진다. $\gamma_1$ 하나에 대한 변화율을 구하는데 여러 장의 이미지가 사용되는 이유는 순전파 과정에서 **시프트 연산을 할 때 $\gamma_1$이 배치 내의 모든 1번 채널 이미지에 대해 적용되었기 때문**이다. 지금까지의 설명을 수식으로 표현하면 아래와 같다. 이때 채널 인덱스($c$)는 고정되어 있고, 배치 인덱스($b$)는 고정되어 있지 않음에 유의해야 한다.


$$
\begin{align}
\frac{\partial J}{\partial \gamma_c} &=
\frac{\partial J}{\textbf Y^{(b)}_c} 
\frac{\partial \textbf Y^{(b)}_c}{\partial \gamma_c} \\\\
&= sum \Big( \sum^m_{b=1} \boldsymbol \delta^{(b)}_c \odot \hat{\textbf X}^{(b)}_c \Big)
\end{align}
$$



### **(2) derivative of $J$ with respect to $\beta_c$**

<br/>

$\partial J / \partial \beta_c$를 구하는 과정도 위와 매우 유사하다. 단, 앞서 출력 이미지를 감마로 편미분 했을 경우에는  $\partial \textbf Y^{(b)}_c / \partial \gamma_c = \hat{\textbf X}^{(b)}_c$이기 때문에 델타 이미지와 정규화 이미지 간 아다마르 곱을 해야 했으나, 여기서는 출력 이미지를 베타로 편미분하면 $\partial \textbf Y^{(b)}_c / \partial \beta_c = 1$이기 때문에 단순히 모든 델타 이미지를 더하면 된다는 차이가 있다. 임의의 행렬 $\textbf A$와 모든 원소가 1인 행렬 간 아다마르 곱은 그대로 행렬 $\textbf A$이기 때문에 곱셉을 생략할 수 있다.


$$
\begin{align}
\frac{\partial J}{\partial \beta_c} &=
\frac{\partial J}{\partial \textbf Y^{(b)}_c} 
\frac{\partial \textbf Y^{(b)}_c}{\partial \beta_c} \\\\
&= sum\Big( \sum^m_{b=1} \boldsymbol \delta^{(b)}_c \Big)
\end{align}
$$



### **(3) derivative of $J$ with respect to $\hat{\textbf X}^{(b)}_c$**

<br/>

$\partial J / \partial \hat{\textbf X}^{(b)}_c$는 아래와 같이 전개할 수 있다. 전개식에서 첫 번째 항은 앞서 (1)과 (2)를 유도하는 과정에서 계산한 것과 같이 델타 이미지 $\boldsymbol \delta^{(b)}_c$가 된다. 두 번째 항인 $\partial \textbf Y^{(b)}_c / \partial \hat{\textbf X}^{(b)}_c$은 출력 이미지를 정규화 이미지로 편미분 한 것이다. 순전파 과정에서 출력 이미지 $\textbf Y^{(b)}_c = \gamma_c \hat{\textbf X}^{(b)}_c \oplus \beta_c$이었기 때문에, 편미분 값은 상수 $\gamma_c$가 된다. 이때 유의해야 할 점은 $\boldsymbol \delta^{(b)}_c \times \gamma_c$가 "행렬 x 스칼라" 형태이기 때문에 델타 이미지의 모든 원소에 감마 값을 곱해야 한다는 것이다.


$$
\begin{align}
\frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} &= \frac{\partial J}{\partial \textbf Y^{(b)}_c}
								\frac{\partial \textbf Y^{(b)}_c}{\partial \hat{\textbf X}^{(b)}_c} \\\\
&= \boldsymbol \delta^{(b)}_c \times \gamma_c
\end{align}
$$



### **(4) derivative of $J$ with respect to $\mu_c$**

<br/>

![back_prop_2](/img/batch normalization/back_prop_2.png)

<br/>

위 그림은 처음에 제시한 순전파 그림에서 $\boldsymbol \mu, \boldsymbol \sigma^2$을 이용해 정규화하는 과정을 떼어서 역전파에 맞게 용어만 수정한 것이다. 먼저 $J$의 $\mu_c$에 대한 편미분은 다음과 같이 두 부분으로 나누어 생각할 수 있다. 이때 수식 전개가 복잡하기 때문에 각 파트를 따로 계산한다.



$$
\begin{align}
\frac{\partial J}{\partial \mu_c} &= \underbrace{ \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c}
									 \frac{\partial \hat{\textbf X}^{(b)}_c}{\partial \mu_c}}_{1} +
									 \underbrace{ \frac{\partial J}{\partial \sigma^2_c}
									 \frac{\partial \sigma^2_c}{\partial \mu_c} }_{2}
\end{align}
$$



먼저 첫 번째 파트에 대한 전개는 다음과 같다. 이때 $\partial J / \partial \hat{\textbf X}^{(b)}_c$는 행렬이라는 점에 유의해야 한다. 아래 수식에서 $\sum$ 이후의 의미는 정규화 이미지의 각 원소에 대한 손실 함수의 변화율을 계산하고, 각각의 값에 일정한 상수$(-1 / \sqrt{\sigma^2_c + \epsilon})$를 곱한 후, 그 행렬을 모두 더한다는 것이다. 마지막으로 $sum()$에 의해 행렬의 모든 원소가 하나의 하나의 스칼라 값으로 더해진다. 한편 배치 인덱스 $b$는 $\sum$에 의해 배치 전체를 순회하지만 채널 인덱스 $c$는 고정되어 있기 때문에 상수는 $\sum$ 바깥으로 뺄 수 있다.



$$
\begin{align}
\frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \frac{\partial \hat{\textbf X}^{(b)}_c}{\partial \mu_c}
&= sum\Big( \sum^m_{b=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \times 
						 \frac{-1}{\sqrt{\sigma^2_c + \epsilon}} \Big) \\\\
&= -(\sigma^2_c + \epsilon)^{-0.5} \times sum\Big( \sum^m_{b=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \Big)
\quad \cdots \quad (1)
\end{align}
$$



두 번째 파트의 첫 번째 항은 아래와 같이 전개한다. 두 번째 줄은 단순히 $\hat{\textbf X}^{(b)}_c$를 풀어서 쓴 것이다.


$$
\begin{align}
\frac{\partial J}{\partial \sigma^2_c} &= \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c}
										  \frac{\partial \hat{\textbf X}^{(b)}_c}{\partial \sigma^2_c} \\\\
&= \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c}
   \frac{\partial (\textbf X^{(b)}_c \ominus \mu_c)(\sigma^2_c + \epsilon)^{-0.5}}{\partial \sigma^2_c} \\\\
&= -0.5 \times sum \Big( \sum^m_{b=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c}
						 \odot (\textbf X^{(b)}_c \ominus \mu_c) \times (\sigma^2_c + \epsilon)^{-1.5} \Big)
\quad \cdots \quad (2-1)
\end{align}
$$


계속해서 두 번째 파트의 두 번째 항에 대한 전개는 아래와 같다.



$$
\begin{align}
\frac{\partial \sigma^2_c}{\partial \mu_c}
&= \frac{\partial \frac{1}{mpq} \sum^m_{b=1} \sum^p_{i=1} \sum^q_{j=1} (x^{(b)}_{c,i,j} - \mu_c)^2}{\partial \mu_c} \\\\
&= \frac{-2}{mpq}\sum^m_{b=1} \sum^p_{i=1} \sum^q_{j=1} (x^{(b)}_{c,i,j} - \mu_c) \\\\
&= -2 \Big( \frac{1}{mpq} \sum^m_{b=1} \sum^p_{i=1} \sum^q_{j=1} x^{(b)}_{c,i,j} - 
			\frac{1}{mpq} \sum^m_{b=1} \sum^p_{i=1} \sum^q_{j=1} \mu_c \Big) \\\\
&= -2 \Big(\mu_c - \frac{mpq}{mpq} \mu_c \Big) \\\\
&= 0
\quad \cdots \quad (2-2)
\end{align}
$$



위에서 볼 수 있듯이 두 번째 파트는 0이 되기 때문에, $\partial J / \partial \mu_c$는 아래와 같이 정리할 수 있다.



$$
\begin{align}
\frac{\partial J}{\partial \mu_c} &= \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c}
									 \frac{\partial \hat{\textbf X}^{(b)}_c}{\partial \mu_c} +
									 \frac{\partial J}{\partial \sigma^2_c}
									 \frac{\partial \sigma^2_c}{\partial \mu_c} \\\\
&= \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c}
   \frac{\partial \hat{\textbf X}^{(b)}_c}{\partial \mu_c} \\\\
&= -(\sigma^2_c + \epsilon)^{-0.5} \times sum\Big( \sum^m_{b=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \Big)
\end{align}
$$


### **(5) derivative of $J$ with respect to $\textbf X^{(b)}_c$**

<br/>

$J$의 $X^{(b)}_c$에 대한 편미분은 다음과 같이 세 개의 파트로 나누어 생각할 수 있다.


$$
\frac{\partial J}{\partial \textbf X^{(b)}_c} =
\underbrace{\frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c}
			\frac{\partial \hat{\textbf X}^{(b)}_c}{\partial \textbf X^{(b)}_c}}_1 +
\underbrace{\frac{\partial J}{\partial \mu_c} \frac{\partial \mu_c}{\partial \textbf X^{(b)}_c}}_2 +
\underbrace{\frac{\partial J}{\partial \sigma^2_c} \frac{\partial \sigma^2_c}{\partial \textbf X^{(b)}_c}}_3
$$


먼저 첫 번째 파트에 대해 전개하면 아래와 같다.


$$
\begin{align}
\frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c}
\frac{\partial \hat{\textbf X}^{(b)}_c}{\partial \textbf X^{(b)}_c}
&= \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \times \frac{1}{\sqrt{\sigma^2_c + \epsilon}} \\\\
&= (\sigma^2_c + \epsilon)^{-0.5} \times \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c}
\end{align}
$$


두 번째 파트에 대한 전개는 아래와 같다.


$$
\begin{align}
\frac{\partial J}{\partial \mu_c} \frac{\partial \mu_c}{\partial \textbf X^{(b)}_c}
&= \frac{\partial J}{\partial \hat{\textbf X^{(b)}_c}}
   \frac{\partial \hat{\textbf X^{(b)}_c}}{\partial \mu_c}
   \frac{\partial \mu_c}{\partial \textbf X^{(b)}_c} \\\\
&= sum\Big( \sum^m_{b=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \times
			\frac{-1}{\sqrt{\sigma^2_c + \epsilon}} \times
            \frac{1}{mpq} \Big) \\\\
&= -\frac{(\sigma^2_c + \epsilon)^{-0.5}}{mpq} \times 
	sum \Big( \sum^m_{b=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \Big)
\end{align}
$$

마지막으로 세 번째 파트에 대한 전개는 아래와 같다.


$$
\begin{align}
\frac{\partial J}{\partial \sigma^2_c} \frac{\partial \sigma^2_c}{\partial \textbf X^{(b)}_c}
&= \underbrace{ \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \cdot
   \frac{\partial \hat{\textbf X}^{(b)}_c} {\partial \sigma^2_c} }_1 \cdot 
   \underbrace{ \frac{\sigma^2_c}{\textbf X^{(b)}_c} }_2 \\\\
&= \underbrace{ -0.5 \times sum \Big( \sum^m_{b'=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b')}_c}
						 \odot (\textbf X^{(b')}_c \ominus \mu_c) \times (\sigma^2_c + \epsilon)^{-1.5} \Big) }_1 \times
   \underbrace{ \frac{2}{mpq} (\textbf X^{(b)}_c \ominus \mu_c) }_2 \\\\
&= -\frac{(\sigma^2_c + \epsilon)^{-0.5}}{mpq} \times 
	\frac{(\textbf X^{(b)}_c \ominus \mu_c)}{\sqrt{\sigma^2 + \epsilon}} \times
	sum \Big( \sum^m_{b'=1}  \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \odot
    		  \frac{(\textbf X^{(b')}_c \ominus \mu_c)}{\sqrt{\sigma^2_c + \epsilon}} \Big) \\\\
&= -\frac{(\sigma^2_c + \epsilon)^{-0.5}}{mpq} \times
	\hat{\textbf X}^{(b)}_c \times
	sum \Big( \sum^m_{b'=1}  \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} \odot \hat{\textbf X}^{(b')}_c \Big)
\end{align}
$$


지금까지 전개한 세 가지 부분을 하나로 합치면 아래와 같다.


$$
\begin{align}
\frac{\partial J}{\partial \textbf X^{(b)}_c}
&= \underbrace{ (\sigma^2_c + \epsilon)^{-0.5} \times \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} }_1 -
\underbrace{ \frac{(\sigma^2_c + \epsilon)^{-0.5}}{mpq} \times 
			 sum \Big( \sum^m_{b'=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b')}_c} \Big)}_2 -
\underbrace{ \frac{(\sigma^2_c + \epsilon)^{-0.5}}{mpq} \times
			 \hat{\textbf X}^{(b)}_c \times
			 sum \Big( \sum^m_{b'=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b')}_c} \odot
			 \hat{\textbf X}^{(b')}_c \Big) }_3 \\\\
&= \frac{(\sigma^2_c + \epsilon)^{-0.5}}{mpq}
   \Big\{ mpq \times \frac{\partial J}{\partial \hat{\textbf X}^{(b)}_c} 
   		  - sum \Big( \sum^m_{b'=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b')}_c} \Big)
   		  - \hat{\textbf X}^{(b)}_c \times sum
   		  \Big( \sum^m_{b'=1} \frac{\partial J}{\partial \hat{\textbf X}^{(b')}_c} \odot
			 \hat{\textbf X}^{(b')}_c \Big)
   \Big\}
\end{align}
$$

## **3. Reference**

- Sergey Ioffe, Christian Szegedy (2015), Batch Normalization: Accelerating deep network training by reducing internal covariate shift, Proceedings of the ICML.

- Kevin Zakka (2016), [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)

