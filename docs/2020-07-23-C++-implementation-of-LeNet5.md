# LeNet-5의 구현 및 성능 향상



본 글은 2020-1학기 캡스톤디자인 과목에서 수행한 프로젝트를 정리하기 위해 작성한 것입니다. 프로젝트의 주요 내용을 요약하여 작성했고, 모델을 구현하면서 어려웠던 부분은 부록에 정리하였습니다.



## 1. Overview



> *참고: 특성맵(feature maps)의 차원은 **"height x width x channels"**로, 커널(kernel)의 차원은 **"height x width x channels x # of kernels(커널의 개수)"**로 표기하였다. 또한 C는 컨볼루션 층(convolution layer)을, S는 서브샘플링 층(subsampling layer; pooling layer)을, K는 커널을 의미한다. 



#### (1) Architecture



![LeNet-5](../../img/cnn/LeNet-5_modified.png)



위의 그림은 본 프로젝트에서 최종적으로 구현한 합성신경망의 아키텍처이다. LeNet-5의 구조를 그대로 따르되, 컨볼루션 층과 완전연결 층 뒤에 배치 정규화 층을 추가하였다. 또한 활성화 함수를 하이퍼볼릭탄젠트에서 렐루(ReLU)로 변경하였으며, 출력 층에서는 소프트맥스(softmax)를 사용하였다. 마지막으로 손실 함수는 크로스 엔트로피(cross entropy)로 변경하였다.



#### (2) Problems



- Vanishing gradient



LeNet-5는 출력 층을 제외한 모든 층에서 활성화 함수로 하이퍼볼릭탄젠트를 이용한다. 하이퍼볼릭탄젠트의 도함수는 입력값이 0에서 멀어질수록 함숫값이 0으로 수렴한다. 이에 따라 그래디언트의 크기가 작아져서 신경망의 학습이 더뎌지는 문제가 발생한다. 현대의 신경망 모델은 이러한 문제를 해결하고자 렐루(ReLU), 소프트맥스(softmax) 등을 활성화 함수로 사용한다. 따라서 LeNet-5의 활성화 함수를 위 두 함수로 변경하고 그에 따른 성능의 차이를 측정한다.



- Covariate shift



LeNet-5가 사용하는 그래디언트 기반 학습 방법은 학습률(learning rate)의 초깃값에 매우 민감하다. 예를 들어 학습률이 지나치게 크면 그래디언트가 폭발하여 학습이 제대로 되지 않는 문제가 발생한다. 또한 신경망의 구조적 특성상 각 층에 입력되는 훈련 집합의 분포가 달라지는 공변량 시프트가 발생한다. 이 역시 신경망의 학습을 저해하는 요인이다. 배치 정규화(batch normalization)은 공변량 시프트 문제를 해결하고 신경망이 학습하는 데 있어서 학습률의 영향을 덜 받도록 하는 기법이다. LeNet-5에 배치 정규화를 적용한 후 그에 따른 성능의 차이를 측정한다.



- Overfitting



신경망을 학습할 때 모델의 복잡도에 비해 데이터가 부족하면, 작은 값이었던 가중치들이 점차 증가하면서 과적합이 발생할 수 있다. 가중치 감쇠(weight decay)는 목적 함수에 규제 항을 두어 가중치의 영향력을 줄이는 식으로 과적합을 억제한다. 규제 항의 종류는 여러 가지이나 본 프로젝트에서는 가장 널리 쓰이는 L2놈을 이용한다. LeNet-5에 L2놈 규제 항을 적용한 후 그에 따른 성능의 차이를 측정한다.



#### (3) Objectives



- C++로 LeNet-5를 구현하며 stl(standard template library)을 제외한 다른 라이브러리는 사용하지 않는다.
- 구현한 LeNet-5의 분류 정확도(accuracy)가 99% 이상이 되도록 한다. 정확도 측정 시 데이터셋은 MNIST를 이용한다.
- 활성화 함수의 변경, 배치 정규화, 가중치 감쇠를 적용한 모델의 정확도와 기존 모델의 정확도를 비교 분석한다.



## 2. Schedule



|  주차   | 내용                                                         |
| :-----: | :----------------------------------------------------------- |
| 2~3주차 | - 관심 주제 탐구 및 신청서 작성.                             |
|  4주차  | - DNN 구현.                                                  |
|         | - Iris 데이터셋을 이용하여 성능을 측정.                      |
|  5주차  | - DNN에 Xavier initialization, Min-max normalization을 적용. |
|         | - MNIST 데이터셋을 이용하여 성능을 측정.                     |
|  6주차  | - LeNet-5 구현 시작.                                         |
|         | - Convolution, Pooling 연산 구현.                            |
|  7주차  | - 논문 리뷰: Yann LeCun et al. (1998), Gradient-based learning applied to document recognition. |
|  8주차  | - CNN의 오차 역전파 과정의 이론적인 내용 학습.               |
|  9주차  | - matrix.h(행렬 연산을 위한 클래스) 구현.                    |
| 10주차  | - LeNet-5 구현 완료.                                         |
|         | - common.h, layer.h, convolutional_layer.h, pooling_layer.h, |
|         | - dense_layer.h, activation_layer.h, output_layer.h 구현.    |
| 11주차  | - MNIST 데이터셋을 이용하여 성능 측정.                       |
|         | - 성능 측정 과정에서 Pooling 연산에서 버그가 발견되어 수정.  |
| 12주차  | - (1) Tanh + MSE 모델(기준이 되는 모델; baseline model)의 성능을 측정. |
| 13주차  | - 활성화 함수(softmax), 손실 함수(Cross entropy), 규제 항(L2 norm) 구현. |
|         | - (2) Tanh + MSE + Regularization(0.001) 모델 성능 측정.     |
|         | - (3) ReLu + Softmax + Cross entropy 모델 성능 측정.         |
|         | - (4) ReLu + Softmax + Cross entropy + Regularization(0.001) 모델 성능 측정. |
| 14 주차 | - Batch normalization 구현.                                  |
|         | - (5) Tanh + MSE + Batch normalization 모델 성능 측정.       |
|         | - (6) ReLu + Softmax + Cross entropy + Batch normalization 모델 성능 측정. |
| 15주차  | - 소스코드 정리 및 보고서 작성                               |



## 3. Results(tested on MNIST)



#### (1) Tanh + MSE(baseline model)

- Epoch: 20, Batch: 20, Learning rate: 0.5
- Training error: 0.62%, Testing error: 1.01%





#### (2) Tanh + MSE + Regularization

- Epoch: 20, Batch:20, Learning rate: 0.5, Lambda: 0.001
- Training error: 1.22%, Testing error: 1.33%





#### (3) Tanh + MSE + Batch normalization

- Epoch: 20, Batch: 20, Learning rate: 0.5
- Training error: 0%, Testing error: 0%



## 4. Conclusion



- 구현한 모델의 에러율(error rate)이 약 1%로 논문에서 제시한 0.95%에 근접함.
- 활성화 함수를 relu, softmax로 손실 함수를 cross entropy로 변경하였을 때 학습이 빨라지는 것을 확인함.
- 가중치 감쇠, 배치 정규화를 적용했을 때 과적합이 억제되는 것을 확인함.
- 한편 배치 정규화를 적용했을 때 모델 성능이 더 향상될 것이라 예상했으나, 측정 결과 오히려 성능이 떨어진 것을 확인함.



## 5. Examples

Construct LeNet-5 with batch normalization

```c++
int n_img_train = 60000, n_img_test = 10000, img_size = 784;
Tensor train_X, test_X;
Vector train_Y, test_Y;

ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X);
ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X);
ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

vector<vector<int>> indices(6, vector<int>(16));
indices[0] = { 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1 };
indices[1] = { 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1 };
indices[2] = { 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1 };
indices[3] = { 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1 };
indices[4] = { 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1 };
indices[5] = { 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1 };

SimpleNN model;
model.add(new Conv2D(6, { 5, 5 }, 2, "uniform", { 28, 28, 1 }));
model.add(new BatchNorm);
model.add(new Activation("tanh"));
model.add(new Pool2D({ 2, 2 }, 2, "avg"));
model.add(new Conv2D(16, { 5, 5 }, 0, "uniform", indices));
model.add(new BatchNorm);
model.add(new Activation("tanh"));
model.add(new Pool2D({ 2, 2 }, 2, "avg"));
model.add(new Flatten);
model.add(new Dense(120, "uniform"));
model.add(new BatchNorm);
model.add(new Activation("tanh"));
model.add(new Dense(84, "uniform"));
model.add(new BatchNorm);
model.add(new Activation("tanh"));
model.add(new Dense(10, "uniform"));
model.add(new BatchNorm);
model.add(new Activation("tanh"));
model.add(new Output(10, "mse"));

int n_epoch = 20, batch = 20;
double l_rate = 0.5, lambda = 0.001;

model.fit(train_X, train_Y, l_rate, n_epoch, batch, lambda, test_X, test_Y);
```



Construct DNN(784 x 500 x 150 x 10 x 10)

```c++
int n_img_train = 60000, n_img_test = 10000, img_size = 784;
Tensor train_X, test_X;
Vector train_Y, test_Y;

ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X, true);
ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X, true);
ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

SimpleNN model;
model.add(new Dense(500, "uniform", 784));
model.add(new Activation("relu"));
model.add(new Dense(150, "uniform"));
model.add(new Activation("relu"));
model.add(new Dense(10, "uniform"));
model.add(new Activation("softmax"));
model.add(new Output(10, "cross entropy"));

int n_epoch = 30, batch = 30;
double l_rate = 0.1, lambda = 0.0;

model.fit(train_X, train_Y, l_rate, n_epoch, batch, lambda, test_X, test_Y);
```



## 6. Appendix: back propagation of each layer



> Notation
>
> - $J$: 손실 함수(loss function)
> - $w^l_{ji}$: $l$ 번째 층의 $j$ 번째 노드와 $l-1$ 번째 층의 $i$ 번째 노드를 연결하는 가중치
> - $z^l_j$: $l$ 번째 층의 $j$ 번째 노드의 가중합(weighted sum)
> - $a^l_j$: $l$ 번째 층의 $j$ 번째 노드의 활성값(activation value)
> - $\delta^l_j$: $l$ 번째 층의 $j$ 번째 노드의 그레이디언트(gradient), 컨볼루션 및 풀링 층의 출력은 행렬이지만 편의상 같은 문자로 표기함
> - $\tau(x)$: 활성화 함수
> - 벡터는 굵은 소문자, 행렬을 굵은 대문자로 나타냄



#### (1) Output layer: tanh + mse



![back_prop_output](..\img\cnn\back_prop_output.png)



- derivative of $J$ with respect to $a^l_j$



$$
\begin{align}
\frac{\partial J}{\partial a^l_j} &= 
\frac{\partial(0.5||\textbf{y} - \textbf{a}^l||^2_2)}{\partial a^l_j} \\\\
&= \frac{\partial(0.5 \sum_k{(y_k - a^l_k)^2})}{\partial a^l_j} \\\\
&= -(y_j - a^l_j)
\end{align}
$$




- derivative of $J$ with respect to $z^l_j$



$$
\begin{align}
\delta^l_j &= \frac{\partial J}{\partial z^l_j}\\\\
&= \frac{\partial J}{\partial a^l_j} \times \frac{\partial a^l_j}{\partial z^l_j} \\\\
&= \frac{\partial J}{\partial a^l_j} \times tanh'(z^l_j) \\\\
&= \frac{\partial J}{\partial a^l_j} \times (1 - tanh(x)^2)
\end{align}
$$




- derivative of $J$ with respect to $w^l_{ji}$



$$
\begin{align}
\Delta{w^l_{ji}} &= \frac{\partial J}{\partial w^l_{ji}} \\\\
&= \frac{\partial J}{\partial a^l_j} \times \frac{\partial a^l_j}{\partial z^l_j} \times \frac{\partial z^l_j}{\partial w^l_{ji}} \\\\
&= \delta^l_j \times a^{l - 1}_j \\\\
\end{align}
$$




- derivative of $J$ with respect to $\textbf{W}^l$(matrix notation)



$$
\Delta \textbf{W}^l = \boldsymbol{\delta}^l \times (\textbf{a}^{l - 1})^T
$$



- derivative of $J$ with respect to $b^l_j$


$$
\begin{align}
\Delta b^l_j &= \frac{\partial J}{\partial b^l} \\\\
&= \frac{\partial J}{\partial a^l_j} \times \frac{\partial a^l_j}{\partial z^l_j} \times \frac{\partial z^l_j}{\partial b^l_j} \\\\
&= \delta^l_j
\end{align}
$$


#### (2) Output layer: softmax + cross entropy



![back_prop_output](..\img\cnn\back_prop_output.png)



- cross entropy loss with softmax



$$
\begin{align}
J &= -\sum^k_j p(y_j) \times log \ q(z^l_j) \\\\
&= -log \ q(z^l_y)\\\\
&= -log \Big(\frac{exp(z^l_y)}{\sum_j exp(z^l_j)}\Big) \\\\
&= -z^l_y + log \Big(\sum_j exp(z^l_j) \Big)
\end{align}
$$



$q(z^l_j)$는 마지막 층의 가중합 $z^l_j$에 대한 확률질량함수이다. 이때 $q(z^l_j) = softmax(z^l_j)$로 표현할 수 있다. 소프트맥스는 그것의 함숫값이 [0, 1]에 위치하고, 모든 함숫값의 합이 1이 되어 확률질량함수의 특성을 갖기 때문이다. 계속해서 $p(y_j)$는 레이블(label) $y_j$에 대한 확률질량함수이다. 일반적으로 레이블은 원-핫 벡터(one-hot vector)로 나타내기 때문에 정답 레이블이 되는 노드는 1의 확률을 그 외의 노드는 0의 확률을 갖는다. 예를 들어 총 5개의 레이블 중에서 정답이 5인 벡터 $\textbf y$는 $(0,0,0,0,1)^T$ 로 나타낼 수 있다. 이러한 경우 당연히 $p(y_y) = p(y_5) = 1$이고 그 외의 값은 0이 된다. 이에 따라 정답 레이블이 아닌 항($p(y_j)$가 0인 항)은 오차 계산시 사라지기 때문에 고려하지 않아도 된다. 여기서 알 수 있는 재밌는 사실은 소프트맥스를 출력 층의 활성화 함수로 사용할 경우, 크로스 엔트로피로 나타낸 손실 함수가 로그우도로 나타낸 손실 함수와 같다는 것이다.



- derivative of $J$ with repect to $z^l_j$



​		$if \quad y = j$
$$
\begin{align}
\frac{\partial J}{\partial z^l_j} &= \frac{\partial\Big(-z^l_y+log\Big(\sum_i 														exp(z^l_i)\Big)\Big)}{\partial z^l_j} \\\\
&= -1 + \frac{exp(z^l_j)}{\sum_i exp(z^l_i)} \\\\
&= -1 + softmax(z^l_j)
\end{align}
$$
​		$if \quad y \ne j$
$$
\begin{align}
\frac{\partial J}{\partial z^l_j} &= \frac{\partial\Big(-z^l_y+log\Big(\sum_i 														exp(z^l_i)\Big)\Big)}{\partial z^l_j} \\\\
&= 0 + \frac{exp(z^l_j)}{\sum_i exp(z^l_i)} \\\\
&= softmax(z^l_j)
\end{align}
$$


- derivative of softmax function



​		$if \quad i = j$
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
​		$if \quad i \ne j$
$$
\begin{align}
\frac{\partial softmax(z^l_i)}{\partial z^l_j} &= 
\frac{\partial \Big(\frac{exp(z^l_i)}{\sum^k_{j=1}exp(z^l_j)}\Big)}{\partial z^l_j}\\\\
&= \frac{0 - exp(z^l_i)exp(z^l_j)}{\Big( \sum_j exp(z^l_j) \Big)^2} \\\\
&= \frac{exp(z^l_i)}{\sum_j exp(z^l_j)} \times \frac{exp(z^l_j)}{\sum_j exp(z^l_j)} \\\\
&= -softmax(z^l_i) \times softmax(z^l_j)
\end{align}
$$



#### (3) Fully connected layer



![back_prop_fc](..\img\cnn\back_prop_fc.png)



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



#### (4) Pooling layer <- Fully connected layer



![back_prop_pool](..\img\cnn\back_prop_pool.png)



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



#### (5) Convolutional layer <- Pooling layer



![back_prop_conv](..\img\cnn\back_prop_conv.png)



$l$ 층은 평균 풀링(average pooling) 층이다. 풀링 층에서는 커널을 필요로 하지 않지만, 이해를 돕기 위해 그림에 나타냈다. 또한 M, N은 각각 입력 이미지의 세로, 가로 크기를 나타내고 m, n은 커널 및 출력 이미지의 세로, 가로 크기를 나타낸다. 보통의 경우 풀링 층에서 활성화 함수를 적용하지 않기 때문에 $a^l_{ij}$ 대신 $z^l_{ij}$로 나타냈다. 이러한 가정 하에 평균 풀링 연산과 $l-1$ 층의 그레이디언트 $\boldsymbol{\delta}^{l-1}$은 아래와 같이 정의할 수 있다. 



- average pooling operation



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



#### (6) Convolutional layer



![back_prop_pool2](..\img\cnn\back_prop_pool2.png)



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


컨볼루션 층의 가중치에 대한 손실 함수의 변화량은 이전 층($l-1$)의 출력 이미지를 현재 층($l$)의 델타 이미지로 컨볼루션하는 것과 같다.



- derivative of $J$ with respect to $b^l$


$$
\begin {align}
\frac{\partial J}{\partial b^l} &= \frac{\partial J}{\partial z^l_{ij}} \times \frac{\partial z^l_{ij}}{\partial b^l} \\\\
&= \sum^{N-m}_{i=0} \sum^{N-m}_{j=0} \delta^l_{ij} \times 1 \\\\
&= \sum^{N-m}_{i=0} \sum^{N-m}_{j=0} \delta^l_{ij}
\end{align}
$$


컨볼루션 층의 바이어스에 대한 손실 함수의 변화량은 현재 층($l$)의 델타 이미지를 모두 더한 것과 같다.



#### (7) Pooling layer <- Convolutional layer







## 7. Reference

- Yann LeCun, Leon Bottou, Yoshua Bengio, & Patric Haffiner (1998), Gradient-based learning applied to document recognition, Proceedings of the IEEE.
- Sergey Ioffe, Christian Szegedy (2015), Batch Normalization: Accelerating deep network training by reducing internal covariate shift, Proceedings of the ICML.
- Yiliang Xie, Hongyuan Jin, & Eric C.C. Tsang (2017), Improving the lenet with batch normalization and online hard example mining for digits recognition, Proceedings of the ICWAPR.
- 오일석 (2017), 기계 학습, 한빛아카데미
- 오일석 (2008), 패턴인식, 교보문고
- 조준우 (2017), [CNN 역전파를 이해하는 가장 쉬운 방법](https://metamath1.github.io/cnn/index.html)
- Jefkine Kafunah (2016), [Backpropagation In Convolutional Neural Networks](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
- Roei Bahumi (2019), [Deep Learning - Cross Entropy Loss Derivative](http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html)
- Kevin Zakka (2016), [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)





