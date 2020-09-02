---
layout: default
title: C++ implementation of LeNet-5
parent: Projects
nav_order: 1
mathjax: true
---

# **LeNet-5의 구현 및 성능 향상**

<br/>

본 글은 2020-1학기 캡스톤디자인 과목에서 수행한 프로젝트를 정리하기 위해 작성한 것입니다. 프로젝트의 주요 내용을 요약하여 작성했고, 구현한 딥러닝 라이브러리는 [여기](https://github.com/stnamjef/SimpleNN)에서 확인할 수 있습니다.

## **1. Overview**

### **(1) Architecture**

![LeNet-5](/img/cnn/LeNet-5_modified.png)

위의 그림은 본 프로젝트에서 최종적으로 구현한 합성신경망의 아키텍처이다. LeNet-5의 구조를 그대로 따르되, 컨볼루션 층과 완전연결 층 뒤에 배치 정규화 층을 추가하였다. 또한 활성화 함수를 하이퍼볼릭탄젠트에서 렐루(ReLU)로 변경하였으며, 출력 층에서는 소프트맥스(softmax)를 사용하였다. 마지막으로 손실 함수는 크로스 엔트로피(cross entropy)로 변경하였다.

### **(2) Problems**

- Vanishing gradient

LeNet-5는 출력 층을 제외한 모든 층에서 활성화 함수로 하이퍼볼릭탄젠트를 이용한다. 하이퍼볼릭탄젠트의 도함수는 입력값이 0에서 멀어질수록 함숫값이 0으로 수렴한다. 이에 따라 그래디언트의 크기가 작아져서 신경망의 학습이 더뎌지는 문제가 발생한다. 현대의 신경망 모델은 이러한 문제를 해결하고자 렐루(ReLU), 소프트맥스(softmax) 등을 활성화 함수로 사용한다. 따라서 LeNet-5의 활성화 함수를 위 두 함수로 변경하고 그에 따른 성능의 차이를 측정한다.

- Covariate shift

LeNet-5가 사용하는 그래디언트 기반 학습 방법은 학습률(learning rate)의 초깃값에 매우 민감하다. 예를 들어 학습률이 지나치게 크면 그래디언트가 폭발하여 학습이 제대로 되지 않는 문제가 발생한다. 또한 신경망의 구조적 특성상 각 층에 입력되는 훈련 집합의 분포가 달라지는 공변량 시프트가 발생한다. 이 역시 신경망의 학습을 저해하는 요인이다. 배치 정규화(batch normalization)은 공변량 시프트 문제를 해결하고 신경망이 학습하는 데 있어서 학습률의 영향을 덜 받도록 하는 기법이다. LeNet-5에 배치 정규화를 적용한 후 그에 따른 성능의 차이를 측정한다.

- Overfitting

신경망을 학습할 때 모델의 복잡도에 비해 데이터가 부족하면, 작은 값이었던 가중치들이 점차 증가하면서 과적합이 발생할 수 있다. 가중치 감쇠(weight decay)는 목적 함수에 규제 항을 두어 가중치의 영향력을 줄이는 식으로 과적합을 억제한다. 규제 항의 종류는 여러 가지이나 본 프로젝트에서는 가장 널리 쓰이는 L2놈을 이용한다. LeNet-5에 L2놈 규제 항을 적용한 후 그에 따른 성능의 차이를 측정한다.

### **(3) Objectives**

- C++로 LeNet-5를 구현하며 stl(standard template library)을 제외한 다른 라이브러리는 사용하지 않는다.
- 구현한 LeNet-5의 분류 정확도(accuracy)가 99% 이상이 되도록 한다. 정확도 측정 시 데이터셋은 MNIST를 이용한다.
- 활성화 함수의 변경, 배치 정규화, 가중치 감쇠를 적용한 모델의 정확도와 기존 모델의 정확도를 비교 분석한다.

## **2. Schedule**

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

## **3. Results(tested on MNIST)**

### **(1) Tanh + MSE(baseline model)**

- Epoch: 20, Batch: 20, Learning rate: 0.5
- Training error: 0.62%, Testing error: 1.01%

### **(2) Tanh + MSE + Regularization**

- Epoch: 20, Batch:20, Learning rate: 0.5, Lambda: 0.001
- Training error: 1.22%, Testing error: 1.33%

### **(3) Tanh + MSE + Batch normalization**

- Epoch: 20, Batch: 20, Learning rate: 0.5
- Training error: 0%, Testing error: 0%

## **4. Conclusion**

- 구현한 모델의 에러율(error rate)이 약 1%로 논문에서 제시한 0.95%에 근접함.
- 활성화 함수를 relu, softmax로 손실 함수를 cross entropy로 변경하였을 때 학습이 빨라지는 것을 확인함.
- 가중치 감쇠, 배치 정규화를 적용했을 때 과적합이 억제되는 것을 확인함.
- 한편 배치 정규화를 적용했을 때 모델 성능이 더 향상될 것이라 예상했으나, 측정 결과 오히려 성능이 떨어진 것을 확인함.

## **5. Examples**

- Construct LeNet-5 with batch normalization

```c++
int n_img_train = 60000;
int n_img_test = 10000;
int n_label = 10;
int img_size = 784;

float* train_X;
float* test_X;
int* train_Y;
int* test_Y;

allocate_memory(train_X, n_img_train * img_size);
allocate_memory(test_X, n_img_test * img_size);
allocate_memory(train_Y, n_img_train);
allocate_memory(test_Y, n_img_test);

ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X);
ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X);
ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

SimpleNN model;
model.add(new Conv2d(6, 5, 2, { 28, 28, 1 }, "uniform"));
model.add(new BatchNorm2d);
model.add(new Activation("tanh"));
model.add(new AvgPool2d(2, 2));
model.add(new Conv2d(16, 5, 0, "uniform"));
model.add(new BatchNorm2d);
model.add(new Activation("tanh"));
model.add(new AvgPool2d(2, 2));
model.add(new Linear(120, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("tanh"));
model.add(new Linear(84, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("tanh"));
model.add(new Linear(10, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("tanh"));

int n_epoch = 30, batch = 32;
float lr = 0.1F, decay = 0.005F;

SGD* optim = new SGD(lr, decay, "MSE");

model.fit(train_X, n_img_train, train_Y, n_label, n_epoch, batch, optim,
		  test_X, n_img_test, test_Y, n_label);

delete_memory(train_X);
delete_memory(test_X);
delete_memory(train_Y);
delete_memory(test_Y);
```

- Construct DNN(500 x 150 x 10) with batch normalization

```c++
int n_img_train = 60000;
int n_img_test = 10000;
int n_label = 10;
int img_size = 784;

float* train_X;
float* test_X;
int* train_Y;
int* test_Y;

allocate_memory(train_X, n_img_train * img_size);
allocate_memory(test_X, n_img_test * img_size);
allocate_memory(train_Y, n_img_train);
allocate_memory(test_Y, n_img_test);

ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X);
ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X);
ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

SimpleNN model;

model.add(new Linear(500, 28 * 28, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("relu"));
model.add(new Linear(150, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("relu"));
model.add(new Linear(10, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("softmax"));

int n_epoch = 30, batch = 32;
float lr = 0.01F, decay = 0;

SGD* optim = new SGD(lr, decay, "cross entropy");

model.fit(train_X, n_img_train, train_Y, n_label, n_epoch, batch, optim,
		  test_X, n_img_test, test_Y, n_label);

delete_memory(train_X);
delete_memory(test_X);
delete_memory(train_Y);
delete_memory(test_Y);
```

## **6. Appendices**

- [Back propagation in CNN](/docs/Machine Learning/2020-08-01-Back propagation in CNN)
- [Back propagation in batch-normalized CNN](/docs/Machine Learning/2020-08-01-Back propagation in batch-normalized CNN)

## **7. Reference**

- Yann LeCun, Leon Bottou, Yoshua Bengio, & Patric Haffiner (1998), Gradient-based learning applied to document recognition, Proceedings of the IEEE.
- Sergey Ioffe, Christian Szegedy (2015), Batch Normalization: Accelerating deep network training by reducing internal covariate shift, Proceedings of the ICML.
- Yiliang Xie, Hongyuan Jin, & Eric C.C. Tsang (2017), Improving the lenet with batch normalization and online hard example mining for digits recognition, Proceedings of the ICWAPR.
- 오일석 (2017), 기계 학습, 한빛아카데미
- 오일석 (2008), 패턴인식, 교보문고
- 조준우 (2017), [CNN 역전파를 이해하는 가장 쉬운 방법](https://metamath1.github.io/cnn/index.html)
- Jefkine Kafunah (2016), [Backpropagation In Convolutional Neural Networks](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
- Roei Bahumi (2019), [Deep Learning - Cross Entropy Loss Derivative](http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html)
- Kevin Zakka (2016), [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)





