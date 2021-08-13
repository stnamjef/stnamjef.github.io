---
layout: post
title: Improving Feature Pyramid Network with Multi-Scale RoI Pooling
categories: ['project']
---

### Improving Feature Pyramid Network with Multi-Scale RoI Pooling

#### 1. Abstract

Feature Pyramid Network(FPN)는 고수준 특징맵(feature map)과 저수준 특징맵을 함께 사용하여 검출 정확도를 향상시켰다. 일반적으로 저수준 특징맵은 공간(spatial) 정보를 잘 보존해 작은 객체의 검출에 유리하지만 의미(semantic) 정보가 적어 사용되지 않았다. FPN은 고수준 특징맵을 저수준 특징맵에 더하여 부족한 의미 정보를 보완했다. 이에 따라 작은 객체는 저수준 특징맵에서, 큰 객체는 고수준 특징맵에서 검출할 수 있게 되었다. 그러나 저수준 특징맵의 의미 정보를 보완했더라도 각 수준의 특징맵이 갖는 특성이 다르기 때문에 객체의 크기에 따라 단일 수준의 특징맵만 사용하는 방식은 성능 향상을 제한할 수 있다. 본 연구는 저수준 특징맵의 공간 정보와 고수준 특징맵의 의미 정보를 함께 고려하여 객체를 검출하는 방법을 제안한다.  [[source](https://github.com/stnamjef/feature_pyramid_network)]

#### 2. Overall architecture

<p class=img>
    <img src="/public/img/fpn_roi_pooling.png"/>
</p>
**그림 1** FPN, FPN+의 전체 구조. FPN+는 본 연구에서 제안하는 모델이다. FPN은 객체의 크기에 따라 단일수준의 특징맵에서 RoI를 추출하여 객체를 검출한다(그림 오른쪽 상단). FPN+는 세 개 수준의 특징맵에서 RoI를 추출하고 이를 하나로 합하여 객체를 검출한다(그림 오른쪽 하단). 예를 들어, 이미지 내 바이크를 검출할 때 FPN은 6번 특징맵에서만 RoI를 추출하는 반면, FPN+는 6번, 5번, 4번 특징맵에서 RoI를 추출한다.

#### 3. Drawbacks of Feature Pyramid Network

FPN은 top-down 구조를 적용해 저수준 특징맵의 의미 정보를 강화했다(그림 1). 이로 인하여 모든 수준의 특징맵이 비슷한 수준의 의미 정보를 갖는다면, 작은 객체일수록 공간 정보를 잘 보존하는 저수준 특징맵에서 검출하는 것이 논리적으로 맞다. 이를 검증하기 위하여 각 수준의 특징맵 중 하나씩만 사용하여 모델을 학습시키고 이들의 AP를 비교했다. 예를 들어, 아래의 좌측 모델은 5번 합성곱층에서 생성된 특징맵만 사용하여 객체를 검출한다. 우측 모델은 4번 합성곱층에서 생성된 특징맵만 사용한다. 두 수준의 특징맵이 비슷한 의미 정보를 갖는다면, 스케일이 큰 특징맵을 사용하는 우측 모델의 AP가 더 높아야 한다고 가정했다.

<p class=img>
    <img src="/public/img/fpn_experiment_models.png"/>
</p>

**그림 2** 실험에 사용된 모델의 예시. 좌: 5번 특징맵만 사용하는 모델, 우: 4번 특징맵만 사용하는 모델

| Model |    AP    |  AP@0.5  | AP@0.75  |   APs   |   APm    |   APl    |
| :---: | :------: | :------: | :------: | :-----: | :------: | :------: |
|  Lv6  |   30.9   |   63.0   |   25.9   |   0.9   |   12.2   |   36.4   |
|  Lv5  |   35.6   |   68.0   |   33.0   |   4.8   |   20.1   |   39.7   |
|  Lv4  | **37.4** | **68.6** |   36.2   | **9.6** | **22.6** | **40.4** |
|  Lv3  |   36.8   |   65.3   | **36.8** |   8.6   |   19.1   |   40.2   |

**표 1** 특징맵의 수준에 따른 모델의 성능 비교. 각 모델은 자신의 이름에 해당하는 수준의 특징만 사용한다. 모델 학습에는 VOC 07 trainval 데이터, 성능 측정에는 VOC 07 test 데이터가 사용되었다.

실험 결과 특징맵의 수준이 낮아질수록(스케일이 커질수록) AP가 증가하다가 Lv3 모델에서는 감소했다(표 1). 이는 3번 특징맵의 의미 정보가 4번 특징맵에 비해 여전히 부족하다는 것을 의미한다. 또한 Lv4 모델이 객체의 크기에 따라 성능을 평가하는 모든 기준(APs, APm, APl)에서 가장 높은 수치를 기록했다. 즉, 저수준(큰 스케일) 특징맵이 작은 객체를 검출하는 데 항상 유리하거나 고수준(작은 스케일) 특징맵이 큰 객체를 검출하는 데 가장 적합한 것은 아니다. 따라서 FPN과 같이 객체의 크기에 따라 단일 수준의 특징만을 사용해 객체를 검출하는 방법은 성능 향상을 제한할 수 있다.

#### 4. Proposed method

| Model |    AP    |  AP@0.5  | AP@0.75  |   APs    |   APm    |   APl    |
| :---: | :------: | :------: | :------: | :------: | :------: | :------: |
|   2   |   38.2   |   70.0   |   36.7   |   14.2   |   23.6   |   40.4   |
|   3   | **39.0** | **71.1** | **37.8** | **16.2** | **24.6** | **41.2** |
|   4   |   38.6   |   70.5   |   37.6   |   11.6   |   23.9   |   41.2   |


**표 2** 특징맵 수에 따른 모델의 성능 비교. 첫 번째부터 차례대로 특징맵을 2개, 3개, 4개 사용하는 모델이다.

FPN의 한계를 보완하기 위해 multi-scale RoI pooling 방법을 제안한다. 이 방법은 총 세 개 수준의 특징맵에서 RoI를 추출한다(그림 1). FPN은 네 개 수준의 특징맵에서 하나만 특정하여 RoI pooling을 하기 때문에 최대 세 개의 특징맵을 추가적으로 사용할 수 있다. 실험 결과 세 개의 특징맵을 사용할 때 가장 AP가 높았다(표 2). 주목할 만한 것은 특징맵의 수를 세 개에서 네 개로 늘렸을 때 APs가 약 4.5 포인트 감소했다는 점이다. 이는, 특징맵의 수준이 높아질수록 수용장(receptive field)의 크기가 커지기 때문에, 6번 특징맵 내에서 RoI에 해당하는 영역이 불필요한 문맥정보를 포함하여 생긴 결과라고 추론할 수 있다(그림 3).

<p class=img>
    <img src="/public/img/fpn_features_small_object.png"/>
</p>

**그림 3** 작은 객체에 대한 FPN의 특징맵

#### 5. Experimental results

##### 5.1. Detection results

<table>
    <tr>
        <th>Test data</th>
        <th>Train data</th>
        <th>Model</th>
        <th>AP</th>
        <th>AP@0.5</th>
        <th>AP@0.75</th>
        <th>APs</th>
        <th>APm</th>
        <th>APl</th>
    </tr>
    <tr>
        <td rowspan="4">VOC 07 <br> test</td>
        <td rowspan="2">VOC 07 <br> trainval</td>
        <td>FPN</td>
        <td border="3">36.2</td>
        <td>68.9</td>
        <td>33.9</td>
        <td><strong>18.0</strong></td>
        <td>22.9</td>
        <td>37.8</td>
    </tr>
    <tr>
        <td>FPN+</td>
        <td><strong>39.0</strong></td>
        <td><strong>71.1</strong></td>
        <td><strong>37.8</strong></td>
        <td>16.2</td>
        <td><strong>24.6</strong></td>
        <td><strong>41.2</strong></td>
    </tr>
    <tr>
        <td rowspan="2">VOC 07 + 12 <br> trainval</td>
        <td>FPN</td>
        <td>43.6</td>
        <td>75.4</td>
        <td>44.7</td>
        <td><strong>18.3</strong></td>
        <td>28.9</td>
        <td>45.4</td>
    </tr>
    <tr>
        <td>FPN+</td>
        <td><strong>45.3</strong></td>
        <td><strong>76.0</strong></td>
        <td><strong>47.3</strong></td>
        <td>18.0</td>
        <td><strong>30.6</strong></td>
        <td><strong>47.5</strong></td>
    </tr>
    <tr>
        <td rowspan="2">COCO 17 <br> train</td>
        <td rowspan="2">COCO 17 <br> val</td>
        <td>FPN</td>
        <td>22.4</td>
        <td>40.5</td>
        <td>22.9</td>
        <td>10.9</td>
        <td>25.9</td>
        <td>30.0</td>
    </tr>
    <tr>
        <td>FPN+</td>
        <td><strong>23.5</strong></td>
        <td><strong>41.6</strong></td>
        <td><strong>24.0</strong></td>
        <td><strong>11.0</strong></td>
        <td><strong>26.5</strong></td>
        <td><strong>32.5</strong></td>
    </tr>
</table>

**표 3** FPN, FPN+의 성능 측정 결과

VOC 07 데이터로 평가했을 때 FPN+가 APs를 제외한 모든 지표에서 높은 수치를 기록했다. 특히 VOC 07, VOC 07+12 데이터로 학습했을 때 FPN+의 AP@0.75가 각각 3.9, 2.6 포인트 상승했다. 이는 FPN+가 FPN보다 더 정확하게 객체의 위치를 파악한다는 의미이다. COCO 17 데이터로 평가했을 때도 비슷한 결과를 얻었다. 이때는 APs도 FPN+가 FPN보다 더 높은 수치를 기록했다.

##### 5.2. Grad-CAM of FPN and FPN+

<p class=img>
    <img src="/public/img/fpn_gradcam.png"/>
</p>
**그림 4** FPN, FPN+의 Grad-CAM(gradient-weighted class activation map). 1행: FPN, 2행: FPN+.

Grad-CAM은 객체를 검출하는 데 각 특징맵의 어떤 부분이 기여했는지를 보여준다. FPN은 단일 수준의 특징맵만 사용하여 객체를 검출하는 반면, FPN+는 세 개 수준의 특징맵을 사용하여 객체를 검출한다(그림 4).

#### 6. FPN+ detection examples

<p class=img>
    <img src="/public/img/fpn_detection_examples.png"/>
</p>
**그림 5** FPN+의 객체 검출 예시. 모델 학습에는 VOC 07 + 12 trainval 데이터, 테스트에는 VOC 07 test 데이터가 사용되었다. Confidence score threshold는 0.6이다.