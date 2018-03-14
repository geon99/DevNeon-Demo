TestMaze
========

* 신경망 기반의 강화 학습 엔진 테스트를 위한 기본적인 샘플

 + 지정된 지형 내에서 최적의 이동경로를 강화학습으로 생성한다.
 + 각 위치에서 다음 이동을 (state, action) 으로 정의 하고 Q-learing 알고리즘으로 Q(quality;reward) 값을 얻고 Q-table 을 신경망에 학습 시킨다.
 + (관련 기술 참조: <https://hunkim.github.io/ml/>)


TestBreakout
============

* 신경망 기반의 강화 학습 엔진을 벽돌 께기 게임에 적용

 + 현재의 볼의 위치와 방향에 대한 바(bar)의 이동을 제어한다.
 + 볼의 위치와 방향, 바의 위치를 상태(state)로 정의, 다음 바의 이동을 동작(action)으로 정의,
  TestMaze 샘플과 같은 방식으로 Q-table 을 신경망에 학습 시킨다.


TestConvNet
===========

* 컨볼루션 네트웍을 활용한 이미지 분류

 + 이미지를 10가지 사물로 분류한다.
 + 5만개의 샘플이미지를 런타임 중에 백그라운드에서 로드 하고, 로드와 동시에 이미지 학습 가능
 + cpu를 통한 학습시 쓰레드 수 설정 기능
 + cuda(gpu)를 사용하여 이미지 학습 가능 (cpu 1 스레드 보다 10배 속도 향상)
 + 레이어 모델 편집 툴 구현,
 + 컨볼루션 필터 이미지를 시각화 표현
 + (관련 기술 참조: <https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html>)
