📘 딥러닝 응용 강의 정리 (1~11주차)
1주차

시험 공지

중간고사: 10월 22일 오전 10시 (실기 코딩 시험)

범위: 인공지능 기초 FAQ 포함

핵심 개념

Pre-trained model: 사전 훈련된 모델 활용

Semi-supervised learning: 지도 + 비지도 혼합

머신러닝 vs 딥러닝 차이

ML: 특징(Feature) 추출 후 입력

DL: 모델 내부에서 특징 추출

Classification vs Regression

Classification: 범주 분류

Regression: 연속값 예측

Overfitting vs Underfitting

Overfitting: 데이터 과다 → 잡음 학습

Underfitting: 데이터 부족 → 학습 불충분

해결법: Feature selection, 규제(L1/L2)

딥러닝 기본 구성요소

활성화 함수: 비선형성 추가

Optimizer: 경사하강법 기반 최적화

Backpropagation / Forward propagation

Loss function: CCE, BCE, MSE

One-hot encoding

2주차

모델 일반화: Train / Test 분리

차원 축소 필요성: 데이터 고차원 → 계산 복잡, Overfitting 위험

Masked learning: 일부 가리고 예측 → 일반화 성능 ↑

교차 검증 (K-fold)

Train/Test 분리 반복 → 평균 성능 평가

데이터 특성 반영 → 일반화 강화

Accuracy vs F1-score

Imbalanced data → Accuracy 높고 F1 낮을 수 있음

따라서 F1-score 중요

4주차

평가 지표

Classification → Accuracy, CCE/BCE

Regression → MSE

CNN vs Dense

CNN: Local feature 추출 (공간 정보 반영)

Dense: Global feature 추출

Convolution 연산

Pooling: 연산량 감소, 정보 축약, 이동 불변성

RNN/LSTM

RNN: 이전 상태 고려

LSTM: Vanishing Gradient 해결 (게이트 구조)

GRU: 계산 단순화

5주차

Attention 개념

데이터 중 중요한 부분 집중

"Attention is all you need"

Seq2Seq 모델

Encoder → Decoder 구조

번역기, 이미지-텍스트 변환

Autoencoder

데이터 축약, Latent vector 추출

정상 데이터 Reconstruction error ↓

비정상 데이터 Reconstruction error ↑

6주차

GPT의 강점

대규모 Pre-trained, 많은 파라미터

Transformer 핵심

Self-attention, Embedding, Positional encoding, Multi-head attention

Batch Normalization

내부 공변량 변화(Internal Covariate Shift) 완화

학습 안정성↑, 속도↑

7주차

CNN 발전 계열

ResNet: Skip connection (Gradient vanishing 방지)

Inception: 다양한 크기 필터 동시 적용

SENet: 채널별 중요도 계산 (Channel Attention)

CBAM: Spatial + Channel attention

Transformer → Vision Transformer

이미지도 Embedding 후 Transformer 적용

CNN(local) + Transformer(global) → 결합 모델

Batch Normalization vs Layer Normalization

BatchNorm: 배치 단위 정규화

LayerNorm: 레이어 단위 정규화

9주차

확률 모델

Likelihood: 데이터 기반 분포 추정

MLE: 가능도 최대화

생성 모델

VAE: 잠재 공간(Latent space)에서 분포 추정

GAN: Generator vs Discriminator

Diffusion model: 노이즈 제거 기반 학습

Split Sequence (순환 데이터)

Sliding window 방식

근처 데이터 연관성 학습

10주차

Latent space

고차원 데이터 → 저차원 특징 공간

PCA, Autoencoder로 차원 축소

VAE Loss (ELBO)

Reconstruction error + KL divergence

GAN 변형

CGAN, CycleGAN, Pix2Pix

Diffusion Model

데이터에 점진적으로 노이즈 추가 → 역과정 학습

11주차

Train/Test 분리 이유: 일반화 성능 확인

평가 지표

Accuracy 한계 → Precision, Recall, F1-score, ROC, AUC

Confusion Matrix → 분류 성능 분석

경사하강법 종류

Batch / Stochastic / Mini-batch

정규화 기법

L1/L2, Dropout, 조기 종료

CNN 목적

합성곱: Local feature 추출

Pooling: 차원 축소, 불변성

RNN, LSTM, GRU 차이

RNN: 단순, 장기 의존성 약함

LSTM: 게이트 구조로 장기 기억 가능

GRU: 간소화된 LSTM

Transformer 핵심 요소

Embedding, Positional Encoding

Self-attention, Multi-head attention

Scaled dot-product attention
