******딥러닝 응용*********



중간고사 10월 22일 10시 실기고사(코딩)





#1주차
공지
연강하면 12시 20분(12시 10분까지 수업,나머지 10분은 질문타임)
중간고사는 인공지능 기초를 위한 FAQ 내용이 포함된다.

 


수업내용
pre-trained: 사전 훈련


반지도학습(sami-supervise)->비지도학습의 반,지도학습의 반을 섞은거
 
확률론으로 볼떄:
데이터가 정규분포를 따를때     -> 안따를때보다 더 분류를 할수 있다.(머신러닝)
   ''                     ''        안따를때 -> 분포를 만들어 내는 함수를 구할수 있다.(딥러닝)

머신러닝과 딥러닝의 차이점: 픽처(특성)를 추출하냐 안하냐의 차이점이다.
입력->모델->훈련->출력
이때 훈련 할떄 픽처를 추출한다.(딥러낭)
머신러닝은 입력하기전에 추출한다.(머신러닝)

classification,regression 의 차이점: 분류(둘 이상의 특징을 서로 나눔) ,회귀(연속적인 값을 정리)classification을 잘게 나누면 결국 regression이 된다.
ficture가 많을 수록 분류가 잘됨.

차원이 많을수록 weight하고 bias를 뽑기가 힘들다.->과적합(overfitting)
해결방법:픽처의 중요도(ficture selection) 를 계산하다. 규제(mapping)를 한다. 
weight를 0으로 근사하게 설정한다.

- 학습의 뜻:weight하고 bias를 얻어서 계산해 결과를 추출함.
overfitting과 underfitting의 차이점: overfiiting은  많은 데이터(noise와 잘못된 데이터)이 있어 잘못된 결과를 추출 하는 경우 underfitting: 데이터를 너무 적게 뽑아서 생긴 문제

chat gpt의 파라미터의 갯수:1750억

딥러닝
1.활성화 함수:층을 쌓으면 쌓을수록 특징이 안쌓인다. 비선형 함수를 사용해서 한번 꼬아준다.
2.옵티마이져: 경사하강법(최적의 값을 찾아가는 방법)
3.백프로파게이션:
4.포워드프로파게이션:
5.로스트 펑션: 측정값과 예상값의 차이점을 계산하는 함수(cce,bce,mse)
6.원핫 인코딩: 0,1로 변형해서함. 숫자 사이의 관계 떔에 사용됨


경사하강법
W_t+1=W_t-rG

r->running mate(학습률)
G->현재 미분값,기울기 값

층이 많이 쌓을수록 백프로파게이션 할때 기울기가 소실된다.(vanishing gradient,기울기 소실문제)
->rnn,lstm,transformers가 나옴.



#2주차

27p ,모델의 일반화: 훈련데이터와 테스트 데이터하고 나눔

34p, 머신러닝:룰을 발견함
 35p, regresssion과 classification의 차이점
연속적인값     어느 범주에 들어 가냐

37p, 차원 축소가 필요한 이유

41p, masked learning 
일부를 가리고 학습을 시킨다.->예측을 계속하므로 학습력이 증가함


50p 51p, 추세or 분류
bais와 weight를 구한다.

54p
양이 적으면 underfitting->데이터를 늘린다.
noise가 많으면 overfitting->픽처를 줄인다.

57p

59p
규제

60p


교차검증
-k개로 나누고 교차로 train,test를 나누고 검증하고 평균을 낸다.
-데이터의 특성이 각각 다르니깐 fx의 일반화를 높인다.

홀드아웃 
일부를 떼내서 일부를 교차로 검증 

데이터 불일치

96p

99p

101p 원핫 인코딩

110p

127p

하이퍼 파라미터 튜닝: 모델을 입력해줘야 하는 파라미터

confusion matrics
accuracy
recall
f1


accuracy와 f1의 차이점
f1는 imblanced 데이터 땜에 쓰인다. f1낮은데 accuracy가 높으면 imblanced데이터이다.
accuracy:대각선/전체
f1:

decision tree, random forest: 기준이 불순도(gini 계수) 이다. 질문을 통해 불순도 높은곳에서 낮은곳으로 내려감           ex;스무고개 
차이점:  

 boot strapping


pca: 차원을 줄이기 위해서 이다. 차원을 줄이기 위해 주성분 분석을 한다. (기울기)







4주차
컴퓨터 할때 결과가 0 or 1이 나온다. 엄밀성(코드가 왜쓰는지,어떤 역활을 하는지)이 중요하다.
index_col=0 -> 첫번째 항목으로 인덱스번호를 쓰느냐 안쓰냐


분류는 평가기준이 matrix:정확도(accuracy)  ,lose:bse,cse    옵티마이져:
회귀는 평가기준이  matrix:mse  ,loss:mse 

x:숫자
y:숫자도 되고 문자도  된다(머신러닝을 한해서만)


cnn하고 dense  layer의 차이점: 
cnn: local적으로 feature을 추출하지만 dense layer은 전체적으로 feature을 추출한다.


convolution 연산:
pulling 하나 맵핑을 하면 연산이 많아지는걸 방지,정보 축약,이동성 불변의 특징

cnn에서 layer를 거치면서 구체적인 부분을 학습하게된다.
앞에서는 대략적인 걸 추출한다.
pre-trained 학습:이미 계산된 모델을 이용하여 학습을 하게된다.


stride
padding: 필터를 읽을 때 

rnn: 이전의 학습된 정보와 상태를 고려하여 학습을 한다.

lstm:rnn의 문제점의 해결(장기억),vanishing gradient의 일부분을 해결

게이트마다 현재의 입력을 반영을 할것인지,기억을 할것인지(4가지 게이트를 사용한다.)

문제점:계산이 복잡해진다.->GRU가 등장함.

그럼에도 vanishing gradient를 해결 안됨-> 잔여 학습이 등장함.


overfiiting 해결: scaling L1L2

하이퍼파라미터
경사하강법 목적:각 뉴런의 weight하고 layer를 최적화 값을 찾는 방법
확률적,미니배치,배치 

확률적:확률적으로 뽑아서 장점:시간 단축 단점:정확 x
배치: 배치해서 장점:정확 단점 시간이 걸림
미니배치=확률적+배치

지니계수:불순도를 측정
앙상블:분산해서 ->random forest 모델
중복을 허용 할것이냐->boot straping
bagging

pca:차원의 축소 (overfitting을 막기위해)

knn,k-mean의 차이점:knn의 목적:지도학습을 위해 (분류) k-mean:  비지도학습을 위해(군집)





fine-tuning(미세조정)하고 pre-trained의 차이점:
1.전이학습 (Transfer Learning):
기존에 학습된 모델(보통 대규모 데이터셋에서 학습된 모델)을 가져와 새로운 작업에 사용하는 방식입니다.
새로운 작업에서는 기존 모델의 가중치를 그대로 고정하고(trainable = False), 마지막 층(또는 몇몇 층)을 새로운 데이터셋에 맞게 교체하여 학습합니다.
즉, 대부분의 모델 가중치는 고정되어 있고, 새로 추가된 층만 학습합니다.

2.미세조정 (Fine-tuning):
전이학습 이후, 특정 층 또는 전체 모델의 가중치를 일부 또는 전체를 다시 학습시키는 과정입니다.
일반적으로 전이학습으로 처음에 모델을 미리 훈련한 다음, base_model.trainable = True와 같이 모델의 일부 층을 다시 학습 가능한 상태로 만들어 미세조정합니다.
이를 통해 새로운 데이터셋에 더 잘 맞도록 모델을 조정할 수 있습니다.

코드
# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

X, y = list(), list()

for i in range(len(sequence)):

# find the end of this pattern

end_ix = i + n_steps

# check if we are beyond the sequence

if end_ix > len(sequence)-1:

break

# gather input and output parts of the pattern

seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

X.append(seq_x)

y.append(seq_y)

return array(X), array(y)

설명:


분류에 쓸수 없다.(

숙제:전이학습 모델로
아발론 분류 데이터를 pre-trained모델 만들기  순환데이터









5주차 
convulution
필터수=특징 적용

예시
grad cam
개와 고양이의 사진에서 판별할때 개를 분류하면 개 픽셀에 집중함
반면 고양이를 분류할때 고양이 픽셀에 집줌

입력되는 정보가 너무 많다. 그중에 어떤것을 우리가 집중해야 하는가?
->attention(입력되는 데이터로 부터 어떤거를 집중해야 하는지)

attention is all you need(논문) 

어텐션에서 가장 중요한 거: 
문장에서 단어가 가르키는 것은 전체 구조를 봐아야지만 알수 있다.

ex:
10개 문장을 통해 멜론을 추측
->학습 하고 층이 넓어지다가 역전파 과정에서 vanishing gradient 발생
->residual connection 제공함.

multi head connection 
데이터가 훨씬 많아야 한다. 일반 cpu로 안됨 그래서 gpu로 해야함
 

q:단어의 가로   k:단어의 세로     v:값
이 단어에서 얼마나 연관되어 있는지

seq-to-seq 모델(변역기에 많이 사용함)
encoder로 받아서 decoder로 출력
이미지를 받아서  텍스트로 변역(만듬)
텍스트를 받아서  이미지를 생성
압축해서 decoder로 출력

ex:암기를 한다고 하자


encoding
모든 딥러닝에서 문자를 숫자로 넣어야 한다.
postional encoding



autoencoder
-정보의 축약
데이터의 차원 축소와 특징 추출

정상적인 데이터->출력값과 차이 적음
비정상적인 ->차이 큼
정상적인 데이터만 넣어서 학습 모델을 구성
학습된 모델에 (정상+비정상)
정상데이터 :RE 작을 것이고
비정상데이터:RE 클것임


har
layer->ae-> 입력/출력 차이
(reconstruction error)
running-> 

encoding

maniford 가설
few shot learning

DSA 데이터에서 SITTING과 jumping만 남김 sitting은 정상  jumping 비정상만 남김

2.두개 activity를 섞어서 train/test

3.train을 다시 training.valid로 나눈고
4. training/vaild에서 jumping 데이터을 삭제(sitting을 정상데이터로 사용)
5.aE로 valid를 목표로 trainning을 학습
6.sitting에 학습된 모델을 가지고,ae에 넣어서 reconstruction error(입력.출력차이)를 구함
7.적당한 threshold값으로 분류수행
8.








@6주차
gpt가 좋은이유
1.훈련데이터가 계속 쌓임
2.pre-trained model땜에
3.이미 계산된 가중치+bias


이미 잘 정의된 모델이 있다.
->pre-trained모델
파라미터의 수가 많으면 많을수록 성능이 올라간다.
chat gpt:1800억개


오토인코더
목적:차원축소를 이용하여 중요한 특성을 추출하려함
->latent vector로 나타냄(보이지 않는 특징)
인코더를 이용하면 latent vector를 만들수 있다.
decoder가 목적이다.
정상인 데이터와 비정상 데이터를 구분해서 못넣는다.



transformer
특징
1.attention(어느 것이 중요하냐)
입력데이터에서 어떤것을 집중을 해야하냐->attention score
self attention: 자기 내에서 집중을 함(비교)
2.embedding
문자를 벡터(숫자)로 변경해야 한다
3.positional encoding
문장에서 순서(위치)에 따라서 의미가 완전히 달라질수있다.
그래서 위치를 고려하여 적용한다.
4.multihead-attention
문장에서 각 단어마다 attetion score를 구할려면 시간이 걸린다.
그래서 다중으로 해서 score를 구한다.

gradient exploding ->claping
gradient vanishing->

pooling의 
이미지의 불변성
정보의 축약

채널마다 독립적인 필터를 적용
다시 합치는거

cnn:
pixel간의 공간적인 특징을 추출


cbam
채널간의 어텐션을 함.
맵 중요도를 계산을 함

 num_transformer_blocks,->encoder를 얼마나 쓸건가
sparse_categorical_crossentropy->원핫 인코딩 안함


n_classes, activation="softmax")->sigmoid
bse,cse로 변경





@@@@@@7주차@@@@@@@@@@@@@@@@@@@@
cnn와 dense layer의 차이점:
cnn은 local적으로 feature을 추출하지만 dense layer은 전체적으로 feature을 추출한다.

민접한 pixel과 같은 공간적정보를 반영->cnn


cnn에서 resnet이 나온 이유:
gradient vanishing problem(신경망의 길이가 커질수록 역전파 과정에서 기울기가 소실됨.)->그래서 rnn(이전의 상태와 입력을 고려) ,lstm(정기적인 cell 고려)이 나옴.그러나 완벽히 해결이 안됨

resnet->skip connection
장점:gradient vanishing problem 방지, 입력(사전 학습 정보)과의 차이로써 학습력을 높임. 파생되어 나온거 ->inception(서로 다른 fiter를 동시에 적용)

inception ->depth convolution layer(서로 다른 필터를 깊게 쌓는다.)
1x1를 사용하는 이유:차원 축소(병목 층을 담당)

senet->정보를 추출함
기존의 cnn은 채널이 중요도가 동일하게 적용함.senet은 각 채널마다 중요도를 계산하여 다르게 하자.(attention의 개념과 동일)채널 attention.
중요도를 계산하는 방법: average pooling를 사용함.
다음:depthwise seperable convolution layer(채널을 나눠서 다르게 적용하고 다시 모음->xception에서 쓰임

cbam:
채널간의 어텐션을 함.
맵 중요도를 계산을 함
spatial attetion와 channel attetion이 결합됨

transformer:
septosep 모델을 기반. (encoder와 deconder의 형태) ex: 변역기 
 
네가지 기능:
1.self attention
2.embedding-one hot encoding(0,1로 나타내며 상관관계를 표시) 이떄 단점은 데이터의 크기가 길어지면 인코딩이 길어짐.=>벡터화를 함.
3.positional encoding
4.multi head attention


transformer의 실행이 느린 이유:
attention score를 구하는 방식이 복잡함.

transformer가 가지고 있는 
vision transformer->이미지로 input으로 넣음.
이미지를 입력으로 넣으면-> 사진의 각각의 부분을 embedding을 한후 나머지의 과정을 거친다.
transformer는 global feature effect(전체적인 특징을 고려) 부분이 있다.
local picture을 추출함->cnn으로 함.
cnn+transformer->지역적 특징과 글로벌 특징을 동시에 잘 학습
리포트: cnn을 붙이고 transformer를 붙인다.


batch normalization:
배치 정규화는 각 계층의 입력을 표준화하기 때문에 신경망에서 핵심 기술입니다. 
이는 각 계층의 입력 분포가 학습 중에 이동하여 학습 프로세스를 복잡하게 만들고,
효율성을 떨어뜨리는 내부 공변량 이동 문제를 해결합니다

"내부 공변량 변화"는 데이터 분석, 기계 학습 또는 통계학에서 주로 사용하는 용어로, 
데이터 분포의 변화와 관련이 있습니다. 이 개념은 특히 딥러닝에서 자주 언급됩니다.
예를 들어, 신경망에서 각 층은 그 이전 층의 출력값을 입력으로 받아 학습하게 되는데, 
학습 과정에서 이전 층의 가중치가 변화하면 그 출력값(즉, 입력값)의 분포도 변화하게 됩니다. 

이렇게 되면 그 다음 층은 학습을 다시 적응해야 하고, 
결과적으로 학습 속도가 느려지고 안정성도 떨어질 수 있습니다.

이를 해결하기 위한 방법 중 하나가 배치 정규화(Batch Normalization)**입니다. 
배치 정규화는 각 층에서 입력 데이터를 정규화하여 분포 변화를 줄여줌으로써 
내부 공변량 변화를 완화하고 학습을 가속화하는 기술입니다. 
이 과정에서 학습이 더 안정적으로 이루어지며, 학습률을 높일 수 있는 효과도 있습니다.




self-attention
cross-attention
scale-dot-product-attention:          / key값으로 나눠줌.softmax 함수(분류)



key 의 내적의 값이 같아질때 self-attention과 scaled-dot-product attention이 같아짐.


옛날과 요즘 ai의 차이점:
옛날은 데이터(입력)를 formating을 하여 넣어준다.그러면 formating해서 출력이 나옴.
요즘은 텍스트 형태로 넣으면 알아
3서 나온다.(LLM:Large Language Model)
이미지 분야+텍스트 분야에서 쓰임.

 autoencoder:차원축소를 이용하여 중요한 특성을 추출하려함
->latent vector로 나타냄(보이지 않는 특징,차원을 줄이고 줄여서 나타냄)
인코더를 이용하면 latent vector를 만들수 있다.
decoder가 목적이다.
정상인 데이터와 비정상 데이터를 구분해서 못넣는다.

autoencoder를 이용한 비지도학습
1.데이터를 입력 받아서
2.train(normal를 남김)과 test(normal과 abnomal)로 나눠서
3.train에서 label 제거
4.auto encoder로 학습
5.restruction error를 측정

작게나올수록: train데이터와 유사


오토인코더에 cnn 적용 가능하다.
구조:conv-dense-lstm순으로 쌓는다.






1.non-linear autoencoder             normal (lying back+lying) abnormal(standing)

                                                      train(noise 있는거,없는거)  test(noise  추가)
                                                       

3.denoise autoencoder







@@@@9주차@@@@
다이나믹 프로그래밍(동적계획법)


확산모델:VAE,GAN,Diffusion model,stable

확률은 분포로 부터 데이터를 얻는다.
likelihood는 데이터가 주어졌을때 분포(확률)를 찾는다.

mle(Maximum Likelihood Estimation):
주어진 데이터가 특정 분포에서 나왔을 가능성을 최대화하는 파라미터를 추정하는 방법

로스함수에서 bse cse의 차이점:이중 분류와 다중분류의 분포가 다름
베르누이 분포->이중분류,  가우시안->다중분류

primary condition->먼저 조건을 세팅을 함.



van과 gan의 차이점: 



만족할 퀄리티가 안나온다.높이면 오류가 생긴다.->.diffusion model를많이 쓴다.


gan 
단점:판별자의 확률 분포를 0.5를 만들기 어렵다.
        학습 조건을 맞추기가 어렵다.
        (생성자와 판별자)밸런스를 맞추기 어렵다.
        noise로 부터 시작함. 


vae
데이터 y로 부터 확률분포를 찾자.
월래 데이터로 부터 압축된 데이터로 해서 찾자.
범위 내에서 하자. 
정규분포를 가지고 오토인코더를 실행함

latent space:변분적 추론을 함.조금식 범위내에서 움직여서 변형해서 값을 변형함.(vae minst)


latent space:
딥러닝이나 머신 러닝에서 데이터의 숨겨진 패턴이나 특징을 압축하고 표현하는 공간.원래의 고차원 데이터를 저차원으로 압축해 표현한 공간으로, 데이터 간의 유사성을 반영하여 더 간단한 방식으로 정보들을 나타내게 됩니다.




순환 데이터
:split sequence함수
회귀와 분류의 경우는 다르다.

회귀로 만들때 데이터를 묶고 다음을 예측한다.
분류로 쓰일때는  데이터를 묶고 많은 데이터로 선택 

ex) 
10 20 30 40 50 60
부분적으로 그룹핑을 함.
3개씩 묶는다.

장점: 근처 데이터의 연관성을 알수있다. 그래프의 추세를 알수 있다.sliding window 방식과  유사하다)

단점:
즉각적인 반영이 안된다.(딜레이가 있다.)






시험
스케일닷 어텐션
마스크드 어텐션
위치 인코딩
vision attention
문제점:induce bias가 없다.

autoencoder
reconstruct
gan



 @@@10주차@@@@

latent space(vector): 보이지 않지만 데이터에 영향을 주는 잠재 변수. 데이터의 차수를 줄여라.
ex: pca, autoencoder

vector와 space의 차이점: 값,공간의 차이이다.


vae
elbo->reconstruction error+ kl발산을 사용

gan


generator
노이즈 벡터를 받아서 가짜 생성

discriminator
판별함 이때 교묘해 져서 둘다 0.5로 수렴함
로스 함수:bse,활성화 함수: sigmoid




cgan,pxi2pix(seqtoseq와 유사),cyclegan




722,723




smote


전공책
767p
768p 
변형 인코더771p
773p
774p 그림 17-11
776p
779p gan
780p
781p
784p *****
785p
786p
788p
791p
793p


확산모델(diffussion model)
원래 데이터에 노이즈를 조금씩 더해준다. 노이즈를 제거할때 마다 역과정에서 학습한다. 

794p
openai 
796p
802p
17장 까지



다음주 1권




@@@@@11주차@@@@@
train data ,test data를 나누는 이유
:model의 일반화 능력을 높이기 위해서

회기와 분류의 차이점:
회기는 연속적인 값을 예측,분류는 이산적인 값을 예측

자기지도 학습
레이블이 없는 채로 데이터를 입력받아 레이블 생성
=unsupervised learing

일반화에 대한 설명
어떤 데이터 이든지 모델에 다 통한다

과대 적합 과소 적합

mse수식

k-fold 교차검증

교차검증
accrucy

정확도의 단점: 데이터의 불균형을 고려하지 않아서
그래서 f1값을 고려한다.


정밀도와 재현율 수식
roc,auc

하이퍼 파라미터 튜닝:
목적: 모델의 성능을 올리려함

confusion matrix 쓰는 이유:
어느 class가 잘 분류가 되었는지

경사하강법의 정의: 손실함수의 최소값을 찾을려고 한다.최적의 w를 찾을려 한다.

배치 ,확률적,미니배치 경사하강법의 차이점:
배치: 전체를 다한다. ->시간이 오래 걸린다.
확률적:랜덤하게 하나를 뽑아서 구한다.-> 값이 일정하지않느나다.
미니배치: 이둘의 단점을 보완함.


조기종료: 일찍 끝냄

softmax 함수의 역할: 
확률이 1이 되게끔 만드는 함수

svm: 

지니 불순도: 
불순도를 낮춰가면서 

앙상블 


배깅,페이스팅

차원의 저주: 픽처가 많을수록 학습이 어렵다.(차원이 많을수록)

주성분 분속,pca

activation fucntion을 사용하는 이유:
좀 더 복잡한 특징을  추출
그레이디언트 소실,폭주->clapping

전이학습의 목적:이미 잘정의된 모델을 쓰자
미세 조정: 찾는데 걸리는 시간을 줄이자

l1,l2 규제

dropout의 목적


2권
cnn을 쓰는 주요한 목적: 
합성곱의 목적:

풀링의 목적: 차원 축소,불변성을 만들어줌

잔차학습의 목적

inception의 특징

senet: 채널의 중요도를 구함.

depth-wise: 채널을 나눠서 가가 다른 필터를 나눠 이후에 다시 합침.

senet, depth-wise의 목적: 저급 파라미터로 높은 성능을 뽑아내려고


rnn의 장점

lstm: gate 의 역할 
gru의 차이점:

transformer
embedding(숫자로 표현),postion encoding(위치를 고려),multihead attetion
self-attention,scaled dot attention(크기를 고려하지 않도록 크기를 나눔),719p

layer normalization 하는 이유: layer하면서 값이 변해짐 그래서 정규화가 필요함

autoencoder
variable autoencoder



