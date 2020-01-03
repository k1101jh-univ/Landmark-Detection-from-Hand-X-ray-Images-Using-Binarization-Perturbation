# 이진화 교란을 이용한 손 엑스레이 영상으로부터 랜드마크 검출

## 코드 수행 과정
- WebProject/code/main/main.py를 연다.
- 해당 파일의 device 변수를 수정하여 코드를 실행할 장치를 설정한다.
- model_folder 변수를 수정하여 생성할 모델의 이름을 설정한다.
- num_epochs 변수를 조정하여 epoch를 설정한다.
- lr 변수를 조정하여 learning rate를 설정한다.
- 코드를 실행한다.  
 코드를 실행할 경우, WebProject/models/Laplace/setup1/'model_folder변수 값' 폴더가 새로
 생성되며 해당 폴더 내에 학습된 모델이 저장된다.
 
### 학습된 모델의 결과 확인
- WebProject/code/visualization/result_visualization.py를 연다.
- 해당 파일의 device 변수를 수정하여 코드를 실행할 장치를 설정한다.
- model_folder 변수를 수정하여 불러올 모델이 저장된 폴더명으로 설정한다.
- model_name 변수를 수정하여 모델이 저장된 폴더 내의 불러올 모델 이름으로 설정한다.
- 코드를 실행한다.

## 주요 실험 결과
- 실험 및 학습은 데이터셋 'Digital Hand Atlas'를 사용하여 이루어졌다.
- 테스트셋에서 실제 랜드마크와 예측한 랜드마크 위치의 거리가 2mm 이하인 비율이 95.76%를 달성하였다.
- 또한 실제 랜드마크와 예측한 랜드마크 거리의 평균과 표준편차가 각각 0.66, ±0.71로 
`이전 논문 <https://doi.org/10.1016/j.media.2019.03.007>`_ (94.99%, 점 사이의 거리 평균: 0.66, ±0.74)보다 뛰어난 성능을 보였다.
