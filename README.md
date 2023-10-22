# pytorch-ssd-implementation

### 공식 논문
REDMON, Joseph, et al. You only look once: Unified, real-time object detection. In: _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2016. p. 779-788.

### 참고자료
	- https://velog.io/@minkyu4506/PyTorch%EB%A1%9C-YOLOv1-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0
	- https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/loss.py
	- https://github.com/motokimura/yolo_v1_pytorch

### 데이터 준비
- python custom_dataloader.py --dataset path/to/your/voc2007/folder
	
- ex) python custom_dataloader.py --dataset ./voc/VOCdevkit/VOC2007

### 학습
	- python train.py

### Inference
- python detect.py --source path/of/image --weights path/of/weights

### 정리
- 이제 슬슬 타 모델의 코드 포맷대로 작성해 보기보단 직접 짜봐야지 란 생각으로 작성해본 코드였음(아예 안보는건 아직까진..). 단순한 아키텍쳐와 빠른 속도를 표방하는 YOLO v1이었던만큼 가능하지 않았나..
- 확실히 inference 속도가 논문에서 나와있던 수치였던 45fps 언저리를 충족시킴. 다만 그만큼의 정확도의 희생도 불가피.
- 클래스 개수, grid 등 하이퍼파라미터가 하드코딩된 상태로 완성되어 약간 아쉬움. 논문에 나와있는 수치 말고도 여러가지 변화시켜가면서 퍼포먼스의 변화를 비교해 보는것도 좋을듯.