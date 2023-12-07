 # 머신러닝 프로젝트
2023.10 - 2023.11 진행된 머신러닝 프로젝트로 개인적으로 공부한 머신러닝 모듈을 사용하고 주력 언어인 Python을 사용해서 프로그램 개발을 했습니다.

## 프로젝트 소개

### 1. 공통 문제
이미지를 주었을 때 해당 이미지의 얼굴을 인식하고 학습하여 새로운 이미지에서 예측 및 분류가 가능한 프로그램 개발

### 2. 문제 구분
##### 2_1. [이미지에서 얼굴을 인식할 수 있는가?](https://github.com/swallow8801/FaceRecognitionReport#1%EB%B2%88-%EB%AC%B8%EC%A0%9C--%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%97%90%EC%84%9C-%EC%96%BC%EA%B5%B4%EC%9D%84-%EC%9D%B8%EC%8B%9D%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8A%94%EA%B0%80)
##### 2_2. [얼굴을 학습시키고 새로운 이미지를 주었을 때 어떤 인물인지 예측 및 판별이 가능한가?](https://github.com/swallow8801/FaceRecognitionReport#2%EB%B2%88-%EB%AC%B8%EC%A0%9C--%EC%96%BC%EA%B5%B4%EC%9D%84-%ED%95%99%EC%8A%B5%EC%8B%9C%ED%82%A4%EA%B3%A0-%EC%83%88%EB%A1%9C%EC%9A%B4-%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A5%BC-%EC%A3%BC%EC%97%88%EC%9D%84-%EB%95%8C-%EC%96%B4%EB%96%A4-%EC%9D%B8%EB%AC%BC%EC%9D%B8%EC%A7%80-%EC%98%88%EC%B8%A1-%EB%B0%8F-%ED%8C%90%EB%B3%84%EC%9D%B4-%EA%B0%80%EB%8A%A5%ED%95%9C%EA%B0%80)
##### 2_3. [예측 및 판별이 가능하다면 그 정확성을 향상시키는 방법을 알아보자.](https://github.com/swallow8801/FaceRecognitionReport#3%EB%B2%88-%EB%AC%B8%EC%A0%9C--%EC%98%88%EC%B8%A1-%EB%B0%8F-%ED%8C%90%EB%B3%84%EC%9D%B4-%EA%B0%80%EB%8A%A5%ED%95%98%EB%8B%A4%EB%A9%B4-%EA%B7%B8-%EC%A0%95%ED%99%95%EC%84%B1%EC%9D%84-%ED%96%A5%EC%83%81%EC%8B%9C%ED%82%A4%EB%8A%94-%EB%B0%A9%EB%B2%95%EC%9D%84-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90)

### 3. 실습 환경
##### Tools : Python IDLE 
##### Library : Pillow , Dlib , OpenCV , ScikitLearn - SVM,KNN,Logistic Regression Module


### 4. 결과 확인
##### [결과 확인](https://github.com/swallow8801/FaceRecognitionReport/blob/main/README.md#%EA%B2%B0%EA%B3%BC-%ED%99%95%EC%9D%B8-1)




## 1번 문제 : 이미지에서 얼굴을 인식할 수 있는가?

### Python Code
```python
import cv2

def detect_faces(image_path):
    image = cv2.imread(image_path) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #이미지 흑백 전환

    # HaarFeature를 이용한 CascadeClassifier 호출
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 이미지에서 얼굴을 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 검출된 얼굴에 사각형을 그림
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 출력
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'wonyoung1.jpg'
detect_faces(image_path)
```
#### [문제로 돌아가기](https://github.com/swallow8801/FaceRecognitionReport#2-%EB%AC%B8%EC%A0%9C-%EA%B5%AC%EB%B6%84)


## 2번 문제 : 얼굴을 학습시키고 새로운 이미지를 주었을 때 어떤 인물인지 예측 및 판별이 가능한가?

### Python Code

#### Import
```python
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
```

#### Train()
```python
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    얼굴 인식을 위한 KNN Classifier를 학습시키는 함수

    :train_dir: 학습할 데이터의 디렉토리 및 폴더 경로
    :model_save_path: 학습모델을 저장할 위치 경로 
    :n_neighbors: Classifier에 필요할 neighbor의 수를 결정. 없을 경우 자동으로 계산 후 입력.
    :knn_algo: Knn을 지원하는 파일의 기본 구조 "ball_tree" 사용

    :return: 학습 데이터로 학습된 KNN classifier
    """
    X = []
    y = []

    # Train_dir의 인물 폴더를 탐색
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # 인물 폴더 내 인물 사진을 탐색
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # 얼굴 인식이 1개 미만이거나 2개 이상 일 경우 학습 데이터로 사용하지 않는다.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # face_encodings로 학습 데이터에 추가
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # KNN Classifier에서 가중치를 위해 사용할 이웃 수 결정
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # KNN classifier 생성 후 학습
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # 학습된 KNN classifier 저장
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf
```

#### predict

```python
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    학습된 KNN classifier 로 이미지를 인식하고 예측하는 함수

    :X_img_path: 인식할 이미지 경로
    :knn_clf: Knn Classifier
    :model_path: 저장된 Knn Classifier의 경로
    :distance_threshold: 얼굴 분류를 위한 거리 임계값
            값이 커질 수록 과적합이 일어날 가능성이 더 커진다.

    :return: 이름과 얼굴 위치의 좌표를 List로 반환 : [(name, bounding box), ...].
        인식하지 못한 얼굴은 name을 "Unknown" 으로 반환
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # KNN 모델 호출
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 이미지 파일과 얼굴 위치를 저장
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # 이미지에 얼굴이 없을 경우
    if len(X_face_locations) == 0:
        return []

    # KNN 모델로 가장 근접한 얼굴을 탐색
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # 임계값 내에 없는 클래스 예측 및 분류 제거
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
```


#### show_prediction_labels_on_images
```python
def show_prediction_labels_on_image(img_path, predictions):
    """
    얼굴 인식 예측 결과를 이미지로 Draw하는 함수

    :img_path: 인식된 이미지 경로
    :predictions: predict()함수의 결과 ( 즉, 예측 결과 )
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # 예측된 얼굴 주위에 사각형을 그림
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # 예측된 얼굴 아래에 Text영역 사각형을 그림
        draw.rectangle(((left, bottom - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))

        # 텍스트를 작성함. ( name )
        font = ImageFont.load_default()
        draw.text((left,bottom-10),name,font=font)

    del draw

    # 결과를 보여줌
    pil_image.show()
```

#### Main
```python
if __name__ == "__main__":
    # STEP 1: KNN classifier를 학습시킴
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2: 학습된 classifier로 testCase에서 예측 후 결과 화면을 보여줌.
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("Looking for faces in {}".format(image_file))

        # 예측 결과를 predictions 로 도출
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # 결과를 콘솔창에 출력
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # 이미지 위에 결과를 그려서 출력
        show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)
```

#### [문제로 돌아가기](https://github.com/swallow8801/FaceRecognitionReport#2-%EB%AC%B8%EC%A0%9C-%EA%B5%AC%EB%B6%84)


## 3번 문제 : 예측 및 판별이 가능하다면 그 정확성을 향상시키는 방법을 알아보자.

### Python Code

```python
import face_recognition
from PIL import Image, ImageDraw,ImageFont
from sklearn import svm
import os

# SVC Classifier

encodings = []
names = []

# 학습 디렉토리
train_dir = os.listdir('knn_examples/train')

# 학습 디렉토리의 인물 Loop
for person in train_dir:
    pix = os.listdir("knn_examples/train/" + person)

    # 인물 디렉토리의 인물 사진 Loop
    for person_img in pix:
        # 각 이미지 파일에서 얼굴에 대한 얼굴 인코딩 가져오기
        face = face_recognition.load_image_file("knn_examples/train/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        # 학습 이미지에 얼굴이 하나만 있을 때
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            encodings.append(face_enc)
            names.append(person)

        # 그 외는 예외 처리 
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")

# SVC classifier 생성 후 학습
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

# numpy array로 test_image를 불러온다.
test_image = face_recognition.load_image_file('ive3.jpg')

# 기본 HOG 기반 모델을 사용하여 테스트 이미지의 모든 얼굴 탐색
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# 얼굴에 대한 예측 시작
print("Found:")

# 얼굴 주위에 사각형을 그려줄 이미지 복사본 생성
pil_image = Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)

for i in range(no):
    # 얼굴에 대한 예측 결과를 콘솔창에 출력
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print(*name)

    # 예측된 얼굴 주위에 사각형 그리기
    top, right, bottom, left = face_locations[i]
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 255), width=2)
    draw.rectangle(((left, bottom - 10), (right, bottom)), fill=(0, 0, 0), outline=(0, 255, 255))
    font = ImageFont.load_default()
    draw.text((left+10,bottom-10),*name,font=font)

# 결과 이미지 보여주기
pil_image.show()

```


#### [문제로 돌아가기](https://github.com/swallow8801/FaceRecognitionReport#2-%EB%AC%B8%EC%A0%9C-%EA%B5%AC%EB%B6%84)



## 결과 확인
