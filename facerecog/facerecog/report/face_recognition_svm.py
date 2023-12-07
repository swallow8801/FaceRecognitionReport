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
