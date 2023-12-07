import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

#확장자명
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


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
