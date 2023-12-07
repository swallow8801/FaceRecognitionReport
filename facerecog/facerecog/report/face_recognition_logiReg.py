import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train_logistic_regression(train_dir, model_save_path=None, verbose=False):
    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("이미지 {}는 학습에 적합하지 않습니다.".format(img_path))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    X = np.array(X)
    y = np.array(y)

    # Logistic Regression 모델 훈련
    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    log_reg.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(log_reg, f)

    return log_reg

def predict_logistic_regression(X_img_path, log_reg=None, model_path=None, probability_threshold=0.5):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("잘못된 이미지 경로: {}".format(X_img_path))

    if log_reg is None and model_path is None:
        raise Exception("Logistic Regression 모델을 제공해야 합니다. log_reg 또는 model_path 중 하나를 사용하세요.")

    if log_reg is None:
        with open(model_path, 'rb') as f:
            log_reg = pickle.load(f)

    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Logistic Regression 모델로 예측
    probabilities = log_reg.predict_proba(faces_encodings)

    # 각 얼굴에 대한 예측 결과
    predictions = [(log_reg.classes_[np.argmax(prob)], loc) for prob, loc in zip(probabilities, X_face_locations) if np.max(prob) >= probability_threshold]

    return predictions


def show_logistic_regression_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # 예측된 얼굴 주위에 사각형을 그림
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # 예측된 얼굴 아래에 Text 영역 사각형을 그림
        draw.rectangle(((left, bottom - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))

        # 텍스트를 작성함. ( name )
        font = ImageFont.load_default()
        draw.text((left, bottom - 10), name, font=font)

    del draw

    # 결과를 보여줌
    pil_image.show()


if __name__ == "__main__":
    # STEP 1: Logistic Regression classifier를 학습시킴
    print("Training Logistic Regression classifier...")
    log_reg_classifier = train_logistic_regression("knn_examples/train", model_save_path="trained_log_reg_model.clf", verbose=True)
    print("Training complete!")

    # STEP 2: 학습된 classifier로 testCase에서 예측 후 결과 화면을 보여줌.
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("Looking for faces in {}".format(image_file))

        # 예측 결과를 predictions로 도출
        predictions = predict_logistic_regression(full_file_path, model_path="trained_log_reg_model.clf")

        # 결과를 콘솔창에 출력
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # 이미지 위에 결과를 그려서 출력
        show_logistic_regression_prediction_labels_on_image(full_file_path, predictions)
