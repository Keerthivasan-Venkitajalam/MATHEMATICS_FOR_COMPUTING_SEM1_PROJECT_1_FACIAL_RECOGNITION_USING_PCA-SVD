import numpy as np
from sklearn.decomposition import PCA
import cv2
import os

def load_and_preprocess_images(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (150, 150))
    return resized_img.flatten()

def preprocess_dataset(images.zip):
    image_files = os.listdir(images.zip)

    face_images = np.array([load_and_preprocess_images(os.path.join(images_path, image_file)) for image_file in image_files])

    mean_face = np.mean(face_images, axis=0)
    centered_face_images = face_images - mean_face

    pca = PCA(n_components=150)
    pca.fit(centered_face_images)

    eigenfaces = pca.components_

    face_labels = [image_file.split("_")[0] + " " + image_file.split("_")[1] for image_file in image_files]

    known_face_coefficients = np.dot(centered_face_images, eigenfaces.T)

    return mean_face, eigenfaces, face_labels, known_face_coefficients

images_path = "Python/Amrita/math/images"
mean_face, eigenfaces, face_labels, known_face_coefficients = preprocess_dataset(images_path)




