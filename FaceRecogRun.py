import numpy as np
from FaceRecogPreload import load_and_preprocess_images,mean_face,eigenfaces, known_face_coefficients, face_labels

def recognize_face(test_image, mean_face, eigenfaces, known_face_coefficients, face_labels):
    test_image_vec = load_and_preprocess_images(test_image)

    projected_test_image = test_image_vec - mean_face
    coefficients = np.dot(projected_test_image, eigenfaces.T)

    distances = np.linalg.norm(coefficients - known_face_coefficients, axis=1)
    min_index = np.argmin(distances)

    recognition_threshold = 0.5
    if distances[min_index] < recognition_threshold:
        return True, face_labels[min_index]
    else:
        return False, None

test_image = "Python/Amrita/math/img.jpg"
recognized, name = recognize_face(test_image, mean_face, eigenfaces, known_face_coefficients, face_labels)

if recognized:
    print(f"Face recognized: {name}")
else:
    print("Face not recognized")
