import tensorflow as tf
from src.utils import preprocess_image
import matplotlib.pyplot as plt

def predict_image(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    img_array = preprocess_image(img_path)

    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    if confidence > 0.5:
        label = f"Dog 🐶 ({confidence*100:.2f}%)"
    else:
        label = f"Cat 🐱 ({(1-confidence)*100:.2f}%)"

    # visualize
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')
    plt.show()
