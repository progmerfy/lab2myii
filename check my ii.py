import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score

# Загрузка модели
model = load_model('zodiac_sign_model.keras')  # Укажите путь к вашей модели

# Путь к папкам с изображениями
data_dir = 'augmented_zodiac_data'  # Укажите путь к папке с 4 подпапками (по одной для каждого знака зодиака)

# Список классов (знаков зодиака)
class_names = sorted(os.listdir(data_dir))  # Получаем имена папок (классов)

# Количество изображений для проверки из каждой папки
num_samples_per_class = 10  # Можно изменить на 20


# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(img_path, target_size=(64, 64)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Нормализация
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность батча
    return img_array


# Сбор случайных изображений и их меток
true_labels = []
predicted_labels = []

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    image_files = os.listdir(class_dir)

    # Выбираем случайные изображения
    selected_images = random.sample(image_files, min(num_samples_per_class, len(image_files)))

    for img_file in selected_images:
        img_path = os.path.join(class_dir, img_file)

        # Загрузка и предобработка изображения
        img_array = load_and_preprocess_image(img_path)

        # Получение предсказания
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]

        # Сохранение результатов
        true_labels.append(class_name)
        predicted_labels.append(predicted_class_name)

# Вычисление точности
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Точность предсказаний: {accuracy * 100:.2f}%')

# Вывод результатов
for true_label, predicted_label in zip(true_labels, predicted_labels):
    print(f'Реальный класс: {true_label}, Предсказанный класс: {predicted_label}')