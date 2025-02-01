import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from images import zodiac_signs

# Путь к исходным данным
data_dir = 'zodiac_data'  # Замените на путь к вашим данным

# Создаем папку для аугментированных данных
os.makedirs('augmented_zodiac_data', exist_ok=True)

# Генератор аугментации
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Загрузка изображений
for sign in zodiac_signs:
    os.makedirs(f'augmented_zodiac_data/{sign}', exist_ok=True)
    sign_dir = os.path.join(data_dir, sign)
    images = os.listdir(sign_dir)
    for img_name in images:
        img_path = os.path.join(sign_dir, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        # Генерация аугментированных изображений
        i = 0
        for batch in datagen.flow(img_array, save_to_dir=f'augmented_zodiac_data/{sign}', save_prefix='aug', save_format='jpeg'):
            i += 1
            if i > 10:  # Генерация 10 аугментированных изображений для каждого оригинала
                break

print("Аугментированные данные созданы!")