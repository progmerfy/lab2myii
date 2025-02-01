import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# Параметры
img_width, img_height = 64, 64\  # Размер изображений
batch_size = 32
num_classes = 4  # Количество классов (знаков зодиака)
epochs = 50  # Количество эпох

# Путь к данным
data_dir = 'augmented_zodiac_data'

# Аугментация данных
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Нормализация пикселей
    rotation_range=20,  # Случайный поворот
    width_shift_range=0.2,  # Случайный сдвиг по ширине
    height_shift_range=0.2,  # Случайный сдвиг по высоте
    shear_range=0.2,  # Случайный сдвиг по диагонали
    zoom_range=0.2,  # Случайное увеличение
    horizontal_flip=True,  # Случайное отражение по горизонтали
    fill_mode='nearest',  # Заполнение пикселей при трансформациях
    validation_split=0.2  # Разделение данных на обучение и валидацию
)

# Генератор для обучения
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Данные для обучения
)

# Генератор для валидации
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Данные для валидации
)

# Построение модели CNN
model = models.Sequential([
    # Первый сверточный слой
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    layers.MaxPooling2D((2, 2)),

    # Второй сверточный слой
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Третий сверточный слой
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Четвертый сверточный слой
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Полносвязные слои
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout для предотвращения переобучения
    layers.Dense(num_classes, activation='softmax')  # Выходной слой
])

# Компиляция модели
model.compile(
    optimizer='adam',  # Оптимизатор Adam
    loss='categorical_crossentropy',  # Функция потерь
    metrics=['accuracy']  # Метрика точности
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),  # Ранняя остановка
    ModelCheckpoint('C:/Users/small/PycharmProjects/iilaba2/best_model.keras', monitor='val_loss', save_best_only=True)  # Сохранение лучшей модели
]

# Обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=callbacks
)

# Сохранение модели
model.save('C:/Users/small/PycharmProjects/iilaba2/zodiac_cnn_model.keras')

# Визуализация результатов обучения
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Построение графиков
plot_training_history(history)

# Предсказание на тестовых данных
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Предсказание классов
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Вывод предсказанных классов
print("Предсказанные классы:", predicted_classes)