from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

# Создаем папку для данных
os.makedirs('zodiac_data', exist_ok=True)

# Список знаков зодиака
zodiac_signs = ['Aries', 'Taurus', 'Gemini', 'Cancer']

# Размер изображений
img_size = (64, 64)

# Шрифт для текста (убедитесь, что шрифт доступен на вашей системе)
font = ImageFont.load_default()

# Генерация изображений
for sign in zodiac_signs:
    os.makedirs(f'zodiac_data/{sign}', exist_ok=True)
    for i in range(100):  # Генерация 100 изображений для каждого знака
        img = Image.new('RGB', img_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Рисуем текст (название знака)
        draw.text((10, 20), sign, font=font, fill=(0, 0, 0))
        # Сохраняем изображение
        img.save(f'zodiac_data/{sign}/{sign}_{i}.png')

print("Синтетические данные созданы!")