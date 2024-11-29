from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar el modelo
model = load_model('clasificador_imagenes.h5')

# Ruta de la imagen a predecir (verifica que esta ruta sea correcta)
img_path = r"C:\vs code\Clasificador-imagenes\Prueba_imagen\prueba4.jpg"

# Cargar y preparar la imagen
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Realizar la predicción
prediction = model.predict(img_array)

# Interpretar el resultado
if prediction[0] > 0.5:
    print("La imagen es un perro.")
else:
    print("La imagen es un gato.")
