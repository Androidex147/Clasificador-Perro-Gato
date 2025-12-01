# Clasificador-Perro-Gato
Es un codigo que con una IA cargada en este permite identificar si un perro o un gato (puede ser por separado o un grupo de imagenes)
Preparar el entorno y dependencias
Asegúrate de tener instaladas las librerías: PIL, numpy y tensorflow/keras (para el modelo). Carga tu modelo preentrenado en una variable llamada model y define nombres_clases como lista de etiquetas (ej: ["Gatos", "Perros"]). Monta Google Drive con drive.mount('/content/drive').​

Organizar las imágenes de prueba
Coloca las imágenes en la carpeta /content/drive/MyDrive/IA Dogs vs Cats/Gatos y perros prueba. Nombra los archivos con 'cat' para gatos o 'dog' para perros (ej: cat.123.jpg). El código ignora archivos sin estas palabras.​

Corregir y ejecutar la función de predicción
Completa la función predecir_imagen: agrega el paréntesis faltante en etiqueta_raw = nombres_clases[indice].strip(). Ejecuta el bucle principal que procesa cada imagen: redimensiona a 224x224, normaliza píxeles a [-1,1] y predice con model.predict.​

Evaluar predicciones y métricas
El script compara la predicción ("gato"/"perro") con la etiqueta esperada del nombre del archivo. Cuenta aciertos, calcula precisión (aciertos/total) y media de probabilidades de aciertos. Registra errores en predicciones_incorrectas.​

Generar y revisar el informe
Al final, imprime el informe con total de predicciones, aciertos, precisión, probabilidad media y lista de errores (archivo, predicción, probabilidad). Usa los resultados para mejorar el modelo, como analizar imágenes mal clasificadas
