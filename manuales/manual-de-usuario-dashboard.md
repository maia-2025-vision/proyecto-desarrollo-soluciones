## Detección de Ganado - Manual de Uso de Dashboard

Esta aplicación permite subir fotografías de sobrevuelos de dron para detectar ganado en ellas. Consta de dos secciones:
1. Cargar imágenes
2. Mostrar detecciones

**Nota**: Es importante hacer este proceso de forma secuencial para garantizar su funcionamiento


## 1. Cargar imágenes

La funcionalidad **Cargar Imágenes** permite subir fotografías de sobrevuelos para procesarlas y obtener detecciones de ganado.  


### Paso 1. Acceder a la sección  
Desde la barra lateral izquierda, seleccione la opción **Cargar Imágenes**.  

<img width="990" height="673" alt="Image" src="https://github.com/user-attachments/assets/c61c358e-2fb9-46a0-a147-a36aa9598547" />  

---

### Paso 2. Configurar datos de la carga  
Ingrese el nombre de la **Finca** y la fecha del **Sobrevuelo** en los campos correspondientes.  
Estos datos son obligatorios para continuar.  

<img width="1464" height="673" alt="Image" src="https://github.com/user-attachments/assets/bfcd96d1-1c89-42d1-9605-d62b21562699" />

En Finca, se acepta cualquier cadena de texto

<img width="564" height="237" alt="Image" src="https://github.com/user-attachments/assets/49f9b6ac-4bee-462f-8cbf-3809b36c5d1d" />

En Sobrevuelo se acepta cualquier cadena de texto pero se recomienda usar la fecha del sobrevuelo en formato **AAAA-MM-DD**

<img width="564" height="237" alt="Image" src="https://github.com/user-attachments/assets/f77f3689-6b33-4040-9040-089eabee9b8c" />

---

### Paso 3. Seleccionar archivos  
Una vez completado el paso anterior, se habilita la carga de archivos. Para ello haga clic en **Browse files** o arrastre los archivos de imagen al recuadro indicado.  

**Nota:** Los formatos permitidos son: **PNG, JPG, JPEG, GIF, BMP** con un límite de **200 MB por archivo**.  

<img width="1167" height="457" alt="Image" src="https://github.com/user-attachments/assets/ea03749e-5bc9-4323-9e1a-f38b50a1b353" />

---

### Paso 4. Confirmar la selección  
Verifique que las imágenes aparezcan en la lista de carga.  
Puede eliminar alguna haciendo clic en el ícono de ❌ a la derecha.  

<img width="1167" height="457" alt="Image" src="https://github.com/user-attachments/assets/b3b190d1-a06e-4d97-9e4d-24e0eaca95dd" />
<img width="1167" height="457" alt="Image" src="https://github.com/user-attachments/assets/d1384f1e-bdc7-42c0-bb5d-067e4ae37dd0" />

---

### Paso 5. Subida en proceso  
Espere a que la barra de progreso se complete para cada archivo.  

https://github.com/user-attachments/assets/9df2efe7-92ed-44e9-9eda-939e6b3201eb

---

### Paso 6. Carga y ejecución de detección  
Cuando la carga se complete, se habilitará un botón de **Cargar y ejecutar detección**, al hacer click se comenzarán a procesar las imágenes subidas por lotes de 4 imágenes

<img width="1167" height="262" alt="Image" src="https://github.com/user-attachments/assets/f70eed90-3b7b-40a1-8a67-5cd4c8b39577" />
<img width="1167" height="262" alt="Image" src="https://github.com/user-attachments/assets/8ae9dcbd-6841-456a-beda-668201cdfb03" />
<img width="1167" height="381" alt="Image" src="https://github.com/user-attachments/assets/b3f622f8-7223-4fee-95b5-edf498ee0b67" />

---

### Paso 7. Finalización detección  
Una vez se finaliza la ejecución, se puede observar la confirmación para cada imágen de la ejecución de la detección y el JSON con los bounding boxes.

<img width="1167" height="669" alt="Image" src="https://github.com/user-attachments/assets/451f6aeb-ce9e-4115-9268-91cab289ea29" />
<img width="1167" height="795" alt="Image" src="https://github.com/user-attachments/assets/5b4dd921-257b-4124-8bd2-9d5750610726" />

### Fin carga de imágenes
Con este último paso realizado, se puede pasar a la sección de visualización de imágenes.


## 2. Visualización Detecciones

La funcionalidad **Visualizar Detecciones** permite observar las detecciones de ganado sobre las imágenes que se subieron en la sección 1.  

### Paso 1. Acceder a la sección  
Desde la barra lateral izquierda, seleccione la opción **Visualizar Detecciones**.  

<img width="1494" height="626" alt="Image" src="https://github.com/user-attachments/assets/af643aa5-3d7a-4e0b-ae7f-f46dd466d6b3" />

### Paso 2. Observar detecciones
En la sección principal del sitio se puede hacer scroll para observar las detecciones sobre cada una de las imágenes

<img width="1157" height="626" alt="Image" src="https://github.com/user-attachments/assets/1a72aefc-ff7f-4815-8f64-e839fb81f937" />

https://github.com/user-attachments/assets/19cc5811-a13d-4c88-bc84-bb18440d73bd 

### Paso 3. Observar detecciones para una imagen individual

Para cada una de las imágenes se puede observar el JSON con las coordenadas de cada una de las detecciones. Este JSON puede ser usado en una fase posterior para obtener un resumen del conteo

<img width="1157" height="788" alt="Image" src="https://github.com/user-attachments/assets/1a904719-25b7-476d-8d0d-5455ac866eee" />

<img width="1157" height="788" alt="Image" src="https://github.com/user-attachments/assets/db966665-0e46-4962-9786-9d0917e958df" />


### Paso 4. Filtrar por umbral de confianza
Desde esta opción se puede filtrar las detecciones según el umbral de confianza. En el siguiente ejemplo se ve como al ajustarlo al 85% las predicciones por debajo de este valor dejan de aparecer en la imagen

<img width="259" height="235" alt="Image" src="https://github.com/user-attachments/assets/442a194e-8dd8-46a1-bb8c-df9110e7fef7" />

Al comparar la siguiente imagen con la misma imagen del paso anterior, se puede observar como las detecciones por debajo del nuevo umbral son excluidas de la visualización.

<img width="1163" height="761" alt="Image" src="https://github.com/user-attachments/assets/2f20b710-1002-44a7-868d-0e7e5664834b" />




