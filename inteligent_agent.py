'''
    Importacion de Librerias.
'''
import csv
import logicadifusa as logica
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

'''
    Lista datos_entrenamiento: lectura del archivo CSV con los registros de los adolescentes para el entrenamiento.
'''
datos_entrenamiento = []

'''
    Se abre el archivo CSV students.csv para recorrerlo linea a linea y asi, ir llenando la lista datos_entrenamiento
    con los resultados de aplicar la logica difusa a la data.
'''
with open('students.csv') as csv_file:
    # archivo separado por comas (,)
    datos_estudiantes = csv.reader(csv_file, delimiter=',')
    next(datos_estudiantes)
    for line in datos_estudiantes:
        datos_temp = []  # lista auxiliar para guardar los resultados de cada linea
        '''
            Se guardan los resultados de aplicar la logica difusa a datos interrelacionados entre si para luego convertirlos
            en tipo Float y guardarlos en la lista datos_entrenamiento.
        '''
        datos_temp.append(
            float(logica.Coeficiente_Personal_1(line[1], line[2])))
        datos_temp.append(
            float(logica.Coeficiente_Personal_2(line[8], line[9])))
        datos_temp.append(
            float(logica.Coeficiente_Personal_3(line[23], line[25])))
        datos_temp.append(
            float(logica.Coeficiente_Personal_4(line[4], line[5])))
        datos_temp.append(
            float(logica.Coeficiente_Personal_5(line[24], line[28])))
        datos_temp.append(
            float(logica.Coeficiente_Escolar_1(line[17], line[18])))
        datos_temp.append(
            float(logica.Coeficiente_Escolar_2(line[15], line[16])))
        datos_temp.append(
            float(logica.Coeficiente_Escolar_3(line[29], line[32])))
        datos_temp.append(float(line[27]))

        # se guarda la lista auxiliar para formar en datos_entrenamiento una lista de listas
        datos_entrenamiento.append(datos_temp)

'''
    Se convierte en un arreglo de la libreria numpy a la lista de listas datos_entrenamiento para poder procesarlo y convertirlo
    en un data frame de la libreria pandas. Un data frame es un tipo de dato utilizado en inteligencia artificial (data mining, machine 
    learning, big data, etc.) para poder manejar correctamente los datos.
'''
numpy_array = np.array(datos_entrenamiento)
dataframe = pd.DataFrame(numpy_array)

# imprime la informacion del data frame: tipos de datos, numero de posiciones, bytes utilizados, etc.
dataframe.info()

'''
    Normaliza los datos, ya que se tiene que mantener una cercania entre los datos para su correcto procesamiento.
'''
datos_entrenar_norm = (dataframe - dataframe.min()) / \
    (dataframe.max()-dataframe.min())

'''
    Se determina que numero de clusters es optimo para esta data formando un codo de jambu.
'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(datos_entrenar_norm)
    wcss.append(kmeans.inertia_)

'''
    Se grafica el codo de jambu para verificar graficamente el numero de clusters que debe tener el proyecto para la 
    respectiva data.
'''
plt.plot(range(1, 11), wcss)
plt.title("Codo de jambu")
plt.xlabel('# de Clusters')
plt.ylabel('WCSS')
plt.show()

'''
    Al verificar que '3' es el numero optimo de clusters, se procede a crear un objeto KMeans con 3 clusters y una 
    iteracion maxima de 300, se le alimenta con los datos de entrenamiento normalizados.
'''
clustering = KMeans(n_clusters=3, max_iter=300)
clustering.fit(datos_entrenar_norm)

'''
    Se crea una lista con datos de prueba para generar predicciones.
'''
datos_de_prueba = [
    [1.857, 2.662, 2.312, 4.309, 1.498, 0.301, 1.073, 1.971, 2],
    [3.102, 1.333, 4.001, 3.997, 4.893, 2.777, 1.112, 2.043, 1],
    [0.991, 1.935, 3.997, 2.347, 4.716, 1.112, 3.122, 3.651, 2],
    [2.883, 0.924, 1.157, 0.834, 3.393, 1.974, 3.894, 1.916, 3],
    [1.112, 1.645, 3.652, 3.311, 0.418, 1.238, 1.541, 4.657, 0],
    [3.864, 3.292, 3.337, 2.261, 1.477, 0.973, 2.524, 0.927, 1],
    [1.734, 2.722, 2.047, 4.489, 2.038, 1.517, 1.005, 1.355, 3],
    [2.289, 3.295, 0.894, 3.813, 2.128, 0.851, 2.208, 1.738, 1],
    [1.029, 1.162, 1.137, 4.729, 1.419, 0.678, 4.148, 4.777, 2],
    [2.021, 2.676, 2.645, 0.858, 4.721, 1.012, 3.244, 2.421, 0]
]

for dato in datos_de_prueba:
    prediction = clustering.predict([dato])
    print('Prediccion para: ' + str(dato) + ' es: ' + str(prediction) + '.')

'''
    Se agrega un header llamado Kmeans_Clusters en nuestro data frame donde se incrustan los resultados de las 
    predicciones.
'''
dataframe['Kmeans_Clusters'] = clustering.labels_
dataframe.info()

'''
    Se usa un objeto PCA para decomponer el numero de variables de nuestra data a 2. Se le envian los datos
    de entrenamiento normalizados conjunto con las predicciones realizadas con los datos de prueba.
'''
pca = PCA(n_components=2)
pca_datos_entrenamiento = pca.fit_transform(datos_entrenar_norm)
pca_datos_entrenamiento_df = pd.DataFrame(
    data=pca_datos_entrenamiento, columns=['Componente_1', 'Componente_2'])
pca_nombres_datos_entrenamiento = pd.concat(
    [pca_datos_entrenamiento_df, dataframe[['Kmeans_Clusters']]], axis=1)

'''
    Se grafica el resultado de la decomposicion por PCA con los resultados de los grados de riesgo de consumir 
    alcohol en adolescentes.
'''
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Grado de riesgo de consumo de alcohol', fontsize=21)

color_theme = np.array(["green", "blue", "red"])
ax.scatter(x=pca_nombres_datos_entrenamiento.Componente_1, y=pca_nombres_datos_entrenamiento.Componente_2,
           c=color_theme[pca_nombres_datos_entrenamiento.Kmeans_Clusters], s=50)

plt.show()
