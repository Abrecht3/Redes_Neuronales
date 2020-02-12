# primer MLP en Keras para predecir
from keras.models import Sequential
from keras.layers import Dense
import numpy

# se fijan las semillas aleatorias para la reproducibilidad
numpy.random.seed(7)

#se cargan los datos
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# variables de entrada (X) y salida (Y)
X = dataset[:,0:8]
Y = dataset[:,8]

# se crea el modelo
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# se Compila el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajuste del modelo
model.fit(X, Y, epochs=150, batch_size=10)

# calcula las predicciones
predictions = model.predict(X)

# redondeamos las predicciones
rounded = [round(x[0]) for x in predictions]
print(rounded)