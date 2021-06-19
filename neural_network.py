from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

"""
w = weight (вес), b = bies (нейрон смещения), i = input (входной слой),
h = hidden (скрытый слой), o = output (выходной слой), l = label (метка, значение)
e.g. w_i_h = wight from input layer to hidden layer

"""

# Эта строка заполняет переменную images 60000 изображениями,
# кажое из которых состоит из 784 значений. shape: (6000, 784) - размерность
# А переменная label заполнена значениями, которые мы ожидаем. shape: (6000, 10) - размерность
images, labels = get_mnist()
# Инициализируем все веса случайным числом от -0.5 до 0.5
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
# Инициализируем все веса нейронов смещения равными 0
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

# Скорость обучения
learn_rate = 0.01
# Кол-во правильно распознанных значений
nr_correct = 0
# Кол-во эпох - сколько раз пройтись по всему датасету
epochs = 3

# Learning/Обучение
for epoch in range(epochs):
    for img, l in zip(images, labels):
        # From vector size of 784 to matrix size of 784 by 1
        img.shape += (1, )
        # From vector size of 10 to matrix size of 10 by 1
        l.shape += (1, )
        # Forward propagation input -> hidden
        # Прямое распространение входной слой -> скрытый слой
        h_pre = b_i_h + w_i_h @ img
        # Нормализация полученного значения с помощью сигмоидной функции активации
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        # Прямое распространение скрытый слой -> выходной слой
        o_pre = b_h_o + w_h_o @ h
        # Нормализация полученного значения с помощью сигмоидной функции активации
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        # Вычисление процента ошибки
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        # Проверка, правильно нс классифицировала ввод
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropogation output -> hidden (cost function derivative)
        # Обратное распространение выходной  слой -> скрытый слой (производная функции затрат)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h* (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch
    # Посмотреть точность текущей эпохи
    print(f"Точность: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show results
# Посмотреть резулбтаты
while True:
    index = int(input("Введите цифру (0 - 59999, -1, чтобы выйти): "))
    if index < 0:
        break
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1, )
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))
    
    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()
