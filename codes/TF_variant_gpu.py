import tensorflow as tf
import time
import tracemalloc
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("\n---------------------\n")

print("TF version:", tf.__version__)
print("GPU доступен:", tf.config.list_physical_devices('GPU'))
a = tf.constant([1.0])
print("Тензор размещен на:", a.device)

#количество батчей
butch_cnt = 64
#количество эпох для сравнения скорости
epoch_list = [10, 20, 30]

print("\n---------------------\n")

#работа с датасетом
dataset = fetch_california_housing(as_frame=True)
print(dataset.DESCR)		#dataset info
features = dataset.data 	#признаки
target_ans = dataset.target 	#выходы

#разделение на датасета на треинировочную и тестовую часть
F_train, F_test, t_train, t_test = train_test_split(features, target_ans, test_size= 0.2, random_state= 42)

#нормализация данных
my_scaler = StandardScaler()
F_train_sc = my_scaler.fit_transform(F_train)
F_test_sc = my_scaler.transform(F_test)

#использование DataSet из tensorflow
#создание
train_dataset = tf.data.Dataset.from_tensor_slices((F_train_sc, t_train))
test_dataset = tf.data.Dataset.from_tensor_slices((F_test_sc, t_test))
#настройка
train_dataset = train_dataset.shuffle(1000).batch(butch_cnt).prefetch(tf.data.AUTOTUNE).repeat()
test_dataset = test_dataset.batch(butch_cnt).prefetch(tf.data.AUTOTUNE)

print("\n---------------------\n")

#создание модели
tf_model = tf.keras.Sequential([
    #начальный слой
    tf.keras.layers.InputLayer(shape=(8,)),

    #скрытый слой
    tf.keras.layers.Dense(128, activation= 'relu', name= 'hidden_layer'),

    #выходной слой
    tf.keras.layers.Dense(1, name= 'output_layer')
])

#компияция модели
tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
tf_model.summary()
print("\n---------------------\n")

#задание функций
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()
train_mae_metric = tf.keras.metrics.MeanAbsoluteError()

#функции train_step
def train_step_notff(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = tf_model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)
    gradients = tape.gradient(loss, tf_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, tf_model.trainable_variables))
    return loss

@tf.function
def train_step_tff(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = tf_model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)
    gradients = tape.gradient(loss, tf_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, tf_model.trainable_variables))
    return loss

train_step_arr = [train_step_tff, train_step_notff]

#обучение c @tf.function, затем без
for train_step in train_step_arr:
    print(f"train_step func: {train_step.__name__}")
    print("\n---------------------\n")

    for epochs in epoch_list:

        #обнуление модели перед обучением
        for layer in tf_model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))

        total_time = time.time()

        #создание метрик
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

        history = { 'time': [], 'loss': [] }

        #начало отслеживания количества памяти
        tracemalloc.start()

        #обучение
        for epoch in range(epochs):
            #сброс метрик в начале эпохи
            train_loss.reset_state()
            train_mae.reset_state()

            #обучение модели с сохранением процесса обучения и выпиской времени
            epoch_time = time.time()
            for step, (f_batch, t_batch) in enumerate(train_dataset):
                if step >= len(F_train_sc) // butch_cnt:
                    break
                batch_loss = train_step(f_batch, t_batch)
                train_loss.update_state(batch_loss)
                train_mae.update_state(t_batch, tf_model(f_batch, training=False))
            epoch_time = time.time() - epoch_time

            print(f"epoch №{epoch + 1} time: {epoch_time:.2f} sec")
            print(f"mae: {train_mae.result().numpy():.4f} \n-")

            #сохранение истории
            history['time'].append(epoch_time)
            history['loss'].append(train_loss.result().numpy())

        #взятие пикового использования и остановка отслеживания использования памяти
        snapshot = tracemalloc.take_snapshot()
        peak_memory = tracemalloc.get_traced_memory()[1] / 1024**2
        tracemalloc.stop()

        #полное время обучения
        total_time = time.time() - total_time
        print("\n---------------------\n")
        print(f"{epochs} epochs learning time: {total_time:.2f} sec")
        print("\n---------------------\n")

        print(f"peak mem use: {peak_memory:.2f} MB")
        print("\n---------------------\n")

        #оценка точности обучения модели
        test_loss, test_mae = tf_model.evaluate(test_dataset)
        print(f"Test loss (mse): {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print("\n---------------------\n")

        #среднее время эпохи
        mean_time = 0
        for t in history['time']:
            mean_time += t
        mean_time /= len(history['time'])
        print(f"Mean time for epoch: {mean_time:.2f} sec")

        #граф обучения
        plt.plot(history['loss'], label='Training Loss')
        plt.legend()
        plt.savefig(f"training_loss_plot_TF_{train_step.__name__}_{epochs}.png")  # Сохранение в файл
        print(f"График сохранён как 'training_loss_plot_TF_{epochs}.png'")
        print("\n---------------------\n")
        plt.close()
