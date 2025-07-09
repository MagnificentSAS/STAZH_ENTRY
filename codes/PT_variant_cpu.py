import torch
import time
import tracemalloc
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#количество батчей
butch_cnt = 64
#количество эпох для сравнения скорости
epoch_list = [10, 20, 30]

print("\n---------------------\n")

#работа с датасетом
dataset = fetch_california_housing(as_frame=True)
print(dataset.DESCR)            #dataset info
features = dataset.data         #признаки
target_ans = dataset.target     #выходы
print("\n---------------------\n")

#разделение на датасета на треинировочную и тестовую часть
F_train, F_test, t_train, t_test = train_test_split(features, target_ans, test_size = 0.2, random_state = 42)

#нормализация данных
my_scaler = StandardScaler()
F_train_sc = my_scaler.fit_transform(F_train)
F_test_sc = my_scaler.transform(F_test)

#преобразование датасета в тензор
F_train_tensor = torch.FloatTensor(F_train_sc)
F_test_tensor = torch.FloatTensor(F_test_sc)
t_train_tensor = torch.FloatTensor(t_train.values)
t_test_tensor = torch.FloatTensor(t_test.values)
dataset_train = TensorDataset(F_train_tensor, t_train_tensor)
dataset_test = TensorDataset(F_test_tensor, t_test_tensor)

#создание загрузчика датасета
dataloader_train = DataLoader(dataset_train, batch_size= butch_cnt, shuffle= True)
dataloader_test = DataLoader(dataset_test, batch_size= butch_cnt, shuffle= True)

#составление модели
class three_layer_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(8,128) #входной слой с 8ю признаками, скрытый 128 нейронов
        self.layer2 = torch.nn.Linear(128,1) #выходной слой с 1 нейроном
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x.squeeze(-1)

#задание функций
criterion = torch.nn.MSELoss()
device = torch.device('cpu')

#функция обучения
def train_model(model, loader, optimizer):
    model.train()
    total_loss = 0 #mse
    total_mae = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += torch.abs(outputs - y_batch).sum().item()

    return total_loss / len(loader), total_mae / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_mae = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            total_loss += criterion(outputs, y_batch).item()
            total_mae += torch.abs(outputs - y_batch).sum().item()

    return total_loss / len(loader), total_mae / len(loader.dataset)

for model in ["compiled", "uncompiled"]:
    print(f"trainig model: {model}")
    print("\n---------------------\n")

    for epochs in epoch_list:

        #сборс и выбор модели для обучения
        if model == "compiled":
            pt_model = torch.compile(three_layer_model().to(device), mode= "max-autotune")
        else:
            pt_model = three_layer_model().to(device)
        optimizer = torch.optim.Adam(pt_model.parameters(), lr= 0.001)

        print("\nWarming up...\n")
        with torch.no_grad():
            for X_batch, y_batch in dataloader_train:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _ = pt_model(X_batch)


        history = { 'loss': [], 'time': [] }
        full_time = time.time()

        tracemalloc.start()

        for epoch in range(epochs):
            epoch_time = time.time()

            train_loss, train_mae = train_model(pt_model, dataloader_train, optimizer)

            epoch_time = time.time() - epoch_time

            #сохранение истории
            history['loss'].append(train_loss)
            history['time'].append(epoch_time)

            print(f"epoch №{epoch + 1} time: {epoch_time:.2f} sec")
            print(f"mae: {train_mae:.4f}\n-")

        #взятие пикового использования памяти
        snapshot = tracemalloc.take_snapshot()
        peak_memory = tracemalloc.get_traced_memory()[1] / 1024**2
        tracemalloc.stop()

        full_time = time.time() - full_time
        print("\n---------------------\n")
        print(f"{epochs} epochs learning time: {full_time:.2f} sec")
        print("\n---------------------\n")

        print(f"peak mem use: {peak_memory:.2f} MB")
        print("\n---------------------\n")

        #оценка точности
        test_loss, test_mae = evaluate(pt_model, dataloader_test)

        print(f"Test loss (mse): {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print("\n---------------------\n")

        #среднее время на эпоху
        print(f"Mean epoch time: {sum(history['time'])/len(history['time']):.2f} sec")

        #граф обучения
        plt.plot(history['loss'], label='Training Loss')
        plt.legend()
        plt.show()
        print("\n---------------------\n")

