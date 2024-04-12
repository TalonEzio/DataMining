import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split

basePath = os.path.dirname(__file__) + "\\Test2\\"

train_count = 5

class EvaluationMetrics:
    def __init__(self, list_acc, list_precision, list_recall, list_f1):
        self.list_acc = np.array(list_acc)
        self.list_precision = np.array(list_precision)
        self.list_recall = np.array(list_recall)
        self.list_f1 = np.array(list_f1)

def evaluate_kmeans_and_print_results(wine_type):
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd

    # Khởi tạo các biến để tính tổng các chỉ số
    total_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    list_acc =[]
    list_precision =[]
    list_recall =[]
    list_f1 =[]
    # Lặp qua ba lượt lặp
    for i in range(train_count):
        train_file = f"{basePath}{wine_type.lower()}_wine_train_{i}.csv"
        test_file = f"{basePath}{wine_type.lower()}_wine_test_{i}.csv"

        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        # Huấn luyện mô hình KMeans cho dữ liệu
        kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(train_data)

        # Dự đoán nhãn cho tập kiểm tra dữ liệu
        pred = kmeans.predict(test_data)

        # Tính các chỉ số đánh giá
        cm = confusion_matrix(test_data['quality'], pred)
        acc = accuracy_score(test_data['quality'], pred)
        precision = precision_score(test_data['quality'], pred, average='macro', zero_division=1)
        recall = recall_score(test_data['quality'], pred, average='macro', zero_division=1)
        f1 = f1_score(test_data['quality'], pred, average='macro')

        list_precision.append(precision)
        list_recall.append(recall)
        list_f1.append(f1)
        list_acc.append(acc)

        # Cập nhật tổng các chỉ số
        total_acc += acc
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        # In kết quả
        print(f"{wine_type} Wine - Confusion matrix for iteration {i}:\n{cm}")
        print(f"{wine_type} Wine - Accuracy for iteration {i}: {acc}")
        print(f"{wine_type} Wine - Precision (Positive Predictive Value) for iteration {i}: {precision}")
        print(f"{wine_type} Wine - Recall (True Positive Rate) for iteration {i}: {recall}")
        print(f"{wine_type} Wine - F1 Score for iteration {i}: {f1}")

    # Tính giá trị trung bình
    avg_acc = total_acc / train_count
    avg_precision = total_precision / train_count
    avg_recall = total_recall / train_count
    avg_f1 = total_f1 / train_count

    # In giá trị trung bình
    print(f"\nAverage performance metrics for {wine_type} Wine:")
    print("Accuracy:", avg_acc)
    print("Precision:", avg_precision)
    print("Recall:", avg_recall)
    print("F1 Score:", avg_f1)

    return EvaluationMetrics(list_acc,list_precision,list_recall,list_f1)
def k_mean_train():
    red = evaluate_kmeans_and_print_results("Red")

    white = evaluate_kmeans_and_print_results("White")
    return red,white
def evaluate_regression(train_data,test_data,wine_type):
    model = LinearRegression()
    model.fit(train_data.drop(columns=['quality']), train_data['quality'])

    # Đánh giá mô hình
    pred = model.predict(test_data.drop(columns=['quality']))
    pred_rounded = [round(p) for p in pred]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_data['quality'], pred_rounded)
    acc = accuracy_score(test_data['quality'], pred_rounded)
    precision = precision_score(test_data['quality'], pred_rounded, average='macro', zero_division=1)
    recall = recall_score(test_data['quality'], pred_rounded, average='macro', zero_division=1)
    f1 = f1_score(test_data['quality'], pred_rounded, average='macro')

    # In kết quả đánh giá
    print(f"{wine_type} Wine - Confusion matrix:\n{cm}")
    print(f"{wine_type} Wine - Accuracy: {acc}")
    print(f"{wine_type} Wine - Precision (Positive Predictive Value): {precision}")
    print(f"{wine_type} Wine - Recall (True Positive Rate): {recall}")
    print(f"{wine_type} Wine - F1 Score: {f1}")

    return acc,precision,recall,f1
def regression_train():

    red_total_acc = 0
    red_total_precision = 0
    red_total_recall = 0
    red_total_f1 = 0

    red_list_acc = []
    red_list_precision = []
    red_list_recall = []
    red_list_f1 = []

    white_total_acc = 0
    white_total_precision = 0
    white_total_recall =0
    white_total_f1 = 0

    white_list_acc = []
    white_list_precision = []
    white_list_recall = []
    white_list_f1 = []

    for i in range(3):
        red_train = f"{basePath}red_wine_train_{i}.csv"
        red_test = f"{basePath}red_wine_test_{i}.csv"
        white_train = f"{basePath}white_wine_train_{i}.csv"
        white_test = f"{basePath}white_wine_test_{i}.csv"

        #Red solve
        red_train_data = pd.read_csv(red_train)
        red_test_data = pd.read_csv(red_test)

        red_acc, red_prediction, red_recall, red_f1 = evaluate_regression(red_train_data, red_test_data, "Red")

        red_total_acc += red_acc
        red_total_precision += red_prediction
        red_total_recall += red_recall
        red_total_f1 += red_f1

        red_list_precision.append(red_prediction)
        red_list_recall.append(red_recall)
        red_list_f1.append(red_f1)
        red_list_acc.append(red_acc)

        #White
        white_train_data = pd.read_csv(white_train)
        white_test_data = pd.read_csv(white_test)
        white_acc, white_prediction, white_recall, white_f1 = evaluate_regression(white_train_data, white_test_data,
                                                                                  "White")

        white_total_acc += white_acc
        white_total_precision += white_prediction
        white_total_recall += white_recall
        white_total_f1 += white_f1

        white_list_precision.append(white_prediction)
        white_list_recall.append(white_recall)
        white_list_f1.append(white_f1)
        white_list_acc.append(white_acc)

    return (
                EvaluationMetrics(red_list_acc, red_list_precision,red_list_recall,red_list_f1),
                EvaluationMetrics(white_list_acc, white_list_precision,white_list_recall,white_list_f1)
            )

def plot_metrics(red_k_mean, red_regression, white_k_mean, white_regression, metric='f1'):
    algorithms = ['Red K-Means', 'Red Linear Regression', 'White K-Means', 'White Linear Regression']
    if metric == 'f1':
        metric_values = [
            red_k_mean.list_f1,
            red_regression.list_f1,
            white_k_mean.list_f1,
            white_regression.list_f1
        ]
        ylabel = 'F1 Score'
    elif metric == 'acc':
        metric_values = [
            red_k_mean.list_acc,
            red_regression.list_acc,
            white_k_mean.list_acc,
            white_regression.list_acc
        ]
        ylabel = 'Accuracy'
    elif metric == 'precision':
        metric_values = [
            red_k_mean.list_precision,
            red_regression.list_precision,
            white_k_mean.list_precision,
            white_regression.list_precision
        ]
        ylabel = 'Precision'
    elif metric == 'recall':
        metric_values = [
            red_k_mean.list_recall,
            red_regression.list_recall,
            white_k_mean.list_recall,
            white_regression.list_recall
        ]
        ylabel = 'Recall'

    colors = ['r', 'g', 'b', 'y']

    plt.figure(figsize=(10, 6))

    for i in range(len(metric_values)):
        plt.plot(metric_values[i], label=algorithms[i], color=colors[i])

    # Đặt tiêu đề và nhãn cho trục
    plt.title(f'{metric.capitalize()} Comparison of Algorithms')
    plt.xlabel('Iterations')
    plt.ylabel(ylabel)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Đọc dữ liệu từ file
    red_wine_data = pd.read_csv(f"{basePath}winequality-red.csv")
    white_wine_data = pd.read_csv(f"{basePath}winequality-white.csv")

    for i in range(train_count):
        red_train, red_test = train_test_split(red_wine_data, test_size=0.1, random_state=i)
        red_train.to_csv(f"{basePath}red_wine_train_{i}.csv", index=False)
        red_test.to_csv(f"{basePath}red_wine_test_{i}.csv", index=False)

        white_train, white_test = train_test_split(white_wine_data, test_size=0.1, random_state=i)
        # Xuất tập huấn luyện và tập kiểm tra cho rượu trắng
        white_train.to_csv(f"{basePath}white_wine_train_{i}.csv", index=False)
        white_test.to_csv(f"{basePath}white_wine_test_{i}.csv", index=False)


    #Task 2
    red_k_mean,white_k_mean = k_mean_train()
    #Task 3
    red_regression,white_regression = regression_train()


    print(red_k_mean.list_f1.shape,white_k_mean.list_f1.shape,red_regression.list_f1.shape,white_regression.list_f1.shape)


    plot_metrics(red_k_mean, red_regression, white_k_mean, white_regression, metric='f1')
    plot_metrics(red_k_mean, red_regression, white_k_mean, white_regression, metric='acc')
    plot_metrics(red_k_mean, red_regression, white_k_mean, white_regression, metric='precision')
    plot_metrics(red_k_mean, red_regression, white_k_mean, white_regression, metric='recall')










