from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def draw_conf_matrix(conf_matrix):
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Матрица ошибок')

    tick_marks = np.arange(len(top_publishers))
    plt.xticks(tick_marks, top_publishers)
    plt.yticks(tick_marks, top_publishers)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            text_color = "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
            plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color=text_color)

    plt.xlabel('Предсказано')
    plt.ylabel('Истина')
    plt.show()


if __name__ == '__main__':
    """
    Найти данные для классификации. 
    Предобработать данные, если это необходимо. 
    """
    df = pd.read_csv(Path('vgsales.csv'))
    df.dropna(inplace=True)

    data_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    target_column = 'Publisher'

    top_publishers = df[target_column].value_counts().head(5).index.tolist()
    df = df[df[target_column].isin(top_publishers)]

    print(df.info())

    parameters = df[data_columns]
    target = df[target_column]

    """
    Изобразить гистограмму, которая показывает баланс классов.
    """
    plt.figure(figsize=(8, 8))
    df[target_column].value_counts().plot(kind='bar')
    plt.title('Баланс классов')
    plt.xlabel('Издатель')
    plt.ylabel('Количество игр')
    plt.xticks(rotation=0)
    plt.show()

    """
    Разбить выборку на тренировочную и тестовую. 
    Тренировочная для обучения модели, тестовая для проверки ее качества.
    """
    random_seed = 7
    x_train, x_test, y_train, y_test = train_test_split(parameters,
                                                        target,
                                                        train_size=0.8,
                                                        random_state=random_seed)

    print(f"Размер для признаков обучающей выборки {x_train.shape}")
    print(f"Размер для признаков тестовой выборки {x_test.shape}")
    print(f"Размер для целевого показателя обучающей выборки {y_train.shape}")
    print(f"Размер для показателя тестовой выборки {y_test.shape}")

    """
    Применить алгоритмы классификации: логистическая регрессия, SVM, KNN. 
    Построить матрицу ошибок по результатам работы моделей 
    """

    #логистическая регрессия
    logistic_regression = LogisticRegression(random_state=random_seed)
    logistic_regression.fit(x_train, y_train)

    logistic_predict = logistic_regression.predict(x_test)

    conf_matrix = confusion_matrix(y_test, logistic_predict)

    print(classification_report(y_test, logistic_predict, zero_division=0))
    draw_conf_matrix(conf_matrix)

    #SVM
    params = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid')}
    svc_model = SVC(random_state=random_seed)
    grid_svm = GridSearchCV(estimator=svc_model, param_grid=params)
    grid_svm.fit(x_train, y_train)

    best_svc_model = grid_svm.best_estimator_
    print(f'Лучшее ядро - {best_svc_model.kernel}')
    svm_preds = best_svc_model.predict(x_test)

    conf_matrix = confusion_matrix(y_test, svm_preds)

    print(classification_report(y_test, svm_preds, zero_division=0))
    draw_conf_matrix(conf_matrix)

    #kNN
    neighbors = np.arange(3, 10)
    model_kNN = KNeighborsClassifier()
    params = {'n_neighbors': neighbors}

    grid_knn = GridSearchCV(estimator=model_kNN, param_grid=params)
    grid_knn.fit(x_train, y_train)

    best_knn = grid_knn.best_estimator_
    print(f'Лучшее количество соседей при модели: {best_knn}')

    knn_preds = best_knn.predict(x_test)

    conf_matrix = confusion_matrix(y_test, knn_preds)

    print(classification_report(y_test, knn_preds, zero_division=0))
    draw_conf_matrix(conf_matrix)
