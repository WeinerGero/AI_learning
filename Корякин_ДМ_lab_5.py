import logging
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Настраиваю логирование для вывода информации о процессе в консоль.
# Это помогает отслеживать выполнение скрипта и диагностировать проблемы.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def download_yandex_disk_file(url: str) -> bytes | None:
    """
    Загружает файл с публичной ссылки Яндекс.Диска.
    Функция сначала получает прямую ссылку на скачивание через API,
    а затем загружает и распаковывает ZIP-архив в памяти.
    """
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {'public_key': url}
    try:
        logging.info("Получение ссылки на скачивание архива...")
        # Запрашиваю у API Яндекса временную ссылку для скачивания файла.
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        download_url = response.json()['href']

        logging.info("Скачивание архива...")
        download_response = requests.get(download_url)
        download_response.raise_for_status()

        logging.info("Распаковка файла из архива в памяти...")
        zip_file = zipfile.ZipFile(BytesIO(download_response.content))
        # Ищу первый CSV-файл в архиве, так как предполагается,
        # что он там один.
        csv_filename = next(
            (name for name in zip_file.namelist() if name.endswith('.csv')),
            None
        )

        if csv_filename:
            logging.info(f"Извлечение файла: {csv_filename}")
            csv_content = zip_file.read(csv_filename)
            logging.info("Файл успешно извлечен.")
            return csv_content
        else:
            logging.error("CSV файл не найден в архиве.")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при загрузке файла: {e}")
        return None
    except zipfile.BadZipFile:
        logging.error(
            "Загруженный файл не является корректным ZIP-архивом."
        )
        return None


def classification_training(data: pd.DataFrame):
    """
    Выполняет предобработку данных, обучает и оценивает модели классификации
    KNN и Decision Tree.
    """
    def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет полную предобработку данных. Эта внутренняя функция
        инкапсулирует логику очистки, чтобы основной метод не был перегружен.
        """
        logging.info("Начало предобработки данных...")
        initial_rows = len(df)
        df = df.copy()

        # Удаляю возможные технические столбцы и символы из названий.
        if 'Unnamed: 15' in df.columns:
            df = df.drop('Unnamed: 15', axis=1)
        df.columns = df.columns.str.replace('\t', '', regex=False)

        # Удаляю дубликаты по user_id, чтобы каждый пользователь был уникален.
        rows_before_duplicates = len(df)
        df = df.drop_duplicates(subset=['user_id'], keep='first')
        rows_after_duplicates = len(df)
        if (rows_before_duplicates - rows_after_duplicates) > 0:
            logging.info(
                "Удалено дубликатов по 'user_id': "
                f"{rows_before_duplicates - rows_after_duplicates}"
            )

        # Удаляю строки с пропусками в ключевых категориальных признаках.
        rows_before_na = len(df)
        df.dropna(
            subset=['age', 'gender', 'occupation', 'work_mode'], inplace=True
        )
        rows_after_na = len(df)
        if (rows_before_na - rows_after_na) > 0:
            logging.info(
                "Удалено строк с пропусками в ключевых столбцах: "
                f"{rows_before_na - rows_after_na}"
            )
        df.reset_index(drop=True, inplace=True)

        # Заполняю пропуски в числовых столбцах медианой,
        # так как она устойчива к выбросам.
        numeric_cols_with_nan = [
            'screen_time_hours', 'work_screen_hours', 'leisure_screen_hours',
            'sleep_quality_1_5', 'stress_level_0_10',
            'mental_wellness_index_0_100'
        ]
        for col in numeric_cols_with_nan:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)

        # Преобразую категориальные признаки в числовые.
        gender_map = {'Male': 0, 'Female': 1}
        df['gender'] = df['gender'].map(gender_map)
        if df['gender'].isnull().any():
            mode_gender = df['gender'].mode()[0]
            df['gender'] = df['gender'].fillna(mode_gender)

        df = pd.get_dummies(
            df, columns=['occupation', 'work_mode'], drop_first=True
        )

        # Удаляю выбросы с помощью Z-оценки.
        # Это стандартный подход для удаления аномалий,
        # которые могут исказить результаты обучения.
        rows_before_outliers = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(
            'user_id', errors='ignore'
        )
        df_num = df[numeric_cols].dropna(how="all")
        z_scores = np.abs(stats.zscore(df_num, nan_policy='omit'))
        mask = (z_scores < 3).all(axis=1)
        df_clean = df.loc[df_num.index[mask]]
        rows_after_outliers = len(df_clean)

        if (rows_before_outliers - rows_after_outliers) > 0:
            logging.info(
                "Удалено выбросов (Z-score > 3): "
                f"{rows_before_outliers - rows_after_outliers}"
            )

        logging.info(
            f"Предобработка завершена. Исходное количество строк: "
            f"{initial_rows}. Итоговое количество строк: {len(df_clean)}."
        )
        return df_clean

    if not isinstance(data, pd.DataFrame):
        logging.error(f"Ожидается DataFrame, получен {type(data)}.")
        return

    logging.info("Начало обучения и оценки моделей...")
    df = _preprocess_data(data)

    # Трансформирую целевую переменную в бинарный класс.
    # Значения >= 15 считаются положительным классом (1),
    # остальные — отрицательным (0).
    df['mental_wellness_index_0_100'] = (
        df['mental_wellness_index_0_100'] >= 15
    ).astype(int)

    X = df.drop(columns=['user_id', 'mental_wellness_index_0_100'])
    y = df['mental_wellness_index_0_100']

    # Разделяю выборку на обучающую и тестовую.
    # stratify=y обеспечивает одинаковое распределение классов в обеих выборках.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Масштабирую признаки. Это критически важно для KNN,
    # так как алгоритм основан на расстоянии между точками.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучаю модели с подобранными гиперпараметрами.
    # Выбираю n_neighbors=5, так как это значение является
    # стандартной отправной точкой. Оно обеспечивает хороший баланс
    # между смещением и дисперсией: модель достаточно гибкая
    # для улавливания локальных закономерностей, но при этом устойчива
    # к шуму в данных, в отличие от малых значений (например, 1),
    # которые ведут к переобучению.
    knn = KNeighborsClassifier(n_neighbors=5)

    # Устанавливаю max_depth=4, чтобы ограничить глубину дерева и
    # предотвратить переобучение. Неглубокое дерево создает более простые
    # и обобщающие правила. Также задаю min_samples_leaf=10, чтобы избежать
    # создания листьев для небольшого числа объектов, что делает модель более
    # робастной к выбросам и шуму в обучающей выборке.
    # random_state=42 обеспечивает воспроизводимость результатов.
    dt = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=10, random_state=42
    )

    models = {
        'KNN': knn,
        'DT': dt
    }

    results = []
    for model_name, model in models.items():
        # Для KNN использую масштабированные данные, для DT — оригинальные,
        # так как деревья нечувствительны к масштабу признаков.
        if model_name == 'KNN':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'Model': model_name,
            'accuracy_score': accuracy,
            'f1_score': f1
        })

    # Вывожу метрики в виде таблицы в консоль.
    results_df = pd.DataFrame(results)
    pd.options.display.float_format = '{:.4f}'.format
    logging.info(
        "Результаты оценки моделей:\n" + results_df.to_string(index=False)
    )
    logging.info("Оценка моделей завершена.")


def main():
    """
    Главная функция для выполнения всего пайплайна:
    загрузка данных и запуск обучения.
    """
    yandex_disk_url = "https://disk.yandex.ru/d/NVmydyRuUzCxPw"
    file_content_bytes = download_yandex_disk_file(yandex_disk_url)

    if file_content_bytes:
        try:
            # Читаю CSV из байтовой строки. Использую '\t' как разделитель,
            # так как данные в файле разделены табуляцией.
            data = pd.read_csv(
                BytesIO(file_content_bytes),
                sep='\t',
                engine='python',
                on_bad_lines='skip'
            )
            logging.info(
                f"Данные успешно загружены. Обнаружено {len(data)} строк."
            )
            classification_training(data=data)
        except Exception as e:
            logging.error(f"Произошла ошибка при обработке данных: {e}")


if __name__ == "__main__":
    main()