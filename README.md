# MovieLens ETL Process

## ��������

���� ������ ��������� ������� ETL (Extract, Transform, Load) ��� ������ �� ������ MovieLens. ������ ������ ������ �� CSV-������, ��������� ������������� � ���������� ������, � ����� ��������� ���������� � ������� JSON � CSV.

## �����������

������� �������� �� ��� �����:

1. **Extract**: ���������� ������ �� CSV-������.
2. **Transform**: ������������� ������, ������� ���������� ������� �� ����� � ������� ������������� ������ ��� ����������� ������.
3. **Load**: �������� ������ � JSON � CSV �����.

## ����� �� �����

- `ratings.csv`: �������� ������ �� ������� �������.
- `movies.csv`: �������� ���������� � �������.
- `links.csv`: �������� ������ �� ������� ���� ������ ��� ������� ������.

## ����� �� ������

- `ratings_hist.json`: JSON-���� � �������������� ������ ��� ����������� ������ � ���� �������.
- `genre_movies.csv`: CSV-���� � ��������, ���������������� �� �����, � ������ `title`, `imdbId`, � `tmdbId`.

## �������������

### ���������

1. ���������� �����������:
    ```bash
    git clone <URL ������ �����������>
    cd <�������� ����� � ��������>
    ```

2. ���������� ����������� �����������:
    ```bash
    pip install -r requirements.txt
    ```
3. �������� ������ ��� ������� � ���������� ����� � ���������� �������:
    ```bash
    curl -O https://files.grouplens.org/datasets/movielens/ml-32m.zip && unzip ml-32m.zip
    ```

### ������

1. ��������� ������:
    ```bash
    python main.py
    ```

2. ����� ���������� ������� � �������� `output` ����� ������� ����� `ratings_hist.json` � `genre_movies.csv`.

### ���������

� ������� ����� �������� ��������� ���������:

- ���� � ������� ������ (`ratings.csv`, `movies.csv`, `links.csv`).
- ������� ��� ���������� �������� ������ (`output`).
- ID ������ ��� ��������� ������.
- ���� ��� ���������� �������.

## �����������

��� �������� � �������� ETL ���������� � ���� `etl_process.log`. � ������ ������������� ������ ��� ����� ����� �������� � ���� ����.

## ����������

- ��� ������ � PySpark �� Windows ���������� ���������, ��� ���������� `winutils.exe` � ��������� ��������� ���������� ����� `HADOOP_HOME` � `hadoop.home.dir`.
- ���� ����������� � ������� ��� ���������� ����� � ������� CSV ����� PySpark, ����������� Pandas ��� ��������� �������.