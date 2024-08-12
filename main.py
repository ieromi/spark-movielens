import json
import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

"""
Настройка логирования
"""
logging.basicConfig(
    filename='./etl_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)


class MovieLensETL:
    """
    Класс для выполнения ETL процесса
    """

    def __init__(self, spark, ratings_path, movies_path, links_path, output_dir):
        """
        Инициализация класса
        :param spark: объект SparkSession
        :param ratings_path: путь к файлу ratings.csv
        :param movies_path: путь к файлу movies.csv
        :param links_path: путь к файлу links.csv
        :param output_dir: путь к каталогу для сохранения результатов
        """
        self.spark = spark
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.links_path = links_path
        self.output_dir = output_dir

    def extract(self):
        """
        Извлечение данных
        :return: три DataFrame: ratings_df, movies_df, links_df
        """
        try:
            logging.info("Начало извлечения данных")
            ratings_df = self.spark.read.csv(self.ratings_path, header=True, inferSchema=True)
            movies_df = self.spark.read.csv(self.movies_path, header=True, inferSchema=True)
            links_df = self.spark.read.csv(self.links_path, header=True, inferSchema=True)
            logging.info("Данные успешно извлечены")
            return ratings_df, movies_df, links_df
        except Exception as e:
            logging.error(f"Ошибка при извлечении данных: {e}")
            raise

    def transform(self, ratings_df, movies_df, links_df, id_film, genre):
        """
        Трансформация данных
        :param ratings_df: DataFrame с оценками
        :param movies_df: DataFrame с фильмами
        :param links_df: DataFrame с ссылками на фильмы
        :param id_film: id фильма
        :param genre: жанр фильма
        :return: два объекта: output_json, genre_movies_df
        """
        try:
            logging.info("Начало трансформации данных")

            # Агрегация данных по фильму с id_film
            film_ratings = ratings_df.filter(col("movieId") == id_film)
            film_hist = film_ratings.groupBy("rating").count().orderBy("rating").collect()
            film_hist_dict = {int(row['rating']): row['count'] for row in film_hist}

            # Подсчет общего количества оценок для всех фильмов
            all_hist = ratings_df.groupBy("rating").count().orderBy("rating").collect()
            all_hist_dict = {int(row['rating']): row['count'] for row in all_hist}

            output_json = {
                "Film_{}".format(id_film): [film_hist_dict.get(i, 0) for i in range(1, 6)],
                "hist_all": [all_hist_dict.get(i, 0) for i in range(1, 6)]
            }

            # Фильтрация фильмов по жанру
            genre_movies_df = movies_df.filter(movies_df.genres.contains(genre)) \
                .join(links_df, "movieId") \
                .select("title", "imdbId", "tmdbId")

            logging.info("Данные успешно трансформированы")
            return output_json, genre_movies_df
        except Exception as e:
            logging.error(f"Ошибка при трансформации данных: {e}")
            raise

    def load(self, output_json, genre_movies_df):
        """
        Загрузка данных
        :param output_json: JSON объект
        :param genre_movies_df: DataFrame с фильмами по жанру
        """
        try:
            logging.info("Начало загрузки данных")

            # Запись JSON файла
            json_output_path = os.path.join(self.output_dir, "ratings_hist.json")
            with open(json_output_path, "w") as json_file:
                json.dump(output_json, json_file, indent=4)
            logging.info(f"JSON файл успешно сохранен: {json_output_path}")

            # Преобразование в Pandas DataFrame и сохранение в CSV
            pandas_df = genre_movies_df.toPandas()

            # Принудительное сохранение строкового типа данных для 'imdbId' и 'tmdbId'
            pandas_df['imdbId'] = pandas_df['imdbId'].apply(lambda x: f'{x:07}')  # Сохранение ведущих нулей

            # Замена NaN на 0 и преобразование float в int для 'tmdbId'
            pandas_df['tmdbId'] = pandas_df['tmdbId'].fillna(0).astype(int)

            csv_output_path = os.path.join(self.output_dir, "genre_movies.csv")

            """
            Должно было быть так, но, при таком методе записи в CSV, вылетает ошибка с HADOOP_PATH,
            поэтому воспользовался костылём с записью в CSV через pandas DataFrame.
            """
            # genre_movies_df.write.option("header", "true") \
            #     .option("sep", ";") \
            #     .mode("overwrite") \
            #     .csv(csv_output_path)

            pandas_df.to_csv(csv_output_path, index=False)
            logging.info(f"CSV файл успешно сохранен: {csv_output_path}")

        except Exception as e:
            logging.error(f"Ошибка при загрузке данных: {e}")
            raise


def main():
    spark = SparkSession.builder \
        .appName("MovieLens Aggregation") \
        .master("local[*]") \
        .getOrCreate()

    etl = MovieLensETL(
        spark=spark,
        ratings_path="./ml-32m/ratings.csv",
        movies_path="./ml-32m/movies.csv",
        links_path="./ml-32m/links.csv",
        output_dir="./output"
    )

    try:
        ratings_df, movies_df, links_df = etl.extract()
        output_json, genre_movies_df = etl.transform(ratings_df, movies_df, links_df, id_film=2011, genre="Children")
        etl.load(output_json, genre_movies_df)
    except Exception as e:
        logging.critical(f"ETL процесс завершился с ошибкой: {e}")
    finally:
        spark.stop()
        logging.info("SparkSession остановлен")


if __name__ == "__main__":
    main()
