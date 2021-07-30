import logging
from typing import List

import psycopg2

logging.basicConfig(level=logging.DEBUG)


class DBHandler:
    def __init__(self, db_name: str, user: str, password: str, host: str = None, port: int = None):
        self._db_name = db_name
        self._user = user
        self._password = password
        self._host = host
        self._port = port

    def connect(self) -> None:
        if self._host and self._port:
            connection_params = {
                'dbname': self._db_name,
                'user': self._user,
                'password': self._password,
                'host': self._host,
                'port': self._port
            }
        else:
            connection_params = {
                'dbname': self._db_name,
                'user': self._user,
            }
        self._connection = psycopg2.connect(**connection_params)
        logging.debug('Database is connected')

    def disconnect(self) -> None:
        self._connection.close()
        logging.debug('Database is disconnected')

    def insert_image(self, img_data: bytes) -> None:
        sql = 'INSERT INTO images (image) VALUES (%s)'
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (img_data,))
        self._connection.commit()

    def get_last_n_images(self, n: int) -> List[bytes]:
        sql = 'SELECT image FROM images ORDER BY id DESC LIMIT %s'
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (n,))
            img_data = cursor.fetchall()
        return [bytes(img[0]) for img in img_data]
