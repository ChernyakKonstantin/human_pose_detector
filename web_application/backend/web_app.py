import logging

from flask import Flask, Response, render_template

from .database_handler.db_handler import DBHandler
from .tcp_client import TcpClient

logging.basicConfig(level=logging.DEBUG)


class WebApplication(Flask):
    def __init__(self, db_name: str, db_user: str, db_password: str, db_host: str, db_port: int,
                 stream_host: str, stream_port: int,
                 *args, **kwargs):
        self._db_handler = DBHandler(db_name, db_user, db_password, db_host, db_port)
        self._stream_receiver = TcpClient(stream_host, stream_port)
        self._db_handler.connect()
        super().__init__(*args, **kwargs)
        self.route("/")(self._index)
        self.route("/video_feed")(self._video_feed)
        self.route("/db")(self._get_db_images)
        logging.debug('Server is ready')

    def _index(self):
        return render_template('index.html')

    def _get_video_stream(self) -> bytes:
        while True:
            try:
                frame: bytes = self._stream_receiver.receive()
            except:
                frame = b''
            yield b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

    # TODO: Исправить, чтобы возвращался не 1, а 10 элементов
    def _get_db_images(self):
        images = self._db_handler.get_last_n_images(10)
        responses = []
        for image in images:
            response_str = b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'
            responses.append(Response(response_str, mimetype='multipart/x-mixed-replace; boundary=frame'))
        return responses[-1]

    def _video_feed(self):
        return Response(self._get_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
