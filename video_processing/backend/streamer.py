import logging
import socket
from threading import Thread

from .request_handler import RequestHandler
from .server import Server

logging.basicConfig(level=logging.DEBUG)


class Streamer(Thread):
    """Класс стриминга видео."""

    def __init__(self, host_ip: str = None, port: int = None, *args, **kwargs):

        self._host_name = socket.gethostname()
        if host_ip:
            self._host_ip = host_ip
        else:
            self._host_ip = socket.gethostbyname(self._host_name)
        if port:
            self._port = port
        else:
            self._port = 80
        self._set_current_frame = None
        super().__init__(name='videostream_thread', *args, **kwargs)
        logging.debug('Streamer is ready')

    def run(self):
        with Server((self._host_ip, self._port), RequestHandler) as streaming_server:
            self._set_current_frame = streaming_server.set_response_image
            streaming_server.serve_forever()

    def update(self, frame: bytes) -> None:
        self._set_current_frame(frame)
