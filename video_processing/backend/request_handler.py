import logging
from socketserver import BaseRequestHandler

from .server import Server

logging.basicConfig(level=logging.DEBUG)


class RequestHandler(BaseRequestHandler):
    """
    Класс обработчика запросов. Возвращает изображение с сервера.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self) -> None:
        self.server: Server
        image = self.server.get_response_image()
        self.request.sendall(image)
