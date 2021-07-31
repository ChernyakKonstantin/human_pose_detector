import logging
from socketserver import TCPServer

logging.basicConfig(level=logging.DEBUG)


class Server(TCPServer):
    def __init__(self, *args, **kwargs):
        self._response_image: bytes = b''
        super().__init__(*args, **kwargs)

    def get_response_image(self) -> bytes:
        return self._response_image

    def set_response_image(self, image: bytes) -> None:
        self._response_image = image

    def serve_forever(self, *args, **kwargs) -> None:
        logging.debug(f'Server started on {self.server_address}')
        super().serve_forever(*args, **kwargs)
