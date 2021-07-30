import socket


class TcpClient:
    """Класс TCP-клиента, опрашивающего сервер для получения результатов сегментации."""

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    def receive(self) -> bytes:
        """Метод получения данных от сервера."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.ip, self.port))
            request = []
            while True:
                packet = s.recv(4096)
                if not packet:
                    break
                request.append(packet)
            package = b"".join(request)
            return package
