from decouple import config

from backend.web_app import WebApplication

if __name__ == "__main__":
    app = WebApplication(db_name=config('DB_NAME'),
                         db_user=config('DB_USER'),
                         db_password=config('DB_PASSWORD'),
                         db_host=config('DB_HOST'),
                         db_port=config('DB_PORT', cast=int),
                         stream_host=config('STREAM_HOST'),
                         stream_port=config('STREAM_PORT', cast=int),
                         import_name=__name__)
    app.run(config('HOST'), config('PORT', cast=int), config('DEBUG', cast=bool))
