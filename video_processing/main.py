from decouple import config

from backend.videoprocessor import VideoProcessor

if __name__ == "__main__":
    video_processor = VideoProcessor(input_video_file=config('INPUT'),
                                     host=config('HOST'),
                                     port=config('PORT', cast=int),
                                     db_name=config('DB_NAME'),
                                     db_user=config('DB_USER'),
                                     db_password=config('DB_PASSWORD'),
                                     db_host=config('DB_HOST'),
                                     db_port=config('DB_PORT', cast=int))
    video_processor.run()
