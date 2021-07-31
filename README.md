Для работы необходимо:
* Поместить видеофайл video.mp4 в директорию video_processing
* Поместить директорию higher-hrnet-w32 в директорию video_processing/backend/pose_estimator


Ссылка на скачивание higher-hrnet-w32: https://drive.google.com/file/d/1NtQmsiEasUffF_yjPylbo2ad1vkjPFzO/view?usp=sharing
Ссылка на скачивание видеофайла: https://drive.google.com/file/d/1UBTUArWPSeqgROgcoqbhMYylaXBZDke3/view?usp=sharing


Детектор реагирует на наличие двух поднятых рук в кадре, согласно скриншоту ТЗ.


Для локального запуска необходимо заменить все значия host в .env-файлах в папках web_application, video_processing на localhost.
