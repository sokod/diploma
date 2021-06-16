Побудувати докер зображення: docker build -t vsr . -f Dockerfile.cpu
Запустити docker run -v $(pwd)/data/:/home/vsr/data -v $(pwd)/weights/:/home/vsr/weights -v $(pwd)/config.yml:/home/vsr/config.yml -it vsr -c config.yml

Або інсталяція на системі (для розробників):
pip install --upgrade pip
pip install -e .

В скрипті python:
from VSR import assistant
assistant.run(config_file="config.yml")
