Побудувати докер зображення: docker build -t vsr . -f Dockerfile.cpu
Запустити docker run -v $(pwd)/data/:/home/vsr/data -v $(pwd)/weights/:/home/vsr/weights -v $(pwd)/config.yml:/home/vsr/config.yml -it vsr -c config.yml
