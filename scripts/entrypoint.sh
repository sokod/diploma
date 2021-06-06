#!/usr/bin/env bash
script_dir="$( cd "$(dirname "$0")" ; pwd -P )"
main_dir="$(dirname $script_dir)"

echo "Привіт. Програму асистента запущено."
echo ""

train_flag=""
prediction_flag=""
default_flag=""
config_file=""

print_usage() {
  printf "Використання:"
  printf "-c : шлях до файлу конфігурації "
  printf "-t : сессія тренування "
  printf "-p : сессія передбачення "
  printf "-d : завантажити значення за замовчуванням із файлу конфігурації "
}

while getopts 'ptdc:' flag; do
  case "${flag}" in
    c) config_file="--config ${OPTARG}" ;;
    t) train_flag="--training" ;;
    p) prediction_flag="--prediction" ;;
    d) default_flag="--default" ;;
    *) print_usage
       exit 1 ;;
  esac
done


python3 $main_dir/VSR/assistant.py $train_flag $prediction_flag $default_flag $config_file
