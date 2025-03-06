import configparser
import csv

from datetime import datetime

import numpy as np


def load_monk_configuration(dataset):
    dataset_conf = parse_conf_file(f'./configs/config_{dataset}.ini', section_name="Training Setting")
    general_conf = parse_conf_file(f'./configs/config_general.ini', section_name="General Setting")
    return {**general_conf, **dataset_conf}


def load_cup_configuration():
    dataset_conf = parse_conf_file(f'./configs/config_cup.ini', section_name="Training Setting")
    return dataset_conf


def _ignore_comments_and_empty_lines(string):
    return string.split("#", 1)[0].strip()


def parse_conf_file(file_path, section_name):
    config = configparser.ConfigParser()
    config.read(file_path)

    settings = {}

    for key, value in config.items(section_name):
        value = _ignore_comments_and_empty_lines(value)
        if value.isdigit():
            settings[key] = int(value)

    for key, value in config.items(section_name):
        value = _ignore_comments_and_empty_lines(value)
        try:
            settings[key]
        except KeyError:
            try:
                settings[key] = float(value)
            except ValueError:
                pass

    true_values = ['True', 'true', '1']
    false_values = ['False', 'false', '0']

    for key, value in config.items(section_name):
        value = _ignore_comments_and_empty_lines(value)
        if value.lower() in true_values:
            settings[key] = True
        elif value.lower() in false_values:
            settings[key] = False

    for key, value in config.items(section_name):
        value = _ignore_comments_and_empty_lines(value)
        if key not in settings:
            settings[key] = value

    return settings


def write_to_csv(file_path, team_member1, team_member2, team_name, predictions):
    with open(file_path, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["# " + team_member1, team_member2])
        writer.writerow(["# " + team_name])
        writer.writerow(["# ML-CUP23"])
        writer.writerow([f"# Submission date: {datetime.today().strftime('%d/%m/%Y')}"])

        for i in range(len(predictions)):
            writer.writerow(np.insert(predictions[i], 0, i + 1))
