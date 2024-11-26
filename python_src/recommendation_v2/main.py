import json
import pandas as pd
import time
from metrics import get_time
from iamodels_v2 import recomendar


def recommendation(config_json_path: str):
    # Leer el archivo de configuración
    with open(config_json_path, 'r', encoding='utf-8') as file:
        config = json.load(file)

    #Ruta del proyecto
    ruta = config["proyect"]
    with open(f"{ruta}/{config_json_path}", 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Usar los valores del archivo JSON
    columna_prompt = config['columna_prompt']
    columnas_outputs = config['columnas_outputs']
    filtro_colaborativo = [(par['columna'], par['valor'])
                           for par in config['filtro_colaborativo']]
    umbral_similitud = config['umbral_similitud']
    # num_recomendaciones = config['num_recomendaciones']
    num_recomendaciones = config["n_token"]   
    data_json_path = f'{ruta}/{config["json_file"]}'
    # Cargar el dataset
    with open(data_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Ajustar los parámetros para la función recomendar
    columnas_prompt = list(columna_prompt.keys())
    prompt = columna_prompt

    # Llamar a la función recomendar y mostrar los resultados
    recomendaciones,score = recomendar(df, columnas_prompt, prompt, filtro_colaborativo,
                                 columnas_outputs, umbral_similitud, num_recomendaciones)
    # print(recomendaciones)

    result_json: str = recomendaciones.to_json(
        force_ascii=False, orient='records')

    return result_json,score


def main():
    init_time = time.perf_counter()

    config_json_path = "config.json"
    
    data_json_path = "python_src/recommendation_3/_temp/data.json"
    
    n_token = 3

    result,_ = recommendation(config_json_path)

    print(result)
    get_time(init_time)