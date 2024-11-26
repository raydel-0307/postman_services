import json
import os
import pytest
import random
import main

# Leer el archivo config.json para obtener el número de proyecto
with open('config.json', 'r', encoding='utf-8') as file:
    config_data = json.load(file)
    project = config_data.get('proyect')

config_file = f"{project}/config.json"

config_structure = {
    'json_file': str,
    'n_token': int,
    'columna_prompt': dict,
    'columnas_outputs': list,
    'filtro_colaborativo': list,
    'umbral_similitud': float,
    'num_recomendaciones': int
}

def get_output_structure(config_file):
    with open(config_file, 'r', encoding='utf-8') as file:
        config = json.load(file)
    output_structure = {col: str for col in config['columnas_outputs']}
    return output_structure

output_structure = get_output_structure(config_file)


# Prueba para verificar si el archivo de configuración existe y tiene la estructura correcta
def test_config_json():
    assert os.path.exists(config_file), f"El archivo de configuración {config_file} no existe"
    
    with open(config_file, 'r', encoding='utf-8') as file:
        config = json.load(file)
    
    for key, value_type in config_structure.items():
        assert key in config, f"Clave '{key}' no encontrada en {config_file}"
        assert isinstance(config[key], value_type), f"Clave '{key}' debería ser de tipo {value_type.__name__} en {config_file}"
    
    assert config['n_token'] > 0, f"n_token debería ser mayor que 0 en {config_file}"
    assert 0.0 <= config['umbral_similitud'] <= 1.0, f"umbral_similitud debería estar entre 0.0 y 1.0 en {config_file}"
    assert config['num_recomendaciones'] > 0, f"num_recomendaciones debería ser mayor que 0 en {config_file}"

# Prueba para verificar si el archivo JSON especificado en la configuración existe y tiene la estructura correcta
def test_json_file_content():
    with open(config_file, 'r', encoding='utf-8') as file:
        config = json.load(file)
    
    json_file_path = f"{project}/{config['json_file']}"
    assert os.path.exists(json_file_path), f"The file {config['json_file']} specified in {config_file} does not exist"
    
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    for item in data:
        for key in config['columnas_outputs']:
            assert key in item, f"Key '{key}' not found in data item in {json_file_path}"
        for key in config['columna_prompt']:
            assert key in item, f"Key '{key}' not found in data item in {json_file_path}"
    
    for filter in config['filtro_colaborativo']:
        column = filter['columna']
        value = filter['valor']
        column_exists = all(column in item for item in data)
        value_exists = any(value in item.get(column, []) for item in data)
        assert column_exists, f"Column '{column}' not found in all data items in {json_file_path}"
        assert value_exists, f"Value '{value}' not found in column '{column}' in at least one data item in {json_file_path}"

# Prueba para verificar si la salida del algoritmo es JSON válido y tiene la estructura correcta
def test_algorithm_output():
    result_json,_ = main.recommendation("config.json")
    
    try:
        result = json.loads(result_json)
    except ValueError:
        pytest.fail("La salida no es JSON válido")
    
    assert isinstance(result, list), "La salida debería ser una lista"
    assert len(result) > 0, "Las recomendaciones no deberían estar vacías"
    
    for recommendation in result:
        for key, value_type in output_structure.items():
            assert key in recommendation, f"Clave '{key}' no encontrada en la recomendación"

# Prueba para verificar si la precisión del algoritmo cumple con el umbral esperado
def test_algorithm_precision():
    _, scores = main.recommendation("config.json")
    
    # Calculate precision
    with open(config_file, 'r', encoding='utf-8') as file:
        config = json.load(file)
    umbral_similitud = config["umbral_similitud"]
    
    precision = sum(score > umbral_similitud for score in scores) / len(scores)
    
    # Assert that the precision meets the expected threshold
    expected_precision = 0.8  # Example expected precision
    assert precision >= expected_precision, f"Precision {precision} is below the expected threshold {expected_precision}"

if __name__ == "__main__":
    pytest.main()