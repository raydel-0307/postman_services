import json
import pandas as pd
import time
#from metrics import get_time
from python_src.recommendation_v2 import iamodels_v2

def recommendation_v2_main(ruta,config):
	init_time = time.perf_counter()
	try:
		columna_prompt = config['columna_prompt']
		columnas_outputs = config['columnas_outputs']
		filtro_colaborativo = [(par['columna'], par['valor']) for par in config['filtro_colaborativo']]
		umbral_similitud = config['umbral_similitud']
		num_recomendaciones = config["n_token"]   
		data_json_path = f'{ruta}/{config["json_file"]}'

		with open(data_json_path, 'r', encoding='utf-8') as file:
			data = json.load(file)
		df = pd.DataFrame(data)

		columnas_prompt = list(columna_prompt.keys())
		prompt = columna_prompt

		recomendaciones,score = iamodels_v2.recomendar(df, columnas_prompt, prompt, filtro_colaborativo,
									columnas_outputs, umbral_similitud, num_recomendaciones)

		result_json: str = recomendaciones.to_json(
			force_ascii=False, orient='records')

		#get_time(init_time)

		return result_json

	except Exception as ex:
		return {"Error": str(ex)}