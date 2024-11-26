from connector import *

def proccess_endpoints(model_name,proyect_name,methods,data):
	
	if model_name == "recommendation_v2":

		return recommendation_v2_main(f"python_src/{model_name}/{proyect_name}",data)