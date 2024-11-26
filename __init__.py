from flask import Flask, request, jsonify, send_from_directory
import os
import json
from utils import proccess_endpoints

app = Flask(__name__)

@app.route("/api/<model_name>", methods=["POST"])
def login(model_name):
	if request.method == 'POST':
		data = request.get_json()
		proyect_name = request.headers.get("ProyectName")
		methods = request.headers.get("methods")
		response = proccess_endpoints(model_name,proyect_name,methods,data)
		return response

@app.route("/geturls", methods=["GET"])
def model_name():
	if request.method == 'GET':
		with open("routes.json", 'r', encoding='utf-8') as file:
			data = json.load(file)
		return jsonify(data)

@app.route('/')
def serve_react_app():
	return send_from_directory('templates', 'index.html')

@app.route('/<path:path>')
def send_static(path):
	return send_from_directory('templates', path)

if __name__=="__main__":
	app.run(debug=True,port=5173)