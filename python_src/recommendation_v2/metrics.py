from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time

def get_time(init_time):
    timer = time.perf_counter() - init_time
    print("Tiempo de ejecución:",timer,"seg")
    
def make_metrics(model,X_test,y_test,features=None):

	metrics = {}
	
	y_pred = model.predict(X_test)

	metrics["accuracy"] = accuracy_score(y_test, y_pred)
	metrics["precision"] = precision_score(y_test, y_pred)
	metrics["recall"] = recall_score(y_test, y_pred)
	metrics["f1"] = f1_score(y_test, y_pred)
	metrics["conf_matrix"] = confusion_matrix(y_test, y_pred)
	metrics["class_report"] = classification_report(y_test, y_pred)
	
	return metrics