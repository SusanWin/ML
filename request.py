import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age':40, 'weight':187})

print(r.json())