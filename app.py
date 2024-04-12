from flask import Flask, request, render_template, send_from_directory, url_for, jsonify
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import json
from flask_cors import CORS
from category_encoders import TargetEncoder
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

embedding_model = load_model('embedding_model.h5')
encoder = joblib.load('target_encoder.pkl')
feature_scaler = joblib.load('feature_scaler.pkl')


connections.connect(host="localhost", port="19530")
collection = Collection(name='UCSAS_VECTORS')
embds = np.load("embds.npy")

search_params = {
    "metric_type": "L2", 
    "params": {"nprobe": 10}, 
}
f = open('playerstoinf.json')
playertoind = json.load(f)

app = Flask(__name__)
cors = CORS(app)

@app.route('/query', methods=['POST'])
def handle_query():
    data_ = request.json
    ind = playertoind[data_['name']]
    search_results = collection.search(
      data = [embds[ind].tolist()], 
      anns_field="vector",  
      param=search_params,
      limit=10,  
      expr=None,  
      output_fields=["*"],  
    )
    details = [[y.entity.long_name, y.entity.player_positions, y.entity.overall, y.entity.value_eur, y.entity.age, y.entity.pace, y.entity.shooting, y.entity.passing, y.entity.dribbling, y.entity.defending] for x in search_results for y in x]
    print(details)
    return jsonify({"body": details}), 200

@app.route('/inference', methods=['POST'])
def handle_inference():
    data_ = request.json
    temp = pd.DataFrame(data_, index=[0])
    temp.drop('club_position', axis=1, inplace=True)
    temp['club_position_encoded'] = encoder.transform(pd.DataFrame(data_, index=[0])['club_position']).values[0][0]
    player_vector = embedding_model.predict(feature_scaler.transform(temp)).tolist()
    search_results = collection.search(
      data = player_vector, 
      anns_field="vector",  
      param=search_params,
      limit=10,  
      expr=None,  
      output_fields=["*"],  
    )
    details = [[y.entity.long_name, y.entity.player_positions, y.entity.overall, y.entity.value_eur, y.entity.age, y.entity.pace, y.entity.shooting, y.entity.passing, y.entity.dribbling, y.entity.defending] for x in search_results for y in x]
    print(details)
    return jsonify({"body": details}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)