import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# app instance
app = Flask(__name__)

# loading dataset and model
df = pd.read_csv('./cleanned_data.csv')
knnr_model = pickle.load(open('./knnr_model.pkl', 'rb'))

# index page


@app.route('/', methods=['GET', 'POST'])
def index():
    patient_id = sorted(df['Patient ID'].unique())
    agent_id = sorted(df['Agent ID'].unique())
    availability_time = sorted(df['Availabilty time (Patient)'].unique())
    test_name = sorted(df['Test name'].unique())
    # sample = sorted(df['Sample'].unique())
    way_of_storage = sorted(df['Way Of Storage Of Sample'].unique())

    return render_template('index.html', patient_id=patient_id, agent_id=agent_id,
                           avl_time=availability_time, test_name=test_name,
                           way_of_storage=way_of_storage)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    patient_id = request.form.get('patientid')
    agent_id = request.form.get('agentid')
    time_slot = request.form.get('timeslot')
    short_agent_lab = request.form.get('agenttolab')
    short_patient_lab = request.form.get('patienttolab')
    short_patient_agent = request.form.get('patienttoagent')

    avl_time_patient = request.form.get('avltime')
    test_name = request.form.get('testname')
    sample = request.form.get('exampleRadios')
    way_of_storage_of_sample = request.form.get('storage')
    time_for_sample_collectn = request.form.get('samplecolltime')
    time_agent_lab = request.form.get('timeagentlab')

    data = {
        'Patient ID': patient_id,
        'Agent ID': agent_id,
        'Time slot': time_slot,
        'shortest distance Agent-Pathlab(m)': short_agent_lab,
        'shortest distance Patient-Pathlab(m)': short_patient_lab,
        'shortest distance Patient-Agent(m)': short_patient_agent,
        'Availabilty time (Patient)': avl_time_patient,
        'Test name': test_name,
        'Sample': sample,
        'Way Of Storage Of Sample': way_of_storage_of_sample,
        'Time For Sample Collection MM': time_for_sample_collectn,
        'Time Agent-Pathlab M': time_agent_lab}

    features = pd.DataFrame(data, index=[0])
    pred = knnr_model.predict(features)

    return render_template("predict.html", prediction=np.round(pred[0], 2))


if __name__ == "__main__":
    app.run(debug=False)
