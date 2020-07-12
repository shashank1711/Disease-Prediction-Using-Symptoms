from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

model = load('./Saved Model/naive_bayes_model.joblib')

symptoms_dict = {'itching': 0,
                ' skin_rash': 1,
                ' continuous_sneezing': 2,
                ' shivering': 3,
                ' stomach_pain': 4,
                ' acidity': 5,
                ' vomiting': 6,
                ' indigestion': 7,
                ' muscle_wasting': 8,
                ' patches_in_throat': 9,
                ' fatigue': 10,
                ' weight_loss': 11,
                ' sunken_eyes': 12,
                ' cough': 13,
                ' headache': 14,
                ' chest_pain': 15,
                ' back_pain': 16,
                ' weakness_in_limbs': 17,
                ' chills': 18,
                ' joint_pain': 19,
                ' yellowish_skin': 20,
                ' constipation': 21,
                ' pain_during_bowel_movements': 22,
                ' breathlessness': 23,
                ' cramps': 24,
                ' weight_gain': 25,
                ' mood_swings': 26,
                ' neck_pain': 27,
                ' muscle_weakness': 28,
                ' stiff_neck': 29,
                ' pus_filled_pimples': 30,
                ' burning_micturition': 31,
                ' bladder_discomfort': 32,
                ' high_fever': 33,
                ' nodal_skin_eruptions': 34,
                ' ulcers_on_tongue': 35,
                ' loss_of_appetite': 36,
                ' restlessness': 37,
                ' dehydration': 38,
                ' dizziness': 39,
                ' weakness_of_one_body_side': 40,
                ' lethargy': 41,
                ' nausea': 42,
                ' abdominal_pain': 43,
                ' pain_in_anal_region': 44,
                ' sweating': 45,
                ' bruising': 46,
                ' cold_hands_and_feets': 47,
                ' anxiety': 48,
                ' knee_pain': 49,
                ' swelling_joints': 50,
                ' blackheads': 51,
                ' foul_smell_of urine': 52,
                ' skin_peeling': 53,
                ' blister': 54,
                ' dischromic _patches': 55,
                ' watering_from_eyes': 56,
                ' extra_marital_contacts': 57,
                ' diarrhoea': 58,
                ' loss_of_balance': 59,
                ' blurred_and_distorted_vision': 60,
                ' altered_sensorium': 61,
                ' dark_urine': 62,
                ' swelling_of_stomach': 63,
                ' bloody_stool': 64,
                ' obesity': 65,
                ' hip_joint_pain': 66,
                ' movement_stiffness': 67,
                ' spinning_movements': 68,
                ' scurring': 69,
                ' continuous_feel_of_urine': 70,
                ' silver_like_dusting': 71,
                ' red_sore_around_nose': 72,
                ' spotting_ urination': 73,
                ' passage_of_gases': 74,
                ' irregular_sugar_level': 75,
                ' family_history': 76,
                ' lack_of_concentration': 77,
                ' excessive_hunger': 78,
                ' yellowing_of_eyes': 79,
                ' distention_of_abdomen': 80,
                ' irritation_in_anus': 81,
                ' swollen_legs': 82,
                ' painful_walking': 83,
                ' small_dents_in_nails': 84,
                ' yellow_crust_ooze': 85,
                ' internal_itching': 86,
                ' mucoid_sputum': 87,
                ' history_of_alcohol_consumption': 88,
                ' swollen_blood_vessels': 89,
                ' unsteadiness': 90,
                ' inflammatory_nails': 91,
                ' depression': 92,
                ' fluid_overload': 93,
                ' swelled_lymph_nodes': 94,
                ' malaise': 95,
                ' prominent_veins_on_calf': 96,
                ' puffy_face_and_eyes': 97,
                ' fast_heart_rate': 98,
                ' irritability': 99,
                ' muscle_pain': 100,
                ' mild_fever': 101,
                ' yellow_urine': 102,
                ' phlegm': 103,
                ' enlarged_thyroid': 104,
                ' increased_appetite': 105,
                ' visual_disturbances': 106,
                ' brittle_nails': 107,
                ' drying_and_tingling_lips': 108,
                ' polyuria': 109,
                ' pain_behind_the_eyes': 110,
                ' toxic_look_(typhos)': 111,
                ' throat_irritation': 112,
                ' swollen_extremeties': 113,
                ' slurred_speech': 114,
                ' red_spots_over_body': 115,
                ' belly_pain': 116,
                ' receiving_blood_transfusion': 117,
                ' acute_liver_failure': 118,
                ' redness_of_eyes': 119,
                ' rusty_sputum': 120,
                ' abnormal_menstruation': 121,
                ' receiving_unsterile_injections': 122,
                ' coma': 123,
                ' sinus_pressure': 124,
                ' palpitations': 125,
                ' stomach_bleeding': 126,
                ' runny_nose': 127,
                ' congestion': 128,
                ' blood_in_sputum': 129,
                ' loss_of_smell': 130
                }
symptoms_dict = pd.DataFrame(list(symptoms_dict.items()), columns= ['Symptoms', 'Count'])


@app.route("/")
def getModel():
    return render_template('index.html')



@app.route("/", methods=['GET','POST'])
def predict():
    input_vector = np.zeros(len(symptoms_dict))
    
    
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms_checkbox')
        for i in range(0, len(symptoms)): 
            symptoms[i] = int(symptoms[i]) 
        for symptom in symptoms:
            symp = []
            symp.append(symptoms_dict.iloc[symptom, 1])
            input_vector[symp] = 1
            pred = model.predict([input_vector])[0]
            
            return render_template('show.html', disease= pred)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)