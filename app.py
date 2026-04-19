# ============================================================
#   SLEEP DISORDER DETECTION - FLASK BACKEND
#   Run with: python app.py
# ============================================================

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'sdd_secret_key_2026'

# ============================================================
# LOAD MODEL
# ============================================================

model         = joblib.load("sleep_disorder_best_model.pkl")
feature_names = joblib.load("model_feature_names.pkl")

DISORDER_MAP = {
    0: 'None',
    1: 'Insomnia',
    2: 'Sleep Apnea',
    3: 'Narcolepsy',
    4: 'Restless Leg Syndrome'
}


DISORDER_INFO = {
    'None': {
        'icon'  : 'fa-check-circle',
        'color' : 'success',
        'hex'   : '#27ae60',
        'desc'  : 'Your sleep patterns appear healthy. No disorder detected.',
        'tips'  : [
            'Maintain a consistent sleep schedule every day',
            'Keep up your regular physical activity routine',
            'Avoid heavy meals and caffeine close to bedtime',
            'Continue monitoring your sleep with wearables',
        ],
        'severity': 'Normal'
    },
    'Insomnia': {
        'icon'  : 'fa-moon',
        'color' : 'primary',
        'hex'   : '#2980b9',
        'desc'  : 'Difficulty falling or staying asleep. Often linked to high stress and anxiety.',
        'tips'  : [
            'Establish a relaxing bedtime routine',
            'Avoid screens at least 1 hour before sleep',
            'Try deep breathing or meditation techniques',
            'Limit caffeine intake after 2 PM',
            'Keep your bedroom cool, dark, and quiet',
        ],
        'severity': 'Moderate'
    },
    'Sleep Apnea': {
        'icon'  : 'fa-lungs',
        'color' : 'danger',
        'hex'   : '#e74c3c',
        'desc'  : 'Breathing repeatedly stops during sleep. Strongly associated with high BMI and low SpO2.',
        'tips'  : [
            'Consult a doctor immediately for CPAP therapy evaluation',
            'Sleep on your side to keep airways open',
            'Weight loss can significantly reduce apnea severity',
            'Avoid alcohol and sedatives before bedtime',
            'Use a pulse oximeter to monitor nightly SpO2',
        ],
        'severity': 'Severe — See a Doctor'
    },
    'Narcolepsy': {
        'icon'  : 'fa-bolt',
        'color' : 'warning',
        'hex'   : '#f39c12',
        'desc'  : 'Excessive daytime sleepiness with sudden uncontrollable sleep attacks.',
        'tips'  : [
            'Take scheduled short naps (10–20 min) during the day',
            'Consult a neurologist for medication options',
            'Never drive or operate machinery when drowsy',
            'Maintain a very strict sleep schedule',
            'Inform your workplace or school about your condition',
        ],
        'severity': 'Moderate — Needs Monitoring'
    },
    'Restless Leg Syndrome': {
        'icon'  : 'fa-running',
        'color' : 'info',
        'hex'   : '#8e44ad',
        'desc'  : 'Uncomfortable sensations in legs with urge to move during sleep.',
        'tips'  : [
            'Gentle leg massages before bedtime can help',
            'Apply warm or cold packs to legs at night',
            'Moderate exercise during the day (not before bed)',
            'Check iron and folate levels with your doctor',
            'Avoid caffeine and alcohol in the evening',
        ],
        'severity': 'Mild — Manageable'
    }
}

# ============================================================
# HELPER: BUILD INPUT VECTOR
# ============================================================

def build_input(data):
    sleep_eff = ((data['sleep_duration'] / 9.5) * 0.5 +
                 (data['sleep_quality'] / 10) * 0.5) * 100

    clinical_risk = ((data['ahi'] / 60) * 0.4 +
                     ((100 - data['sao2']) / 15) * 0.4 +
                     (data['stress'] / 10) * 0.2) * 100

    activity_idx = ((data['daily_steps'] / 15000) * 0.5 +
                    (data['physical_activity'] / 90) * 0.5) * 100

    s, d = data['systolic'], data['diastolic']
    if s < 120 and d < 80:   bp_enc = 0
    elif s < 130 and d < 80: bp_enc = 1
    elif s < 140 or d < 90:  bp_enc = 2
    else:                     bp_enc = 3

    age = data['age']
    if age < 30:   age_enc = 0
    elif age < 45: age_enc = 1
    elif age < 60: age_enc = 2
    else:          age_enc = 3

    wearable_flag = int(data['w_spo2'] < 92 or data['movement'] > 7)
    bmi_map = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    bmi_enc = bmi_map[data['bmi_cat']]
    gender_enc = 1 if data['gender'] == 'Male' else 0

    occ_list = ['Driver','Engineer','Manager','Nurse','Other',
                'Retired','Sales','Student','Teacher','Doctor']
    occ_encoded = {f"Occ_{o}": 0 for o in occ_list}
    occ_key = f"Occ_{data['occupation']}"
    if occ_key in occ_encoded:
        occ_encoded[occ_key] = 1

    scale_vals = {
        'Age'                          : (data['age'] - 46) / 15,
        'Sleep_Duration_hrs'           : (data['sleep_duration'] - 6.3) / 1.3,
        'Quality_of_Sleep_1_10'        : (data['sleep_quality'] - 5.5) / 2.5,
        'Physical_Activity_min_day'    : (data['physical_activity'] - 40) / 22,
        'Stress_Level_1_10'            : (data['stress'] - 5.5) / 2.8,
        'Systolic_BP'                  : (data['systolic'] - 125) / 14,
        'Diastolic_BP'                 : (data['diastolic'] - 78) / 9,
        'Heart_Rate_bpm'               : (data['heart_rate'] - 73) / 9,
        'Daily_Steps'                  : (data['daily_steps'] - 6500) / 3200,
        'AHI_Score'                    : (data['ahi'] - 12) / 14,
        'SaO2_Level_pct'               : (data['sao2'] - 94) / 3.5,
        'Wearable_Movement_Actigraphy' : (data['movement'] - 4.5) / 2.8,
        'Wearable_SpO2_pct'            : (data['w_spo2'] - 94) / 3.5,
        'HRV_ms'                       : (data['hrv'] - 48) / 16,
        'Body_Temp_C'                  : (data['body_temp'] - 36.7) / 0.35,
        'Respiratory_Rate_bpm'         : (data['resp_rate'] - 16) / 2.5,
        'Sleep_Efficiency_Score'       : (sleep_eff - 60) / 18,
        'Clinical_Risk_Score'          : (clinical_risk - 30) / 20,
        'Activity_Index'               : (activity_idx - 45) / 22,
    }

    row = {}
    for f in feature_names:
        if f in scale_vals:          row[f] = scale_vals[f]
        elif f == 'Gender_Encoded':  row[f] = gender_enc
        elif f == 'BMI_Encoded':     row[f] = bmi_enc
        elif f == 'BP_Encoded':      row[f] = bp_enc
        elif f == 'Age_Group_Encoded': row[f] = age_enc
        elif f == 'Wearable_Risk_Flag': row[f] = wearable_flag
        elif f in occ_encoded:       row[f] = occ_encoded[f]
        else:                        row[f] = 0

    return pd.DataFrame([row])[feature_names]

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'age'              : int(request.form['age']),
            'gender'           : request.form['gender'],
            'occupation'       : request.form['occupation'],
            'sleep_duration'   : float(request.form['sleep_duration']),
            'sleep_quality'    : int(request.form['sleep_quality']),
            'physical_activity': int(request.form['physical_activity']),
            'stress'           : int(request.form['stress']),
            'bmi_cat'          : request.form['bmi_cat'],
            'systolic'         : int(request.form['systolic']),
            'diastolic'        : int(request.form['diastolic']),
            'heart_rate'       : int(request.form['heart_rate']),
            'daily_steps'      : int(request.form['daily_steps']),
            'ahi'              : float(request.form['ahi']),
            'sao2'             : float(request.form['sao2']),
            'movement'         : float(request.form['movement']),
            'w_spo2'           : float(request.form['w_spo2']),
            'hrv'              : float(request.form['hrv']),
            'body_temp'        : float(request.form['body_temp']),
            'resp_rate'        : int(request.form['resp_rate']),
        }

        X       = build_input(data)
        pred    = model.predict(X)[0]
        proba   = model.predict_proba(X)[0]

        disorder    = DISORDER_MAP[pred]
        confidence  = round(float(proba[pred]) * 100, 1)
        info        = DISORDER_INFO[disorder]

        all_probs = {
            DISORDER_MAP[i]: round(float(p) * 100, 1)
            for i, p in enumerate(proba)
        }

        warnings = []
        if data['ahi'] > 15:
            warnings.append("High AHI Score detected — please consult a sleep specialist immediately.")
        if data['sao2'] < 92 or data['w_spo2'] < 92:
            warnings.append("Low oxygen saturation detected — seek medical attention promptly.")
        if data['stress'] >= 8:
            warnings.append("Very high stress level — consider professional stress management support.")
        if data['sleep_duration'] < 5:
            warnings.append("Critically low sleep duration — chronic sleep deprivation detected.")

        result = {
            'disorder'  : disorder,
            'confidence': confidence,
            'info'      : info,
            'all_probs' : all_probs,
            'warnings'  : warnings,
            'data'      : data,
            'timestamp' : datetime.now().strftime("%B %d, %Y at %I:%M %p")
        }

        session['last_result'] = json.dumps(result)
        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('predict.html', error=str(e))


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/api/stats')
def api_stats():
    stats = {
        'model_accuracy'  : 98.5,
        'total_features'  : 34,
        'training_samples': 1830,
        'test_samples'    : 200,
        'disorders'       : 5,
        'disorder_dist'   : {
            'None': 44, 'Insomnia': 26,
            'Sleep Apnea': 18, 'Narcolepsy': 6,
            'Restless Leg Syndrome': 6
        },
        'model_comparison': {
            'Random Forest': 98.0,
            'XGBoost'      : 98.5
        },
        'top_features': [
            'AHI_Score', 'SaO2_Level_pct', 'Sleep_Duration_hrs',
            'Clinical_Risk_Score', 'Stress_Level_1_10',
            'Wearable_SpO2_pct', 'HRV_ms', 'Quality_of_Sleep_1_10'
        ]
    }
    return jsonify(stats)


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  Sleep Disorder Detection System")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True)
