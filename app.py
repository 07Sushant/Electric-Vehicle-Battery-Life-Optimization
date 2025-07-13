from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime
import pandas as pd
from scipy import stats

app = Flask(__name__)
app.secret_key = 'battery_rul_prediction_secret_key'

# Database setup
def get_db_connection():
    conn = sqlite3.connect('battery_app.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        cycle_index REAL,
        discharge_time REAL,
        decrement_36_34V REAL,
        max_voltage_discharge REAL,
        min_voltage_charge REAL,
        time_at_415V REAL,
        time_constant_current REAL,
        charging_time REAL,
        predicted_rul REAL,
        prediction_date TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Load the model with error handling
def load_rul_model():
    try:
        model_path = 'model//battery_rul_model.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        return load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model = load_rul_model()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please log in to access this page')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['username'] = username
            session['user_id'] = user['id']
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        existing_user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        
        if existing_user:
            flash('Username already exists')
        else:
            hashed_password = generate_password_hash(password)
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            flash('Registration successful! Please log in.')
            conn.close()
            return redirect(url_for('login'))
        
        conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    prediction_result = None
    
    # Get historical data for visualization
    conn = get_db_connection()
    historical_data = pd.read_sql_query(
        'SELECT cycle_index, discharge_time, max_voltage_discharge, predicted_rul, prediction_date FROM predictions WHERE user_id = ? ORDER BY prediction_date',
        conn,
        params=(session['user_id'],)
    )
    conn.close()
    
    if request.method == 'POST':
        try:
            # Get form data
            cycle_index = float(request.form['cycle_index'])
            discharge_time = float(request.form['discharge_time'])
            decrement_36_34V = float(request.form['decrement_36_34V'])
            max_voltage_discharge = float(request.form['max_voltage_discharge'])
            min_voltage_charge = float(request.form['min_voltage_charge'])
            time_at_415V = float(request.form['time_at_415V'])
            time_constant_current = float(request.form['time_constant_current'])
            charging_time = float(request.form['charging_time'])
            
            # Input validation
            if cycle_index < 0 or discharge_time < 0 or time_at_415V < 0 or charging_time < 0:
                flash("Input values cannot be negative")
                return render_template('predict.html')
            
            # Arrange features in the correct order and shape for the model
            features = np.array([cycle_index, discharge_time, decrement_36_34V,
                                max_voltage_discharge, min_voltage_charge,
                                time_at_415V, time_constant_current, charging_time], dtype=np.float32)
            features = features.reshape(1, 1, 8)  # (batch, timestep, features)
            
            # Check if model is loaded
            if model is None:
                flash('Model not available. Please contact the administrator.')
                return render_template('predict.html')
                
            # Make prediction
            predicted_rul = model.predict(features)
            prediction_result = round(float(predicted_rul[0][0]), 2)
            
            # Store the prediction in the database
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO predictions (user_id, cycle_index, discharge_time, decrement_36_34V, max_voltage_discharge, min_voltage_charge, time_at_415V, time_constant_current, charging_time, predicted_rul, prediction_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (session['user_id'], cycle_index, discharge_time, decrement_36_34V, max_voltage_discharge, min_voltage_charge, time_at_415V, time_constant_current, charging_time, prediction_result, datetime.now())
            )
            conn.commit()
            conn.close()
            
            # Prepare visualization data
            voltage_trend = historical_data['max_voltage_discharge'].tolist() + [max_voltage_discharge]
            discharge_trend = historical_data['discharge_time'].tolist() + [discharge_time]
            cycle_trend = historical_data['cycle_index'].tolist() + [cycle_index]
            
            # Calculate statistical insights
            voltage_z_score = stats.zscore(voltage_trend)[-1] if len(voltage_trend) > 1 else 0
            discharge_z_score = stats.zscore(discharge_trend)[-1] if len(discharge_trend) > 1 else 0
            
            # Generate health insights
            insights = {
                'voltage_health': 'normal' if abs(voltage_z_score) < 2 else 'abnormal',
                'discharge_health': 'normal' if abs(discharge_z_score) < 2 else 'abnormal',
                'cycle_impact': 'high' if cycle_index > np.mean(cycle_trend) + np.std(cycle_trend) else 'normal'
            }
            
            # Prepare visualization data
            viz_data = {
                'prediction': prediction_result,
                'voltage_trend': voltage_trend,
                'discharge_trend': discharge_trend,
                'cycle_trend': cycle_trend,
                'insights': insights
            }
            
            # Store the prediction and visualization data in the session
            session['prediction'] = prediction_result
            session['viz_data'] = viz_data
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(viz_data)
            
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            flash(f'Prediction failed: {str(e)}')
    
    return render_template('predict.html')

@app.route('/dashboard')
@login_required
def dashboard():
    prediction = session.get('prediction', None)
    
    # Get the last 5 predictions from the database
    conn = get_db_connection()
    recent_predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY prediction_date DESC LIMIT 5', 
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('dashboard.html', prediction=prediction, recent_predictions=recent_predictions)

@app.route('/history')
@login_required
def history():
    # Get all predictions for the user
    conn = get_db_connection()
    all_predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY prediction_date DESC', 
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('history.html', predictions=all_predictions)

@app.route('/visualize/<int:prediction_id>')
@login_required
def visualize(prediction_id):
    conn = get_db_connection()
    prediction = conn.execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?', 
                            (prediction_id, session['user_id'])).fetchone()
    
    # Get all historical data up to this prediction
    historical_data = conn.execute('''
        SELECT cycle_index, discharge_time, predicted_rul, prediction_date,
               max_voltage_discharge, min_voltage_charge, charging_time
        FROM predictions 
        WHERE user_id = ? AND prediction_date <= (
            SELECT prediction_date FROM predictions WHERE id = ?
        )
        ORDER BY cycle_index
    ''', (session['user_id'], prediction_id)).fetchall()
    
    conn.close()

    if not prediction:
        flash('Prediction not found')
        return redirect(url_for('history'))

    # Prepare historical trend data
    cycle_indices = [row['cycle_index'] for row in historical_data]
    rul_values = [row['predicted_rul'] for row in historical_data]
    discharge_times = [row['discharge_time'] for row in historical_data]

    # Calculate health indicators
    latest_rul = prediction['predicted_rul']
    rul_status = 'Healthy' if latest_rul > 200 else 'Warning' if latest_rul > 100 else 'Critical'
    rul_color = 'rgb(34, 197, 94)' if latest_rul > 200 else 'rgb(234, 179, 8)' if latest_rul > 100 else 'rgb(239, 68, 68)'

    # Calculate trend lines using polynomial fit
    if len(cycle_indices) > 1:
        z_rul = np.polyfit(cycle_indices, rul_values, 2)
        p_rul = np.poly1d(z_rul)
        
        z_capacity = np.polyfit(cycle_indices, discharge_times, 2)
        p_capacity = np.poly1d(z_capacity)
        
        # Convert to list directly instead of using tolist()
        x_smooth = list(np.linspace(min(cycle_indices), max(cycle_indices), 100))
        rul_smooth = list(p_rul(x_smooth))
        capacity_smooth = list(p_capacity(x_smooth))
        
        # Calculate degradation rate
        rul_gradient = list(np.gradient(rul_smooth))
        avg_degradation = np.mean(rul_gradient)
    else:
        x_smooth = cycle_indices
        rul_smooth = rul_values
        capacity_smooth = discharge_times
        avg_degradation = 0

    # Prepare plot data with enhanced styling
    plot_data = {
        'rulCyclePlot': {
            'data': [
                {
                    'x': cycle_indices,
                    'y': rul_values,
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Actual Points',
                    'marker': {
                        'size': 10,
                        'color': rul_color,
                        'symbol': 'circle',
                        'line': {'width': 2, 'color': 'white'}
                    },
                    'hovertemplate': 'Cycle: %{x}<br>RUL: %{y:.1f}<extra></extra>'
                },
                {
                    'x': x_smooth,  # No need for .tolist() if already a list
                    'y': rul_smooth,  # No need for .tolist() if already a list
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Trend',
                    'line': {
                        'color': rul_color,
                        'width': 3,
                        'dash': 'solid',
                        'shape': 'spline'
                    },
                    'hovertemplate': 'Trend RUL: %{y:.1f}<extra></extra>'
                }
            ],
            'layout': {
                'title': {
                    'text': f'RUL vs Cycle (Status: {rul_status})',
                    'font': {'size': 24, 'color': rul_color}
                },
                'xaxis': {
                    'title': 'Cycle Index',
                    'gridcolor': 'rgba(0,0,0,0.1)',
                    'zerolinecolor': 'rgba(0,0,0,0.2)'
                },
                'yaxis': {
                    'title': 'Remaining Useful Life',
                    'gridcolor': 'rgba(0,0,0,0.1)',
                    'zerolinecolor': 'rgba(0,0,0,0.2)'
                },
                'showlegend': True,
                'height': 500,
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'t': 60, 'b': 40, 'l': 40, 'r': 40},
                'annotations': [{
                    'text': f'Degradation Rate: {abs(avg_degradation):.2f} units/cycle',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0,
                    'y': 1.05,
                    'showarrow': False,
                    'font': {'size': 12, 'color': 'gray'}
                }]
            }
        },
        'capacityPlot': {
            'data': [
                {
                    'x': cycle_indices,
                    'y': discharge_times,
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Capacity Points',
                    'marker': {
                        'size': 10,
                        'color': discharge_times,
                        'colorscale': 'Viridis',
                        'showscale': True,
                        'symbol': 'circle',
                        'line': {'width': 2, 'color': 'white'}
                    },
                    'hovertemplate': 'Cycle: %{x}<br>Discharge Time: %{y:.1f}s<extra></extra>'
                },
                {
                    'x': x_smooth,  # No need for .tolist() if already a list
                    'y': capacity_smooth,  # No need for .tolist() if already a list
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Capacity Trend',
                    'line': {
                        'color': 'rgba(102, 51, 153, 0.8)',
                        'width': 3,
                        'shape': 'spline'
                    },
                    'hovertemplate': 'Trend Capacity: %{y:.1f}s<extra></extra>'
                }
            ],
            'layout': {
                'title': {
                    'text': 'Battery Capacity Evolution',
                    'font': {'size': 24}
                },
                'xaxis': {
                    'title': 'Cycle Index',
                    'gridcolor': 'rgba(0,0,0,0.1)',
                    'zerolinecolor': 'rgba(0,0,0,0.2)'
                },
                'yaxis': {
                    'title': 'Discharge Time (s)',
                    'gridcolor': 'rgba(0,0,0,0.1)',
                    'zerolinecolor': 'rgba(0,0,0,0.2)'
                },
                'showlegend': True,
                'height': 500,
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'t': 60, 'b': 40, 'l': 40, 'r': 40}
            }
        },
        'voltagePlot': {
            'data': [{
                'values': [prediction['max_voltage_discharge'], 
                          prediction['min_voltage_charge']],
                'labels': ['Max Discharge', 'Min Charge'],
                'type': 'pie',
                'name': 'Voltage Distribution'
            }],
            'layout': {
                'title': 'Voltage Analysis'
            }
        },
        'timePlot': {
            'data': [{
                'x': ['Discharge', 'Charging', '4.15V', 'Constant Current'],
                'y': [prediction['discharge_time'], 
                      prediction['charging_time'],
                      prediction['time_at_415V'],
                      prediction['time_constant_current']],
                'type': 'bar',
                'name': 'Time Distribution'
            }],
            'layout': {
                'title': 'Time Analysis',
                'xaxis': {'title': 'Phase'},
                'yaxis': {'title': 'Time (s)'}
            }
        },
        'featurePlot': {
            'data': [{
                'x': ['Cycle', 'Discharge', 'Voltage Dec', 'Max V', 'Min V', 
                      'Time 4.15V', 'Const Current', 'Charging'],
                'y': [prediction['cycle_index'], prediction['discharge_time'],
                      prediction['decrement_36_34V'], prediction['max_voltage_discharge'],
                      prediction['min_voltage_charge'], prediction['time_at_415V'],
                      prediction['time_constant_current'], prediction['charging_time']],
                'type': 'bar',
                'name': 'Feature Values'
            }],
            'layout': {
                'title': 'Feature Analysis',
                'xaxis': {'title': 'Features'},
                'yaxis': {'title': 'Value'},
                'height': 400,
                'margin': {'t': 40, 'b': 40, 'l': 40, 'r': 40}
            }
        },
        'performancePlot': {
            'data': [{
                'values': [prediction['predicted_rul'], 
                          max(0, 300 - prediction['predicted_rul'])],
                'labels': ['Remaining Life', 'Used Life'],
                'type': 'pie',
                'hole': .4,
                'name': 'Battery Life'
            }],
            'layout': {
                'title': 'Battery Performance',
                'height': 400,
                'margin': {'t': 40, 'b': 40, 'l': 40, 'r': 40}
            }
        }
    }

    return render_template('visualize.html', 
                         prediction=prediction, 
                         plot_data=plot_data, 
                         status=rul_status, 
                         degradation_rate=abs(avg_degradation))


if __name__ == '__main__':
    app.run(debug=True)
