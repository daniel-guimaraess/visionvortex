import os
import jwt
import cv2
from datetime import datetime
import requests
import sqlite3
from ultralytics import YOLO
from dotenv import load_dotenv
from middleware import authenticate_token
from multiprocessing import Process
from flask import Flask, Response, render_template, jsonify

load_dotenv()
app = Flask(__name__)

def get_running_status():
    conn = sqlite3.connect('/home/agente/Documentos/database')
    cursor = conn.cursor()
    cursor.execute('SELECT status FROM monitoring WHERE id=1')
    row = cursor.fetchone()
    conn.close()
    return row[0] 


def set_running_status(status):
    conn = sqlite3.connect('/home/agente/Documentos/database')
    cursor = conn.cursor()
    cursor.execute('UPDATE monitoring SET status=? WHERE id=1', (int(status),))
    conn.commit()
    conn.close()


def checkThresholdTime(last_time, current_time):    
    difference = current_time - last_time
    if(difference >= 90):
        return True
    else:
        return False


def monitoring():
    try:
        cap = cv2.VideoCapture(0)
        model = YOLO('/var/www/html/visionvortex.com.br/models/best.pt')
        skip_frames = 15
        current_frame = 0
        threshold = 0.9

        alerts = {
            'tinha_eating': {
                'ids': [],
                'time': 0
            },
            'tinha_drinking': {
                'ids': [],
                'time': 0
            },
            'tinha_in_box': {
                'ids': [],
                'time': 0
            },
            'lua_eating': {
                'ids': [],
                'time': 0
            },
            'lua_drinking': {
                'ids': [],
                'time': 0
            },
            'lua_in_box': {
                'ids': [],
                'time': 0
            }
        }

        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_resized = cv2.resize(frame, (640,480))

            if current_frame % skip_frames == 0:
                results = model.track(frame_resized, show=True, persist=True, verbose=False, device='cuda')[0]
                if results:
                    for result in results:
                        if result.boxes and result.boxes.data.tolist():
                            detection = result.boxes.data.tolist()[0]

                            if len(detection) >= 7 and detection[6] == 0:
                                if detection[5] > threshold:
                                    id = detection[4]
                                    if id not in alerts['tinha_eating']['ids']:
                                        if(alerts['tinha_eating']['time'] == 0 or checkThresholdTime(alerts['tinha_eating']['time'], int(datetime.now().timestamp()))):
                                            file_path = f'snapshots/alert_tinha_eating_{current_frame}.jpg'
                                            cv2.imwrite(file_path, frame_resized)
                                            alerts['tinha_eating']['ids'].append(id)
                                            alerts['tinha_eating']['time'] = (int(datetime.now().timestamp()))
                            
                            elif len(detection) >= 7 and detection[6] == 1:
                                if detection[5] > threshold:
                                    id = detection[4]
                                    if id not in alerts['tinha_drinking']['ids']:
                                        if(checkThresholdTime(alerts['tinha_drinking']['time'] == 0 or alerts['tinha_drinking']['time'], int(datetime.now().timestamp()))):
                                            file_path = f'snapshots/alert_tinha_drinking_{current_frame}.jpg'
                                            cv2.imwrite(file_path, frame_resized)
                                            alerts['tinha_drinking']['ids'].append(id)
                                            alerts['tinha_drinking']['time'] = (int(datetime.now().timestamp()))

                            elif len(detection) >= 7 and detection[6] == 2:
                                if detection[5] > threshold:
                                    id = detection[4]
                                    if id not in alerts['tinha_in_box']['ids']:
                                        if(checkThresholdTime(alerts['tinha_in_box']['time'] == 0 or alerts['tinha_in_box']['time'], int(datetime.now().timestamp()))):
                                            file_path = f'snapshots/alert_tinha_in_box_{current_frame}.jpg'
                                            cv2.imwrite(file_path, frame_resized)
                                            alerts['tinha_in_box']['ids'].append(id)
                                            alerts['tinha_in_box']['time'] = (int(datetime.now().timestamp()))

                            elif len(detection) >= 7 and detection[6] == 3:
                                if detection[5] > threshold:
                                    id = detection[4]
                                    if id not in alerts['lua_eating']['ids']:
                                        if(checkThresholdTime(alerts['lua_eating']['time'] == 0 or alerts['lua_eating']['time'], int(datetime.now().timestamp()))):
                                            file_path = f'snapshots/alert_lua_eating_{current_frame}.jpg'
                                            cv2.imwrite(file_path, frame_resized)
                                            alerts['lua_eating']['ids'].append(id)
                                            alerts['lua_eating']['time'] = int(datetime.now().timestamp())
                            
                            elif len(detection) >= 7 and detection[6] == 4:
                                if detection[5] > threshold:
                                    id = detection[4]
                                    if id not in alerts['lua_drinking']['ids']:
                                        if(checkThresholdTime(alerts['lua_drinking']['time'] == 0 or alerts['lua_drinking']['time'], int(datetime.now().timestamp()))):
                                            file_path = f'snapshots/alert_lua_drinking_{current_frame}.jpg'
                                            cv2.imwrite(file_path, frame_resized)
                                            alerts['lua_drinking']['ids'].append(id)
                                            alerts['lua_drinking']['time'] = (int(datetime.now().timestamp()))

                            elif len(detection) >= 7 and detection[6] == 5:
                                if detection[5] > threshold:
                                    id = detection[4]
                                    if id not in alerts['lua_in_box']['ids']:
                                        if(checkThresholdTime(alerts['lua_in_box']['time'] == 0 or alerts['lua_in_box']['time'], int(datetime.now().timestamp()))):
                                            file_path = f'snapshots/alert_lua_in_box_{current_frame}.jpg'
                                            cv2.imwrite(file_path, frame_resized)
                                            alerts['lua_in_box']['ids'].append(id)
                                            alerts['lua_in_box']['time'] = (int(datetime.now().timestamp()))

            current_frame += 1
    
    except Exception as e:
        print(f'Erro no processo de monitoramento: {e}')
    finally:
        set_running_status(0)

def send_alert(detection, confidence, file_path):
    files = {'file': open(file_path, 'rb')}

    payload = {
        'type': 'notification', 
        'detection': detection,
        'confidence': confidence
    }

    url = f'{os.getenv("BACKEND")}/alert'
    secret_key = os.getenv('SECRET_KEY')
    token = jwt.encode({}, secret_key, algorithm='HS256')
    bearer_token = f'Bearer {token}'

    try:
        response = requests.post(url, files=files, data=payload, headers={'Authorization': bearer_token})
        if response.status_code == 200:
            os.remove(file_path)
            print(f'Alerta emitido com sucesso')
        else:
            os.remove(file_path)
            print(f'Erro ao emitir alerta: {response.text}')

    except requests.exceptions.RequestException as e:
        os.remove(file_path)
        print(f'Erro na requisição HTTP: {e}')
        

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['GET'])
@authenticate_token
def start_monitoring():
    global monitor_process
    if not get_running_status() == 1:
        set_running_status(1)
        monitor_process = Process(target=monitoring)
        monitor_process.start()

        return jsonify({'message': 'Monitoramento iniciado'}), 200
    
    else:
        return jsonify({'message': 'Monitoramento ativo'}), 200
    

@app.route('/stop', methods=['GET'])
@authenticate_token
def stop_monitoring():
    global monitor_process

    if get_running_status() == 1:
        set_running_status(0)

        return jsonify({'message': 'Monitoramento desabilitado'}), 200
    else:
        return jsonify({'message': 'Monitoramento desabilitado'}), 200
    

@app.route('/status', methods=['GET'])
@authenticate_token
def get_status():
    if get_running_status():
        return jsonify({'message': 'Monitoramento ativo'}), 200
    else:
        return jsonify({'message': 'Monitoramento desabilitado'}), 200


@app.route('/monitoring')
def monitoringCamera():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=80, debug=True)