import os
import jwt
import cv2
import requests
from ultralytics import YOLO
from dotenv import load_dotenv
from middleware import authenticate_token
from multiprocessing import Process, Value
from flask import Flask, Response, render_template, jsonify
import logging

load_dotenv()
app = Flask(__name__)

monitor_process = None
running = Value('b', False)

@app.route('/')
def index():
    return render_template('index.html')


def monitoring(running):
    try:
        print('iniciou o monitoramento')
        cap = cv2.VideoCapture('rtsp://admin:admin@192.168.15.50:554/live/av0')
        model = YOLO('/var/www/html/visionvortex.com.br/models/yolov8n.pt')
        skip_frames = 15
        current_frame = 0
        ids = []

        while running.value:
            print(running.value)
            print('entrou no loop')
            success, frame = cap.read()
            if not success:
                break

            if current_frame % skip_frames == 0:
                results = model.track(frame, persist=True, verbose=False, device='cuda')[0]
                if results:
                    for result in results:
                        if result.boxes and result.boxes.data.tolist():
                            detection = result.boxes.data.tolist()[0]

                            if len(detection) >= 7 and detection[6] == 0:
                                if detection[5] > 0.75:
                                    id = detection[4]
                                    if id not in ids:
                                        file_path = f'snapshots/alert_person.jpg'
                                        cv2.imwrite(file_path, frame)
                                        send_alert('Pessoa detectada', detection[5], file_path)
                                        ids.append(id)

            current_frame += 1

        print('saiu do loop')
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f'Erro no processo de monitoramento: {e}')
    finally:
        running.value = False


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


@app.route('/start', methods=['GET'])
@authenticate_token
def start_monitoring():
    global monitor_process, running

    if not running.value:
        running.value = True
        monitor_process = Process(target=monitoring, args=(running,))
        monitor_process.start()

        return jsonify({'message': 'Monitoramento iniciado'}), 200
    
    elif running.value == True:

        return jsonify({'message': 'Monitoramento ativo'}), 200
    else:       
       return jsonify({'message': 'Não foi possível iniciar monitoramento'}), 400 
    

@app.route('/stop', methods=['GET'])
@authenticate_token
def stop_monitoring():
    global monitor_process, running

    if running.value:
        running.value = False
        monitor_process.join()
        monitor_process = None

        return jsonify({'message': 'Monitoramento desabilitado'}), 200
    else:
        return jsonify({'message': 'Monitoramento desabilitado'}), 200
    

@app.route('/status', methods=['GET'])
@authenticate_token
def get_status():
    global running

    if running.value:
        return jsonify({'message': 'Monitoramento ativo'}), 200
    else:
        return jsonify({'message': 'Monitoramento desabilitado'}), 200
    

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
            

@app.route('/monitoring')
@authenticate_token
def monitoringCamera():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=8085, debug=True)