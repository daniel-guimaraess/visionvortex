import os
import cv2
import jwt
import requests
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

def send_alert(detection, confidence, file_path):
    files = {'file': open(file_path, 'rb')}
    payload = {
        'type': 'notification', 
        'detection': detection,
        'confidence': confidence
    }

    url = f'{os.getenv("BACKEND")}/alert'
    secret_key = os.getenv('SECRET_KEY')
    token = jwt.encode({}, secret_key, algorithm="HS256")
    bearer_token = f"Bearer {token}"

    try:
        response = requests.post(url, files=files, data=payload, headers = {"Authorization": bearer_token})
        if response.status_code == 200:
            os.remove(file_path)
            print(f'Alerta emitido com sucesso')
        else:
            os.remove(file_path)
            print(f'Erro ao emitir alerta: {response.text}')

    except requests.exceptions.RequestException as e:
        os.remove(file_path)
        print(f'Erro na requisição HTTP: {e}')


def monitoring(camera_ip):

    cap = cv2.VideoCapture(camera_ip)
    model = YOLO('models/yolov8n.pt')
    ids = []

    while True:
        success, frame = cap.read()

        if not success:
            break
        
        results = model.track(frame, persist=True, verbose=False, device='cuda')[0]
        
        if results:
            for result in results:
                if result.boxes and result.boxes.data.tolist():
                    detection = result.boxes.data.tolist()[0]
     
                    if len(detection) >= 7 and detection[6] == 0:
                        if detection[5] > 0.75:
                            id = detection[4]
                            if id not in ids:
                                file_path = f'imgs/alert_person.jpg'
                                cv2.imwrite(file_path, frame)
                                send_alert('Pessoa detectada', detection[5], file_path)
                                ids.append(id)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    monitoring(0)