import cv2
from dotenv import load_dotenv
from flask import render_template
from flask import Flask, Response, render_template
from middleware import authenticate_token

load_dotenv()
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

def generate_frames():
    camera_ip = "rtsp://usuario:senha@ip_da_camera:porta/caminho"
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
def monitoring():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=80, debug=True)