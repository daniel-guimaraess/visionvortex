import jwt
from functools import wraps
from flask import request, jsonify
from dotenv import load_dotenv
import os

load_dotenv('/var/www/html/visionvortex.com.br/.env')

def authenticate_token(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'message': 'Token not found'}), 401

        try:
            payload = jwt.decode(token, str(os.getenv('SECRET_KEY')), algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Expired token'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401

        return func(*args, **kwargs)

    return decorated