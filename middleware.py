import jwt
from functools import wraps
from flask import request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()
def authenticate_token(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'message': 'Token de autenticação ausente'}), 401

        try:
            payload = jwt.decode(token, os.getenv('SECRE_KEY'), algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expirado'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token inválido'}), 401

        return func(*args, **kwargs)

    return decorated