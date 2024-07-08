import sys
import logging

sys.path.insert(0, '/var/www/html/visionvortex.com.br')
sys.path.insert(0, '/var/www/html/visionvortex.com.br/venv/lib/python3.12/site-packages/')

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

from app import app as application