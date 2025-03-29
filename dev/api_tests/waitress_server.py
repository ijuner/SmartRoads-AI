from waitress import serve
from rest_api_demo_app import app

if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    serve(app, host='0.0.0.0', port=5000, threads=6)