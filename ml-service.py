from flask import Flask, jsonify

class MLService:
    def __init__(self, host='0.0.0.0', port=5000):
        """Initialize the ML Service."""
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Define API routes."""
        @self.app.route('/service', methods=['GET'])
        def service():
            return jsonify({'message': 'Service is running'}), 200
    
    def run(self):
        """Run the Flask application."""
        self.app.run(host=self.host, port=self.port)

if __name__ == "__main__":
    service = MLService()
    service.run()
