import sys
import os

# Add project root to path so rag/ and helpers/ modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag.generator import SearchSemantic

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = (data or {}).get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    try:
        answer = SearchSemantic(user_message, 'frontend')
        return jsonify({'reply': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
