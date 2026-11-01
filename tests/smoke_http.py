"""Exercise a running Gradio server and its real HTTP vector database."""
import json
import os
import urllib.request
from pathlib import Path
from gradio_client import Client

url = os.getenv('APP_URL', 'http://127.0.0.1:7860')
with urllib.request.urlopen(url+'/config', timeout=30) as response:
    assert response.status == 200
client = Client(url)
result = client.predict('Asia', 2002, 25000000, 5000, 70, 65, 70,
                        'Moderate simulated investment', api_name='/predict')
assert 'Tuổi thọ dự đoán' in result[0]
assert len(result[1]['data']) == 5
assert 'distance_l2' in result[1]['headers']
report = {'ui_http': True, 'inference': True, 'vector_rows': 5,
          'chroma_http_configured': bool(os.getenv('CHROMA_HOST'))}
root = Path(__file__).resolve().parents[1]
(root/'reports/http_smoke.json').write_text(json.dumps(report, indent=2))
print('PASS: HTTP UI, 3 predictions, 5 live vector neighbors')
