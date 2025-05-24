import os
from flask import Flask, request, jsonify
from transformers.models.auto.tokenization_auto import AutoTokenizer
from adapters import AutoAdapterModel

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

@app.route('/', methods=['POST'])
def index():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Invalid input data'}), 400

    embedding = get_embedding(str(text))
    return jsonify(embedding)
    
def get_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = model(**inputs)

    embedding = output.last_hidden_state[:, 0, :].squeeze().detach().cpu().tolist()

    return embedding 

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))