import json
import random
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import torch.serialization
from numpy._core.multiarray import _reconstruct
from flask import Flask, request, jsonify, render_template

# 1. GÜVENLİ GLOBALLERİ EKLE
torch.serialization.add_safe_globals([LabelEncoder, _reconstruct])

# 2. CİHAZ AYARI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 3. MODEL YÜKLEME
model_path = "hybrid_intent_model.pt"
try:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {str(e)}")
    exit()

# 4. TOKENIZER VE LABEL ENCODER
tokenizer = AutoTokenizer.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    **checkpoint['tokenizer_config']
)
le = LabelEncoder()
le.classes_ = checkpoint['label_encoder'].classes_
print("\nTanımlı intentler:", list(le.classes_))


# 5. MODEL SINIFI
class HybridIntentClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")

        # Tek katmanlı LSTM
        self.lstm = torch.nn.LSTM(
            input_size=self.transformer.config.hidden_size,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        lstm_out, _ = self.lstm(transformer_outputs.last_hidden_state)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)


# 6. MODELİ OLUŞTUR
model = HybridIntentClassifier(len(le.classes_))

# Ağırlıkları yükle
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("\nModel ağırlıkları başarıyla yüklendi")
except Exception as e:
    print(f"\nAğırlık yükleme hatası: {str(e)}")
    exit()

model.to(device)
model.eval()

# 7. INTENT VERİLERİ
with open('intents.json', 'r', encoding='utf-8') as f:
    intents_data = json.load(f)
intent_responses = {intent['tag']: intent['responses'] for intent in intents_data['intents']}


# 8. TAHMİN FONKSIYONU
def predict_intent(text, model, tokenizer, le, device, threshold=0.1):
    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    intent = le.inverse_transform([pred.item()])[0] if conf.item() >= threshold else "unknown"
    return intent, conf.item()


def get_response(intent, intent_responses):
    if intent == "unknown":
        return random.choice([
            "Üzgünüm, tam olarak anlayamadım",
            "Bu konuda yardımcı olamayacağım",
            "Lütfen başka şekilde ifade edin"
        ])
    return random.choice(intent_responses.get(intent, ["Anlamadım."]))


# 9. FLASK API
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Kullanıcı mesajını al
        user_message = request.json.get("message")

        # Modelin tahmini
        intent, confidence = predict_intent(user_message, model, tokenizer, le, device)

        # İlgili yanıtı al
        response_text = get_response(intent, intent_responses)

        # Yanıtı JSON formatında döndür
        return jsonify({
            "text": response_text,
            "type": "text",  # burada başka bir type dönebilirsiniz
            "confidence": confidence
        })
    except Exception as e:
        print(f"Hata: {e}")
        return jsonify({"text": "Bir hata oluştu, lütfen tekrar deneyin."}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

