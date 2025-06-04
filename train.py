#BUNU KULLANIYORUM hybrid model
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from scipy import sparse

# 1. VERİ HAZIRLAMA
with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = []
labels = []
for intent in data['intents']:
    texts.extend(intent['patterns'])
    labels.extend([intent['tag']] * len(intent['patterns']))

# Label Encoding
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

# 2. TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    return_token_type_ids=False,
    padding='max_length',
    truncation=True,
    max_length=64
)


# 3. DATASET SINIFI
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)


# 4. MODEL SINIFI
class HybridIntentClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.lstm = torch.nn.LSTM(
            input_size=self.transformer.config.hidden_size,
            hidden_size=128,
            bidirectional=True,
            batch_first=True,
            num_layers=2,  # Dropout için layer sayısı artırıldı
            dropout=0.3
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


# 5. EARLY STOPPING
class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# 6. DEĞERLENDİRME FONKSİYONU
def evaluate(model, data_loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Softmax ile olasılıkları hesapla
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    # ROC-AUC hesaplama
    all_probs = np.concatenate(all_probs, axis=0)
    true_labels_onehot = np.eye(num_classes)[np.array(all_labels)]

    try:
        roc_auc = roc_auc_score(true_labels_onehot, all_probs, multi_class='ovo', average='weighted')
    except:
        roc_auc = 0.0

    return (total_loss / len(data_loader)), (correct / total), \
        f1_score(all_labels, all_preds, average='weighted'), roc_auc


# 7. EĞİTİM FONKSİYONU
def train_model(model, train_loader, val_loader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=3)

    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [],
               'val_f1': [], 'val_roc_auc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        val_loss, val_acc, val_f1, val_roc_auc = evaluate(model, val_loader, device)

        # Kayıt
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc_auc)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.4f}")
        print(
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")

        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    return model, history


# 8. GRAFİK FONKSİYONU
def plot_metrics(history):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(history['val_f1'], label='F1 Score', color='green')
    plt.title('Validation F1 Score')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(history['val_roc_auc'], label='ROC-AUC', color='purple')
    plt.title('Validation ROC-AUC Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()


# 9. TAHMİN FONKSİYONU
def predict_intent(text, model, tokenizer, le, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs).item()

    return le.inverse_transform([pred_idx])[0]


# 10. VERİ YÜKLEME VE MODEL EĞİTİMİ
if __name__ == "__main__":
    # Veri yükleme
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels_encoded, test_size=0.2, random_state=42
    )

    train_dataset = IntentDataset(train_texts, train_labels)
    val_dataset = IntentDataset(val_texts, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Model eğitimi
    model = HybridIntentClassifier(num_classes)
    trained_model, history = train_model(model, train_loader, val_loader, epochs=20)

    # Metrikleri görselleştirme
    plot_metrics(history)

    # Test
    test_texts = [
        "siber güvenlik çözümleri",
        "ürün iade nasıl yapılır",
        "hesabımı kapatmak istiyorum",
        "merhaba",
        "yardım istiyorum"
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for text in test_texts:
        print(f"'{text}' -> {predict_intent(text, trained_model, tokenizer, le, device)}")

    # Model kaydetme
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'label_encoder': le,
        'tokenizer_config': tokenizer.init_kwargs,
        'history': history
    }, 'hybrid_intent_model.pt')



