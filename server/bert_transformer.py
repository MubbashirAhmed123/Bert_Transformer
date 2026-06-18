import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "saved_model"
BASE_MODEL = "bert-base-uncased"

intents = [
    "greeting",
    "password_reset",
    "refund_request",
    "order_status",
    "payment_issue",
    "complaint",
]

data = [
("hello",0),("hi",0),("hey",0),("good morning",0),("good evening",0),
("good afternoon",0),("hi there",0),("hello there",0),("hey buddy",0),
("hey bro",0),("hi team",0),("greetings",0),("yo",0),("what's up",0),
("how are you",0),("nice to meet you",0),("pleased to meet you",0),
("hi support",0),("hello support",0),("good day",0),
("hi everyone",0),("hey there",0),("morning",0),("evening",0),
("hello my friend",0),("hey guys",0),("hello team",0),("hi buddy",0),
("good to see you",0),("hi sir",0),("hi madam",0),("hello sir",0),
("hello madam",0),("hey support team",0),("hi customer care",0),
("hello customer service",0),("greetings team",0),("hey hello",0),
("hi hi",0),("hello again",0),

("i forgot my password",1),("reset my password",1),("cannot login",1),
("i can't log in",1),("password not working",1),("help me reset password",1),
("forgot login password",1),("account login issue",1),
("unable to access my account",1),("how do i reset password",1),
("need password reset",1),("change my password",1),
("recover my password",1),("lost my password",1),
("login problem",1),("cannot access account",1),
("forgot credentials",1),("reset account password",1),
("my password expired",1),("login error",1),
("can't sign in",1),("trouble logging in",1),
("need help logging in",1),("sign in problem",1),
("account locked",1),("unlock my account",1),
("reset login details",1),("forgot username and password",1),
("password reset link not working",1),("login failed",1),
("authentication issue",1),("verify my account",1),
("can't remember password",1),("how to recover account",1),
("forgot security code",1),("reset security password",1),
("login verification failed",1),("otp not working",1),
("unable to sign in",1),("password assistance needed",1),

("i want refund",2),("return my money",2),("refund my payment",2),
("money back please",2),("cancel order and refund",2),
("i need a refund",2),("refund this product",2),
("i want my money back",2),("refund request",2),
("please process refund",2),("refund status",2),
("how to get refund",2),("return the item",2),
("send my refund",2),("requesting refund",2),
("refund for damaged item",2),("refund for wrong product",2),
("i am not satisfied refund",2),("cancel and refund",2),
("refund amount",2),("refund initiated?",2),
("give my money back",2),("refund policy",2),
("claim refund",2),("start refund process",2),
("refund my order",2),("i need refund immediately",2),
("refund problem",2),("payment refund required",2),
("please refund asap",2),("return and refund",2),
("money not returned",2),("refund still pending",2),
("waiting for refund",2),("refund issue",2),
("refund confirmation",2),("refund not received",2),
("cancel purchase refund",2),("refund delay",2),
("i want cancellation refund",2),

("where is my order",3),("track my package",3),
("order status please",3),("order tracking",3),
("when will it arrive",3),("delivery status",3),
("shipment tracking",3),("order update",3),
("check order progress",3),("where is my package",3),
("track shipment",3),("order delivery date",3),
("is my order shipped",3),("order dispatch status",3),
("tracking number",3),("delivery update",3),
("order not delivered",3),("order delay",3),
("order expected date",3),("track order number",3),
("package status",3),("shipment delay",3),
("order location",3),("delivery tracking",3),
("track my order now",3),("check delivery time",3),
("is order out for delivery",3),("delivery issue",3),
("order confirmation",3),("has my order shipped",3),
("order progress",3),("delivery schedule",3),
("track delivery",3),("order not arrived",3),
("shipment status update",3),("order still processing",3),
("delivery ETA",3),("when will order reach",3),
("order arrival time",3),("order tracking details",3),

("payment failed",4),("transaction not working",4),
("card declined",4),("payment error",4),
("unable to pay",4),("transaction failed",4),
("payment declined",4),("payment problem",4),
("card not accepted",4),("billing issue",4),
("cannot complete payment",4),("checkout error",4),
("payment gateway error",4),("payment timeout",4),
("payment not processed",4),("money deducted but not confirmed",4),
("double charged",4),("charged twice",4),
("wrong payment amount",4),("refund not processed",4),
("payment pending",4),("transaction stuck",4),
("card transaction failed",4),("online payment issue",4),
("payment declined by bank",4),("bank rejected payment",4),
("unable to process payment",4),("payment unsuccessful",4),
("failed checkout",4),("error while paying",4),
("card error",4),("payment blocked",4),
("problem with billing",4),("invoice payment failed",4),
("debit card declined",4),("credit card error",4),
("payment verification failed",4),("upi payment failed",4),
("wallet payment error",4),("cannot make payment",4),

("this is bad service",5),("very poor experience",5),
("i am unhappy",5),("not satisfied",5),
("worst service ever",5),("very disappointed",5),
("bad customer service",5),("this is terrible",5),
("really bad experience",5),("i want to complain",5),
("service quality is bad",5),("i am frustrated",5),
("very bad support",5),("not happy with service",5),
("unsatisfactory experience",5),("i am angry",5),
("this is unacceptable",5),("bad behavior",5),
("poor response",5),("not impressed",5),
("terrible support",5),("very slow service",5),
("support is useless",5),("extremely disappointed",5),
("horrible experience",5),("customer service is awful",5),
("this is ridiculous",5),("very bad service quality",5),
("no proper support",5),("bad experience overall",5),
("i feel cheated",5),("service is pathetic",5),
("not good at all",5),("very dissatisfied",5),
("unhappy with support",5),("complaint about service",5),
("this is not acceptable",5),("i don't like this service",5),
("angry about service",5),("bad support team",5),
]

_tokenizer = None
_model = None
_device = None


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=32,
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_and_save(epochs=5):
    df = pd.DataFrame(data, columns=["text", "label"])
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    train_texts, _, train_labels, _ = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(intents),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Training complete. Model saved to {MODEL_DIR}")


def _load_model():
    global _tokenizer, _model, _device

    if _model is not None:
        return

    if not MODEL_DIR.exists():
        print("No saved model found — training once (this takes a few minutes)...")
        train_and_save()

    print("Loading saved model...")
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    _model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.to(_device)
    _model.eval()
    print("Model ready.")


def predict(text):
    _load_model()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item()

    return {
        "intent": intents[predicted_class],
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    train_and_save()

    tests = [
        "hello",
        "i need to reset my password",
        "where is my package",
        "payment not working",
        "i want my money back",
        "i am very unhappy with the service",
        "this is terrible",
        "very bad service",
    ]

    for t in tests:
        result = predict(t)
        print("Text:", t)
        print("Predicted:", result["intent"], f"({result['confidence']:.1%})")
        print("------")
