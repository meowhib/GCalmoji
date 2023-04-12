import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Define your event titles and corresponding emojis (as integer labels)
event_titles = [
    "Birthday Party",
    "Wedding Ceremony",
    "Job Interview Work",
    "Dinner Restaurant Reservation",
    "Call",
    "Gym Workout Session",
    "Festival Celebration Fest"]

emojis = list(range(len(event_titles)))  # Map your emojis to integer labels

# Initialize the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = len(set(emojis))  # Number of unique emojis
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Define a function for suggesting an emoji
def suggest_emoji(text, model, tokenizer):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
        input_ids, attention_mask = encoding.input_ids.to(device), encoding.attention_mask.to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        prediction = torch.argmax(logits, dim=-1).item()
        return prediction

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the saved model
model.load_state_dict(torch.load("emoji_suggester.pt"))

# Suggest an emoji for a given text
text = "Anna's birthday"
emoji_label = suggest_emoji(text, model, tokenizer)
print(emoji_label)  # Convert the label back to the corresponding emoji

while True:
    text = input("Enter event title: ")
    emoji_label = suggest_emoji(text, model, tokenizer)
    print(emoji_label)