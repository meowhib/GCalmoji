import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define a custom dataset class for your event titles and emojis
class EmojiDataset(Dataset):
    def __init__(self, event_titles, emojis, tokenizer):
        self.event_titles = event_titles
        self.emojis = emojis
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.event_titles)

    def __getitem__(self, idx):
        title = self.event_titles[idx]
        emoji = self.emojis[idx]

        encoding = self.tokenizer(title, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
        target = torch.tensor(emoji, dtype=torch.long)

        return encoding.input_ids.squeeze(), encoding.attention_mask.squeeze(), target

# Define a function for training the model
def train(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Define a function for suggesting an emoji
def suggest_emoji(text, model, tokenizer):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
        input_ids, attention_mask = encoding.input_ids.to(device), encoding.attention_mask.to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        prediction = torch.argmax(logits, dim=-1).item()
        return prediction

# Define your event titles and corresponding emojis (as integer labels)
event_titles = [
    "Birthday Party",
    "Wedding Ceremony",
    "Job Interview Work",
    "Dinner Restaurant Reservation",
    "Call",
    "Gym Workout Session"]

emojis = [0, 1, 2, 3, 4, 5]  # Map your emojis to integer labels

# Prepare the dataset and dataloader
dataset = EmojiDataset(event_titles, emojis, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = len(set(emojis))  # Number of unique emojis
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Fine-tune the model
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(model, dataloader, optimizer, device)

# Save the model
torch.save(model.state_dict(), "emoji_suggester.pt")

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