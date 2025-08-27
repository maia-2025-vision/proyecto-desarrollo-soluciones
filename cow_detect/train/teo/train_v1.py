import torch
from typer import Typer

import torchvision
from torchvision.transforms import functional as F
from transformers import DetrForObjectDetection, DetrImageProcessor



def train():
# Set up the model, optimizer, and dataset
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Use DetrForObjectDetection from Hugging Face
    model_name = "facebook/detr-resnet-50"
    image_processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 10

# Create the dataset and dataloader
root_dir = './'  # Assuming images/ and annotations/ are in the current directory
dataset = CowDataset(root_dir=root_dir, image_processor=image_processor)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

# Fine-tuning loop
model.train()
for epoch in range(num_epochs):
    for i, batch in enumerate(data_loader):
        pixel_values = batch['pixel_values'].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

        outputs = model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

# Save the fine-tuned model
torch.save(model.state_dict(), 'detr_cow_detection.pth')
print("Fine-tuning complete. Model saved.")