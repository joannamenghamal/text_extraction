import os
import zipfile
import io
import pytesseract
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import time

# Tesseract setup for OCR
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Adjust based on your `which tesseract` output

# Preprocess image for better OCR results
def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy.
    :param image: PIL Image object.
    :return: Preprocessed PIL Image object.
    """
    # Convert to grayscale
    image = image.convert('L')
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Resize image
    image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)  # Updated here
    # Denoise image
    image = image.filter(ImageFilter.MedianFilter())
    return image

# Extract text from an image with preprocessing
def extract_text(image_path):
    image = Image.open(image_path)
    preprocessed_image = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    return clean_text(text)

# Clean extracted text
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = text.replace('0', 'O').replace('1', 'I')  # Replace common OCR errors
    return text

# Custom dataset for receipt images and their corresponding ground truth text
class ReceiptDataset(Dataset):
    def __init__(self, zip_path, ground_truth_csv, transform=None):
        """
        Custom dataset for loading receipt images and their corresponding ground truth text.
        :param zip_path: Path to the zip file containing receipt images.
        :param ground_truth_csv: Path to the CSV containing filenames and ground truth text.
        :param transform: Optional transform to be applied on a sample.
        """
        self.zip_path = zip_path
        self.ground_truth_df = pd.read_csv(ground_truth_csv)
        self.transform = transform
        self.image_filenames = self.ground_truth_df['Filename'].tolist()
        self.ground_truth_texts = self.ground_truth_df['Ground Truth Text'].tolist()

        # Open zip file to extract image files
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            self.image_files = [f for f in zip_ref.namelist() if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        ground_truth = self.ground_truth_texts[idx]

        # Extract image from zip file
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open(filename) as file:
                image = Image.open(io.BytesIO(file.read()))

        if self.transform:
            image = self.transform(image)

        return image, ground_truth


# CNN model to classify text (as labels)
class CNN_Text_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Text_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                           # 28x28 -> 14x14
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)  # Output size is num_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))      
        x = F.relu(self.conv2(x))      
        x = self.pool(x)               
        x = self.dropout(x)
        x = x.view(-1, 64 * 14 * 14)   
        x = F.relu(self.fc1(x))        
        x = self.dropout(x)
        x = F.relu(self.fc2(x))        
        x = self.fc3(x)  # Outputs the logits
        return F.log_softmax(x, dim=1)


# Train the model using the custom dataset
def train_model(zip_path, ground_truth_csv, epochs=5):
    # Step 1: Prepare the dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = ReceiptDataset(zip_path=zip_path, ground_truth_csv=ground_truth_csv, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 2: Encode labels (Ground Truth Texts)
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset.ground_truth_texts)
    num_classes = len(label_encoder.classes_)

    # Step 3: Check for GPU and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 4: Initialize model, loss, and optimizer
    model = CNN_Text_Model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 5: Train the model
    start_time = time.time()  # Record the start time
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, texts in train_loader:
            images, labels = images.to(device), torch.tensor(label_encoder.transform(texts)).to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        # Print loss and accuracy for each epoch
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%, Time: {time.time() - start_time:.2f} seconds")
    #Print total accuracy for all epochs and time
    print(f"Total Training Accuracy: {100 * correct / total:.2f}%")
    print(f"Total Training Time: {time.time() - start_time:.2f} seconds")
        


# Extract text from a zip file and extract information from receipt image files
def extract_text_from_zip(zip_path):
    """
    Extracts text from images in a zip file.
    :param zip_path: Path to the zip file.
    :return: Dictionary with filenames as keys and extracted text as values.
    """
    extracted_text = {}
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        with zip_ref.open(file_info) as file:
                            image_data = Image.open(io.BytesIO(file.read()))
                            preprocessed_image = preprocess_image(image_data)
                            custom_config = r'--oem 3 --psm 6'
                            text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
                            text = clean_text(text)
                            extracted_text[file_info.filename] = text
                    except Exception as img_error:
                        print(f"Error processing file {file_info.filename}: {img_error}")
    except Exception as e:
        print(f"Error extracting text from zip: {e}")
    return extracted_text


# Save extracted text to a CSV file
def save_text_to_csv(extracted_text, output_csv):
    """
    Saves extracted text to a CSV file.
    :param extracted_text: Dictionary with filenames as keys and extracted text as values.
    :param output_csv: Path to the output CSV file.
    """
    try:
        df = pd.DataFrame(list(extracted_text.items()), columns=['Filename', 'Extracted Text'])
        df.to_csv(output_csv, index=False)
        print(f"Extracted text saved to {output_csv}")
    except Exception as e:
        print(f"Error saving text to CSV: {e}")


# Create ground truth template CSV
def create_ground_truth_template(zip_path, ground_truth_csv):
    """
    Creates a ground truth CSV template based on image filenames in the zip file.
    :param zip_path: Path to the zip file.
    :param ground_truth_csv: Path to save the ground truth CSV.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            image_filenames = [f for f in zip_ref.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]

            # Create a dataframe with filenames and placeholder for ground truth text
            df = pd.DataFrame(image_filenames, columns=['Filename'])
            df['Ground Truth Text'] = 'Expected text for ' + df['Filename']
            df.to_csv(ground_truth_csv, index=False)
            print(f"Ground truth template created: {ground_truth_csv}")
    except Exception as e:
        print(f"Error creating ground truth template: {e}")


# Main function to extract text from zip and save to CSV
def main():
    zip_path = 'L05_DL_Vision_Receipts.zip'
    ground_truth_csv = 'ground_truth.csv'  # Output CSV file path

    # Step 1: Extract text from zip file and create a ground truth CSV
    extracted_text = extract_text_from_zip(zip_path)
    if extracted_text:
        save_text_to_csv(extracted_text, 'extracted_text.csv')

        # Step 2: Create the ground truth template if not present
        if not os.path.exists(ground_truth_csv):
            create_ground_truth_template(zip_path, ground_truth_csv)

        # Step 3: Train the CNN model
        train_model(zip_path, ground_truth_csv, epochs=20)
    else:
        print("No text extracted from zip file.")


if __name__ == "__main__":
    main()
