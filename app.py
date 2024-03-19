import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        # print(x.shape, '1st')
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape, '2nd')
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape, '3rd')
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape, '4th')
        x = x.view(-1, 64 * 37 * 37)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
model = CNN()
# model = models.inception_v3(pretrained=False) 
# for param in model.parameters():
#     param.requires_grad = False  

# model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
# model.fc = nn.Sequential(
#     nn.Flatten(), 
#     nn.Linear(2048, 128),
#     nn.ReLU(), 
#     nn.Dropout(0.1), 
#     nn.Linear(128, 5)
# )
# model.aux_logits = False
model = model.to(device)
model_path = 'cnn_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
mapping = {0: 'bkl', 1: 'nv', 2: 'df', 3: 'mel', 4: 'bcc'}
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        print(mapping.get(predicted.item()))
        return mapping[predicted.item()]

st.title('Skin Lesion Classifier')
uploaded_file = st.file_uploader("Choose an image dummy!", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    label = predict(image)
    st.write(f"The predicted type of skin lesion is: {label}")
