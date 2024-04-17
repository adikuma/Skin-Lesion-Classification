import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

loc_time = time.strftime("%H%M%S", time.localtime())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ratio = 8


class FixCapsNet(nn.Module):
    def __init__(self,conv_inputs,conv_outputs,
                 primary_units,primary_unit_size,
                 output_unit_size,num_classes=7,
                 init_weights=False,mode="DS"):
        super().__init__()

        self.Convolution = make_features(cfgs[mode],f_c=conv_inputs,out_c=conv_outputs)

        self.CBAM = Conv_CBAM(conv_outputs,conv_outputs)

        self.primary = Primary_Caps(in_channels=conv_outputs,#128
                                    caps_units=primary_units,#8
                                    )

        self.digits = Digits_Caps(in_units=primary_units,#8
                                   in_channels=primary_unit_size,#16*6*6=576
                                   num_units=num_classes,#classification_num
                                   unit_size=output_unit_size,#16
                                   )
        if init_weights:
            self._initialize_weights()
        self.classifier = nn.Linear(num_classes * output_unit_size, num_classes)  # Assuming 'num_classes' is the final output dimension

    def forward(self, x):
        x = self.Convolution(x)
        x = self.CBAM(x)
        x = self.primary(x)
        x = self.digits(x)
        # print("Shape after digits:", x.shape)  # Debugging output
        x = x.view(x.size(0), -1)  # Flatten the outputs
        # print("Shape before classifier:", x.shape)  # Debugging output
        x = self.classifier(x)  # Classify to the final number of classes
        return x



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    #margin_loss
    def loss(self, img_input, target, size_average=True):
        batch_size = img_input.size(0)
        # ||vc|| from the paper.
        v_mag = torch.sqrt(torch.sum(img_input**2, dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).to(device)
        m_plus, m_minus = 0.9, 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2
        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = torch.sum(L_c,1)

        if size_average:
            L_c = torch.mean(L_c)

        return L_c

class Primary_Caps(nn.Module):
    def __init__(self, in_channels, caps_units):
        super(Primary_Caps, self).__init__()

        self.in_channels = in_channels
        self.caps_units = caps_units

        def create_conv_unit(unit_idx):
            unit = ConvUnit(in_channels=in_channels)
            self.add_module("Caps_" + str(unit_idx), unit)
            return unit
        self.units = [create_conv_unit(i) for i in range(self.caps_units)]

    #no_routing
    def forward(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).
        u = [self.units[i](x) for i in range(self.caps_units)]
        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1)
        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.caps_units, -1)
        # Return squashed outputs.
        return squash(u)

class Digits_Caps(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size):
        super(Digits_Caps, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.unit_size = unit_size  # Define this properly if it's supposed to be accessible

        self.W = nn.Parameter(torch.randn(1, in_channels, self.num_units, unit_size, in_units))

    #routing
    def forward(self, x):
        batch_size = x.size(0)
        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)
        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)
        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x)
        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1)).to(device)

        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            #c_ij = F.softmax(b_ij, dim=0)
            c_ij = b_ij.softmax(dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            # s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            s_j = torch.sum(c_ij * u_hat, dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = squash(s_j)#CapsuleLayer.squash

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()
        Caps_out = in_channels // ratio
        self.Cpas = nn.Sequential(
                        nn.Conv2d(in_channels,Caps_out,9,stride=2,groups=Caps_out, bias=False),
                    )

    def forward(self, x):
        output = self.Cpas(x)
        return output

def squash(s):
    mag_sq = torch.sum(s**2, dim=2, keepdim=True)
    mag = torch.sqrt(mag_sq)
    s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return s

class Conv_CBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv_CBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)#LayerNorm(c2, eps=1e-6, data_format="channels_first")#
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = ChannelAttention(c2, reduction=1)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size,padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# CAM
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        me_c = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(channels, me_c, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(me_c, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def make_features(cfg: list,f_c,out_c=None,g=1,step=2):
    layers = []
    output = out_c
    f_channels = f_c
    for i in range(len(cfg)):
        if cfg[i] == 'N':
            g = 3

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(2, 2)]
        elif v == "A":
            layers += [nn.AdaptiveMaxPool2d(20)]
        elif v == "F":
            layers += [nn.FractionalMaxPool2d(2, output_size=(20,20))]
        elif v == "B":
            f_channels = out_c
            layers += [nn.BatchNorm2d(f_channels,affine=True)]
            # layers += [LayerNorm(f_channels, eps=1e-6, data_format="channels_first")]
        elif v == "R":
            layers += [nn.ReLU(inplace=True)]
        elif v == "N":
            layers += [nn.Conv2d(f_channels,out_c,1,stride=step)]
        elif v == "C":
            layers += [nn.Conv2d(f_channels,f_channels,3,stride=step)]
        else:
            layers += [nn.Conv2d(f_channels, v, 18,stride=step,groups=g)]
            f_channels = v
    return nn.Sequential(*layers)

cfgs= {
    "DS": [3,'N','B','R','F'],# g = 3,  primary_unit_size = 16 * 6 * 6
    "DS2": ["C",3,'N','B','R','F'],# g = 3,  primary_unit_size = 16 * 6 * 6
    "256" : [256,'R','F'],# g = 1,  primary_unit_size = 32 * 6 * 6
    "128" : [128,'R','F'],# g = 1, primary_unit_size = 16 * 6 * 6
    "64"  : [64,'R','F'],# g = 1 , primary_unit_size = 8 * 6 * 6

}


class MetadataModel(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super(MetadataModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc_layers(x)
    

class AdvancedClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes):
        super(AdvancedClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.residual = nn.Linear(input_dim, 256)
        self.final = nn.Linear(256, num_classes)

    def forward(self, x):
        out1 = self.layer1(x)
        res_out = self.residual(x)
        combined = out1 + res_out
        output = self.final(combined)
        return output

class FusionModel(nn.Module):
    def __init__(self, image_model, metadata_model, num_classes=7):
        super(FusionModel, self).__init__()
        self.image_model = image_model
        self.metadata_model = metadata_model
        # Define fixed sizes for output features of image and metadata models
        image_output_size = 7  # this needs to be corrected based on the actual output size of the image_model
        metadata_output_size = 64  # from MetadataModel
        self.classifier = AdvancedClassifier(image_output_size + metadata_output_size, 128, num_classes)

    def forward(self, image, metadata):
        image_features = self.image_model(image)
        metadata_features = self.metadata_model(metadata)
        # print("Image features size:", image_features.shape)  # Debug output size
        # print("Metadata features size:", metadata_features.shape)  # Debug output size
        combined_features = torch.cat((image_features, metadata_features), dim=1)
        # print("Combined features size:", combined_features.shape)  # Debug output size
        output = self.classifier(combined_features)
        return output
    

n_channels = 3
n_classes = 7
conv_outputs = 128 
num_primary_units = 8
primary_unit_size = 16 * 6 * 6
output_unit_size = 16
img_size = 299
mode='128'
image_model = FixCapsNet(conv_inputs=n_channels,
                     conv_outputs=conv_outputs,
                     primary_units=num_primary_units,
                     primary_unit_size=primary_unit_size,
                     num_classes=n_classes,
                     output_unit_size=16,
                     init_weights=True,
                     mode=mode)
image_model.to('cuda')


num_metadata_features = 3
metadata_model = MetadataModel(input_dim=num_metadata_features)

fusion_model = FusionModel(
    image_model=image_model,
    metadata_model=metadata_model,
    num_classes=n_classes
)


mapping = {'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'bcc': 4, 'akiec': 5, 'vasc': 6}
class_names = ['Benign Keratosis-like Lesions (bkl)',
               'Melanocytic Nevi (nv)',
               'Dermatofibroma (df)',
               'Melanoma (mel)',
               'Basal Cell Carcinoma (bcc)',
               'Actinic Keratoses and Intraepithelial Carcinoma / Bowens disease (akiec)',
               'Vascular Lesions (vasc)']


def mask_and_predict(image, model, metadata, device):
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)
    else:
        masked_image = image_np

    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(masked_image)
    image = Image.fromarray(image_np)
    image_transform = transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    metadata_tensor = torch.tensor([metadata], dtype=torch.float32).to(device)

    with torch.set_grad_enabled(True):
        output = model(image_tensor, metadata_tensor)
        pred_probabilities = torch.nn.functional.softmax(output, dim=1)
        max_prob, pred_index = torch.max(pred_probabilities, dim=1)
        max_prob = max_prob.item() * 100  
    
    if max_prob < 40:
        pred_class_name = "Healthy Skin"
    else:
        pred_class_name = class_names[pred_index.item()]

    return pil_image, pred_class_name, output, pred_index, max_prob


fusion_model.load_state_dict(torch.load('best_model_65.pth'))
fusion_model.to(device)
fusion_model.eval()  

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

cbam_layer = fusion_model.image_model.CBAM.conv
forward_handle = cbam_layer.register_forward_hook(forward_hook)
backward_handle = cbam_layer.register_backward_hook(backward_hook)

st.title('Skin Lesion Detector with Metadata')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
age = st.slider("Age", 0, 150, 25)
gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
localization = st.selectbox("Localization", options=["Neck", "Head", "Arm"])

gender_dict = {"Male": 0, "Female": 1, "Other": 2}
localization_dict = {"Neck": 0, "Head": 1, "Arm": 2}
metadata = [age, gender_dict[gender], localization_dict[localization]]

if uploaded_file and st.button('Analyze Lesion'):
    image = Image.open(uploaded_file)
    masked_image, lesion_type, output, pred_index, max_prob = mask_and_predict(image, fusion_model, metadata, 'cuda')
    st.title(f'Predicted Lesion Type: {lesion_type} - Confidence: {max_prob:.2f}%')
    st.image(masked_image, caption='Masked Lesion Image', use_column_width=True)

    if lesion_type != "Healthy Skin":
        output[0, pred_index].backward()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)
        heatmap /= torch.max(heatmap)

        heatmap_np = heatmap.cpu().numpy()
        plt.imshow(heatmap_np, cmap='viridis')
        plt.axis('off')
        st.pyplot(plt)

    forward_handle.remove()
    backward_handle.remove()
