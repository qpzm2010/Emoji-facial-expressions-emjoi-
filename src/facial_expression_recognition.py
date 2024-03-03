import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class FacialExpressionRecognitionModel(nn.Module):
    def __init__(self, num_classes=7):  # 7 是面部表情的类别数
        super(FacialExpressionRecognitionModel, self).__init__()
        # 使用预训练的 ResNet18 模型
        self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        # 替换最后一层全连接层，以适应我们的类别数
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

def load_and_preprocess_image(image_path):
    # 加载图像并进行预处理
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def recognize_facial_expression(model, image_path, class_mapping):
    # 将模型设置为评估模式
    model.eval()

    # 加载和预处理图像
    input_batch = load_and_preprocess_image(image_path)

    # 进行面部表情识别
    with torch.no_grad():
        output = model(input_batch)

    # 获取预测的类别
    _, predicted_idx = torch.max(output, 1)
    predicted_class = class_mapping[predicted_idx.item()]

    return predicted_class

if __name__ == "__main__":
    # 创建面部表情识别模型
    model = FacialExpressionRecognitionModel()

    # 替换成你的预训练权重文件路径
    pretrained_weights_path = 'models/pretrained_emotion_model.pth'
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=torch.device('cpu')))
    model.eval()

    # 替换成你的类别映射字典
    class_mapping = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Surprise", 4: "Neutral", 5: "Disgust", 6: "Fear"}

    # 替换成你的测试图像路径
    test_image_path = 'path/to/your/test/image.jpg'

    # 进行面部表情识别
    predicted_expression = recognize_facial_expression(model, test_image_path, class_mapping)

    # 打印识别结果
    print("Predicted Facial Expression:", predicted_expression)
