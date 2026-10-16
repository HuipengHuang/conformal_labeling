import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import argparse
import pandas as pd
from torchvision.models import ResNet34_Weights, DenseNet161_Weights, ResNeXt50_32X4D_Weights

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--path", type=str, help="Path to Imagenet dataset")
parser.add_argument("--model", type=str, default="resnet34")
args = parser.parse_args()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = args.model

if model_name == "resnet34":
    model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device).eval()
elif model_name == "densenet161":
    model = models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1).to(device).eval()
elif model_name == "resnext50":
    model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1).to(device).eval()
else:
    raise NotImplementedError(f"Model {model_name} not supported")

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


test_dataset = torchvision.datasets.ImageFolder(
    root=args.path,
    transform=test_transform)


dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

# Initialize lists to store results
all_confidences = []
all_y_hat = []
all_y_true = []

with torch.no_grad():
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        logits = model(data)
        prob = torch.softmax(logits, dim=-1)
        y_hat = torch.argmax(prob, dim=-1)
        conf = prob[torch.arange(prob.size(0)), y_hat]

        # Store results
        all_confidences.extend(conf.cpu().numpy())
        all_y_hat.extend(y_hat.cpu().numpy())

        # Remap targets if needed
        batch_targets = target.cpu().numpy()
        all_y_true.extend(batch_targets)

# Create DataFrame with correct labels
df = pd.DataFrame({
    'Y': all_y_true,
    'Yhat': all_y_hat,
    'confidence': all_confidences,
})

# Save to CSV
output_file = f'datasets/{args.model}_imagenet.csv'
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
