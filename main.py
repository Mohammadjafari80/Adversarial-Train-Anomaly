from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from model import FeatureExtractor
from utills import knn_score, test_AUC, auc_softmax, auc_softmax_adversarial, get_data
import yaml
from torchvision.datasets import CIFAR10, MNIST
import numpy as np
import os
import torchvision.transforms as transforms
from pytorch_pretrained_gans import make_gan
from torchattacks import FGSM, PGD
from tqdm import tqdm
from torchvision.utils import save_image
import pandas as pd

#################
#  Load CONFIG  #
#################

config = None
with open('./config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

batch_size = config['batch_size']
normal_class = config['normal_class'] #@param ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']                                      
NUMBER_OF_EPOCHS = config['n_epochs']
attack_params = config['attack_params']
AUC_EVERY = config['auc_every']

if not os.path.exists(config['results_path']):
    os.makedirs(config['results_path'])

#################
#  Set Device   #
#################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#################
#  Load Model   #
#################


model = FeatureExtractor(model=config['backbone'], pretrained=config['pretrained'])
model.to(device)
model.eval()
attack_params['model'] = model

#######################
#  Define Transform   #
#######################

transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)


#######################
#  Prepare Datasets   #
#######################

cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
normal_class_indx = cifar_labels.index(normal_class)

trainset = CIFAR10(root=os.path.join('~', 'cifar10'), train=True, download=True, transform=transform)
trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
trainset.targets  = [0 for t in trainset.targets]
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = CIFAR10(root=os.path.join('~/', 'cifar10'), train=False, download=True, transform=transform)
testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


###############
#  Load GAN   #
###############


G = make_gan(gan_type='biggan', model_name='biggan-deep-128').to(device)


###################
#  Attack Module  #
###################

attack = None

if config['attack_type'] == 'PGD':
    attack = PGD(**attack_params)
else:
    attack = FGSM(**attack_params)

attack.set_mode_targeted_least_likely()


results = {}
try:
    results = pd.read_csv(os.path.join(config['results_path'], f'{config["output_file_name"]}.csv')).to_dict(orient='list')
except:
    results['Softmax AUC'] = []
    results['Softmax Adversairal AUC'] = []
    results['KNN AUC'] = []
    results['KNN Adversairal AUC'] = []
    results['Train Accuracy'] = []
    results['Loss'] = []

##############
#  Training  #
##############

learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(NUMBER_OF_EPOCHS):


  if epoch % AUC_EVERY == 0 :
        torch.cuda.empty_cache()
        model.eval()
        print('*' * 50)
        auc = auc_softmax(model=model, epoch=epoch, test_loader=test_loader, device=device)
        results["Softmax AUC"].append(auc * 100)
        print('*' * 50)
        adv_auc = auc_softmax_adversarial(model=model, epoch=epoch, test_loader=test_loader, attack=attack, device=device)
        results["Softmax Adversairal AUC"].append(adv_auc * 100)
        print('*' * 50)
        auc, adv_auc = test_AUC(model=model, epoch=epoch, train_loader=train_loader, test_loader=test_loader, attack=attack, device=device)
        results["KNN AUC"].append(auc * 100)
        results["KNN Adversairal AUC"].append(adv_auc * 100)
        print('*' * 50)
  else:
        results["Softmax AUC"].append('-')
        results["Softmax Adversairal AUC"].append('-')
        results["KNN AUC"].append('-')
        results["KNN Adversairal AUC"].append('-')

  model.train()
  preds = []
  true_labels = []
  running_loss = 0

  with tqdm(train_loader, unit="batch") as tepoch:
     torch.cuda.empty_cache()
     for i, (data, target) in enumerate(tepoch):
       tepoch.set_description(f"Epoch {epoch + 1}/{NUMBER_OF_EPOCHS}")
       data, target = data.to(device), target.to(device)
       data, target = get_data(model, G, data, target, attack, device)
       optimizer.zero_grad()
       output = model(data)
       predictions = output.argmax(dim=1, keepdim=True).squeeze()
       loss = criterion(output, target)
       true_labels += target.detach().cpu().numpy().tolist()
       preds += predictions.detach().cpu().numpy().tolist()
       correct = (torch.tensor(preds) == torch.tensor(true_labels)).sum().item()
       accuracy = correct / len(preds)
       running_loss += loss.item() * data.size(0)
       loss.backward()
       optimizer.step()
       tepoch.set_postfix(loss=running_loss / len(preds), accuracy=100. * accuracy)

  save_image(tensor=data, fp=os.path.join(config['results_path'], f'sample{epoch:03d}.png'), scale_each=True, normalize=True, nrow=4)
  results["Train Accuracy"].append(100. * accuracy)
  results["Loss"].append(running_loss / len(preds))
  df = pd.DataFrame(results)
  df.to_csv(os.path.join(config['results_path'], f'{config["output_file_name"]}.csv'), index=False)
  print(f'Updated resutls at {os.path.join(config["results_path"])}')

