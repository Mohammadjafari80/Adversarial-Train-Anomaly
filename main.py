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
from pytorch_pretrained_gans import make_gan
from torchattacks import FGSM, PGD
import logging
from tqdm import tqdm
from torchvision.utils import save_image
import pandas as pd
from datasets import Exposure, get_dataloader


#################
#  Load CONFIG  #
#################

config = None
with open('./config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

batch_size = config['batch_size']
dataset = config['dataset']
normal_class_indx = int(config['normal_class_indx'])                              
NUMBER_OF_EPOCHS = config['n_epochs']
attack_params = config['attack_params']
AUC_EVERY = config['auc_every']

if not os.path.exists(config['results_path']):
    os.makedirs(config['results_path'])


#############
#  LOGGING  #
#############

import logging

logging.basicConfig(filename=os.path.join(config['results_path'], 'app.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


#################
#  Set Device   #
#################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device : {device}')

#################
#  Load Model   #
#################

model = FeatureExtractor(model=config['backbone'], pretrained=config['pretrained'])
model.to(device)
model.eval()
attack_params['model'] = model



#######################
#  Prepare Datasets   #
#######################

train_loader, test_loader = get_dataloader(dataset, normal_class_indx, batch_size)
logger.info(f'Dataset: {dataset} - Normal-Class-Index: {normal_class_indx}')


####################
#  Load Exposure   #
####################

G, exposure_dataset, exposure_loader = None, None, None

if config['use_gan']:
    G = make_gan(gan_type='biggan', model_name='biggan-deep-128').to(device)
else:
    exposure_dataset = Exposure(root=config['exposure_folder'])
    exposure_loader = torch.utils.data.DataLoader(exposure_dataset, shuffle=True, batch_size=batch_size)


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
        if config['knn_attack']:
            auc, adv_auc = test_AUC(model=model, epoch=epoch, train_loader=train_loader, test_loader=test_loader, attack=attack, device=device, attack_type=config['attack_type'])
            results["KNN AUC"].append(auc * 100)
            results["KNN Adversairal AUC"].append(adv_auc * 100)
            print('*' * 50)
        else:
            results["KNN AUC"].append('-')
            results["KNN Adversairal AUC"].append('-')

  else:
        results["Softmax AUC"].append('-')
        results["Softmax Adversairal AUC"].append('-')
        results["KNN AUC"].append('-')
        results["KNN Adversairal AUC"].append('-')

  model.train()
  preds = []
  true_labels = []
  running_loss = 0
  
  first_batch = None

  with tqdm(train_loader, unit="batch") as tepoch:
     torch.cuda.empty_cache()
     for i, (data, target) in enumerate(tepoch):
       tepoch.set_description(f"Epoch {epoch + 1}/{NUMBER_OF_EPOCHS}")
       data, target = data.to(device), target.to(device)
       data, target = get_data(config['use_gan'], model, exposure_loader, G, data, target, attack, device)
       target = target.type(torch.LongTensor).cuda()
       if i == 0:
          first_batch = data.detach().clone()
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

  image_shape = first_batch.shape[0]
  normal_images, adversarial_images = first_batch[:image_shape//2], first_batch[image_shape//2:]
  save_image(tensor=torch.cat((first_batch, normal_images - adversarial_images)), fp=os.path.join(config['results_path'], f'sample{epoch:03d}.png'), scale_each=True, normalize=True, nrow=image_shape//2)
  results["Train Accuracy"].append(100. * accuracy)
  results["Loss"].append(running_loss / len(preds))
  df = pd.DataFrame(results)
  df.to_csv(os.path.join(config['results_path'], f'{config["output_file_name"]}.csv'), index=False)
  print(f'Updated resutls at {os.path.join(config["results_path"])}')
  logger.info(f'Updated resutls at {os.path.join(config["results_path"])}')

