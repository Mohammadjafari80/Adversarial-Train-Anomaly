import numpy as np
import faiss
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm
import torchvision.transforms as transforms

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def test_AUC(model, epoch, train_loader, test_loader, attack, device):
  
    train_features = []
    k = 2
    test_distances = []
    test_distances_adv = []
    test_labels = []
    model.eval()

    print('Test Started ...')

    ###########################################################################

    print('Extracting Train Features...')

    with torch.no_grad():
      with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            inputs = inputs.to(device)
            train_features += model.get_feature_vector(inputs).detach().cpu().numpy().tolist()

    print('Extracting Train Features Finished...')
    # features_tensor = torch.Tensor(train_features).to(device)
    train_features = np.array(train_features).astype(np.float32)
    ###########################################################################

    print('Extracting Test Features...')

    test_features = []
    adv_test_features = []

    
    with tqdm(test_loader, unit="batch") as tepoch:
      for inputs, labels in tepoch:
          inputs = inputs.to(device)
          labels = labels.to(device)
          test_features += model.get_feature_vector(inputs).detach().cpu().numpy().tolist()
          adv_inputs = attack(inputs, labels)
          adv_test_features += model.get_feature_vector(adv_inputs).detach().cpu().numpy().tolist()
          test_labels += labels.detach().cpu().numpy().tolist()

    test_features = np.array(test_features).astype(np.float32)
    adv_test_features = np.array(adv_test_features).astype(np.float32)

    test_distances = knn_score(train_features, test_features)
    test_distances_adv = knn_score(train_features, adv_test_features)


    print('Extracting Test Features Finished...')

    auc = roc_auc_score(test_labels, test_distances)
    adv_auc = roc_auc_score(test_labels, test_distances_adv)

    print(f'AUC score on epoch {epoch} is: {auc * 100}')
    print(f'AUC Adversairal score on epoch {epoch} is: {adv_auc * 100}')

    return auc, adv_auc

def auc_softmax(model, epoch, test_loader, device):
  soft = torch.nn.Softmax(dim=1)
  anomaly_scores = []
  test_labels = []
  print('AUC Softmax Started ...')
  with torch.no_grad():
    with tqdm(test_loader, unit="batch") as tepoch:
      torch.cuda.empty_cache()
      for i, (data, target) in enumerate(tepoch):
        data, target = data.to(device), target.to(device)
        output = model(data)
        probs = soft(output).squeeze()
        anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()
        test_labels += target.detach().cpu().numpy().tolist()

  auc = roc_auc_score(test_labels, anomaly_scores)
  print(f'AUC - Softmax - score on epoch {epoch} is: {auc * 100}')
  return auc



def auc_softmax_adversarial(model, epoch, test_loader, attack, device):
  soft = torch.nn.Softmax(dim=1)
  anomaly_scores = []
  test_labels = []
  print('AUC Adversarial Softmax Started ...')
  with tqdm(test_loader, unit="batch") as tepoch:
    torch.cuda.empty_cache()
    for i, (data, target) in enumerate(tepoch):
      data, target = data.to(device), target.to(device)
      labels = target.to(device)
      adv_data = attack(data, target)
      output = model(adv_data)
      probs = soft(output).squeeze()
      anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()
      test_labels += labels.detach().cpu().numpy().tolist()

  auc = roc_auc_score(test_labels, anomaly_scores)
  print(f'AUC Adversairal - Softmax - score on epoch {epoch} is: {auc * 100}')
  return auc

  
def get_data(model, G, data, target, attack, device):
    model.eval()
    y = G.sample_class(batch_size=data.shape[0])
    z = G.sample_latent(batch_size=data.shape[0])  
    x = G(z=z.to(device), y=y.to(device)).to(device)
    x = transforms.Resize((32, 32))(x)

    fake_data = (x - torch.min(x))/(torch.max(x)- torch.min(x))
    fake_target = torch.ones_like(target, device=device)

    new_data = torch.cat((fake_data, data))
    new_target = torch.cat((fake_target, target))

    adv_data = attack(new_data, new_target)

    final_data = torch.cat((new_data, adv_data))
    final_target = torch.cat((new_target, new_target))
    model.train()
    return final_data, final_target