import numpy as np
from sklearn.metrics import accuracy_score
import time
import torch 
from modules import globals

def train(train_loader, model, epoch, optimizer, cross_entropy_function):
  model.train()

  start = time.time()

  epoch_loss  = []
  pred_list = []
  rotulo_list = []

  for k, batch in enumerate(train_loader):
    print(f'{k}/{len(train_loader)}')
    dado, rotulo = batch

    dado = dado.to(globals.DEVICE)
    rotulo = rotulo.to(globals.DEVICE)

    # Forward
    optimizer.zero_grad()
    ypred = model(dado)
    loss = cross_entropy_function(ypred, rotulo)
    epoch_loss.append(loss.cpu().data)

    _, pred = torch.max(ypred, axis=1)

    pred_list.extend(pred.cpu().numpy())
    rotulo_list.extend(rotulo.cpu().numpy())

    # Backpropagation
    loss.backward()
    optimizer.step()

  epoch_loss = np.asarray(epoch_loss)
  pred_list  = np.asarray(pred_list)
  rotulo_list  = np.asarray(rotulo_list)

  acc = accuracy_score(pred_list, rotulo_list)

  end = time.time()
  print('\nTreino')
  print(f'Loss: {epoch_loss.mean()} +/- {epoch_loss.std()}, Acc: {acc*100}, Time: {end-start}')

  return epoch_loss.mean()


def validate(test_loader, model, epoch, cross_entropy_function): 
  model.eval() 

  start = time.time()

  epoch_loss  = []
  pred_list = [] 
  rotulo_list = [] 

  with torch.no_grad():
    for k, batch in enumerate(test_loader):

      print(f'{k}/{len(test_loader)}')
      dado, rotulo = batch

      dado = dado.to(globals.DEVICE)
      rotulo = rotulo.to(globals.DEVICE) 

      # Forward
      ypred = model(dado)
      loss = cross_entropy_function(ypred, rotulo)
      epoch_loss.append(loss.cpu().data)

      _, pred = torch.max(ypred, axis=1)
      pred_list.extend(pred.cpu().numpy())
      rotulo_list.extend(rotulo.cpu().numpy())

  epoch_loss = np.asarray(epoch_loss)
  pred_list  = np.asarray(pred_list)
  rotulo_list  = np.asarray(rotulo_list)

  acc = accuracy_score(pred_list, rotulo_list)

  end = time.time()
  print('\nValidate')
  print(f'Loss: {epoch_loss.mean()} +/- {epoch_loss.std()}, Acc: {acc*100}, Time: {end-start}')

  return epoch_loss.mean()