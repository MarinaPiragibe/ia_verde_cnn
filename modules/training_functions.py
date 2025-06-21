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
    print('\r--{0}/{1}--'.format(k, len(train_loader)), end='', flush=True)
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
  print('\n#################### Train ####################')
  print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))

  return epoch_loss.mean()


def validate(test_loader, model, epoch, cross_entropy_function): 
  model.eval() 

  start = time.time()

  epoch_loss  = []
  pred_list = [] 
  rotulo_list = [] 

  with torch.no_grad():
    for k, batch in enumerate(test_loader):

      print('\r--{0}/{1}--'.format(k, len(test_loader)), end='', flush=True)
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
  print('\n********** Validate **********')
  print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f\n' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))

  return epoch_loss.mean()