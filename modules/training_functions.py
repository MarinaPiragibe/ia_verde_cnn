import numpy as np
from sklearn.metrics import accuracy_score
import time
import torch 
from modules import globals

def train(train_loader, model, epoch, optimizer, cross_entropy_function):
  # Training mode
  model.train()

  start = time.time()

  epoch_loss  = []
  # Mude aqui: pred_list e rotulo_list serão listas de valores, não de arrays
  pred_list = []
  rotulo_list = []

  for k, batch in enumerate(train_loader):
    print('\r--{0}/{1}--'.format(k, len(train_loader)), end='', flush=True)
    dado, rotulo = batch

    # Cast do dado na GPU/CPU (use a variável args['device'] do seu contexto)
    dado = dado.to(globals.DEVICE) # Ajuste: use args.device aqui
    rotulo = rotulo.to(globals.DEVICE) # Ajuste: use args.device aqui

    # Forward
    optimizer.zero_grad()
    ypred = model(dado)
    loss = cross_entropy_function(ypred, rotulo)
    epoch_loss.append(loss.cpu().data)

    _, pred = torch.max(ypred, axis=1)

    # Mude aqui: use .extend() para adicionar os elementos individualmente
    pred_list.extend(pred.cpu().numpy())
    rotulo_list.extend(rotulo.cpu().numpy())

    # Backpropagation
    loss.backward()
    optimizer.step()

  epoch_loss = np.asarray(epoch_loss)
  # Agora, pred_list e rotulo_list já são listas de valores,
  # então basta convertê-los para np.array (o .ravel() não é estritamente necessário,
  # mas não causa problema se a lista já é 1D)
  pred_list  = np.asarray(pred_list) # Remova o .ravel() ou mantenha, não fará diferença se já for 1D
  rotulo_list  = np.asarray(rotulo_list) # Remova o .ravel() ou mantenha

  acc = accuracy_score(pred_list, rotulo_list)

  end = time.time()
  print('\n#################### Train ####################')
  print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))

  return epoch_loss.mean()


def validate(test_loader, model, epoch, cross_entropy_function): # Ajustado para receber model
  # Evaluation mode
  model.eval() # Ajustado para model.eval()

  start = time.time()

  epoch_loss  = []
  pred_list = [] # Mude aqui
  rotulo_list = [] # Mude aqui

  with torch.no_grad():
    for k, batch in enumerate(test_loader):

      print('\r--{0}/{1}--'.format(k, len(test_loader)), end='', flush=True)
      dado, rotulo = batch

      # Cast do dado na GPU/CPU (use a variável args['device'] do seu contexto)
      dado = dado.to(globals.DEVICE) # Ajuste: use args.device aqui
      rotulo = rotulo.to(globals.DEVICE) # Ajuste: use args.device aqui

      # Forward
      ypred = model(dado) # Ajustado para model(dado)
      loss = cross_entropy_function(ypred, rotulo)
      epoch_loss.append(loss.cpu().data)

      _, pred = torch.max(ypred, axis=1)
      pred_list.extend(pred.cpu().numpy()) # Mude aqui
      rotulo_list.extend(rotulo.cpu().numpy()) # Mude aqui

  epoch_loss = np.asarray(epoch_loss)
  pred_list  = np.asarray(pred_list) # Mude aqui
  rotulo_list  = np.asarray(rotulo_list) # Mude aqui

  acc = accuracy_score(pred_list, rotulo_list)

  end = time.time()
  print('\n********** Validate **********')
  print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f\n' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))

  return epoch_loss.mean()