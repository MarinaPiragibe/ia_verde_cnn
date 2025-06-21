import torch
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from modules import globals

def train(train_loader, model, optimizer, cross_entropy_function):
    model.train()

    start = time.time()

    epoch_loss = []
    pred_list = []
    rotulo_list = []

    for k, batch in enumerate(train_loader):
        print(f'\r{k+1}/{len(train_loader)}', end='', flush=True) 

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

    final_pred_array = np.asarray(pred_list)
    final_rotulo_array = np.asarray(rotulo_list)

    accuracy = accuracy_score(final_rotulo_array, final_pred_array)
    precision = precision_score(final_rotulo_array, final_pred_array, average='binary', zero_division=0)
    recall = recall_score(final_rotulo_array, final_pred_array, average='binary', zero_division=0)
    f1 = f1_score(final_rotulo_array, final_pred_array, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(final_rotulo_array, final_pred_array, labels=[0, 1])

    end = time.time()
    print('\nTreino')
    print(f'Loss: {epoch_loss.mean():.4f} +/- {epoch_loss.std():.4f}, Acc: {accuracy*100:.2f}, Time: {end-start:.2f}')

    return accuracy, precision, recall, f1, conf_matrix, epoch_loss.mean()

# acc_train, prec_train, rec_train, f1_train, cm_train, loss_train = train(train_loader, resnet_model, epoch, optimizer, criterion)

def validate(test_loader, model, cross_entropy_function):
    model.eval() 
    start = time.time()

    epoch_loss = []
    pred_list = []
    rotulo_list = []

    with torch.no_grad(): # Desativa o cálculo de gradientes
        for k, batch in enumerate(test_loader):
            print(f'\r{k+1}/{len(test_loader)}', end='', flush=True) 

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

    final_pred_array = np.asarray(pred_list)
    final_rotulo_array = np.asarray(rotulo_list)

    accuracy = accuracy_score(final_rotulo_array, final_pred_array)
    precision = precision_score(final_rotulo_array, final_pred_array, average='binary', zero_division=0)
    recall = recall_score(final_rotulo_array, final_pred_array, average='binary', zero_division=0)
    f1 = f1_score(final_rotulo_array, final_pred_array, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(final_rotulo_array, final_pred_array, labels=[0, 1])

    end = time.time()
    print('\nValidação')
    print(f'Loss: {epoch_loss.mean():.4f} +/- {epoch_loss.std():.4f}, Acc: {accuracy*100:.2f}, Time: {end-start:.2f}\n')

    return accuracy, precision, recall, f1, conf_matrix, epoch_loss.mean()

# acc_val, prec_val, rec_val, f1_val, cm_val, loss_val = validate(test_loader, resnet_model, epoch, criterion)