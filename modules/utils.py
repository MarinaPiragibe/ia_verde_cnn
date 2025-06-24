import time
from matplotlib import pyplot as plt
import torch
from modules import globals
import torchvision.models as models
from torch import nn
import seaborn

def plot_sample(train_loader, data_type):
    plt.figure(figsize=(10, 5))
    dataiter = iter(train_loader)
    images, labels = next(dataiter) 

    for i in range(min(len(images), 9)):
        plt.subplot(1, 9, i + 1)
        plt.imshow(images[i].numpy().transpose((1, 2, 0)))
        plt.title(globals.CLASS_NAMES[labels[i]], fontsize=10)
        plt.axis("off")
    plt.suptitle(f"CIFAKE {data_type} Images", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 1.5])
    plt.show()

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(8, 6))
    seaborn.heatmap(confusion_matrix,
                annot=True,      
                fmt='d',         
                cmap='Blues',    
                cbar=True,
                linewidths=0.5,  
                linecolor='black'
            )
    plt.title('Matriz de Confusão', fontsize=16)
    plt.xticks(rotation=45, ha='right') 
    plt.yticks(rotation=0)             
    plt.tight_layout()
    plt.show() 

def measure_inference_time(model, dummy_input, num_warmup=10, num_runs=100):
    model.eval()
    
    dummy_input = dummy_input.to(next(model.parameters()).device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)

    mean_time_ms = (sum(timings) / num_runs) * 1000
    return mean_time_ms

def measure_model_memory_usage(model):
    mem_bytes = sum(parameter.element_size() * parameter.nelement() for parameter in model.parameters())
    mem_mb = mem_bytes / (1024 * 1024)
    return mem_mb

def load_model_from_file(model_path):
    model = models.resnet18(pretrained=True)
    num_in_features = model.fc.in_features
    model.fc = nn.Linear(num_in_features, 2)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(globals.DEVICE)
        model.eval()
        print(f"\nModelo '{model_path}' carregado para avaliação de desempenho.")
        return model
    except FileNotFoundError:
        print(f"ERRO: Modelo '{model_path}' não encontrado. Não será possível comparar.")
        model = None
        return model
    
def measure_sparcity(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            sparsity = 100 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
            print(f"Sparsity in {name}: {sparsity:.2f}%")