# ia_verde_cnn

## Autora: Marina Piragibe

A estrutura do projeto é:

- modules:
    - global.py
    - training_function.py
    - utils.py
- old:
    - v1
    - v2
- main.ipynb
- resnet18_cifake_finetuned_float32.pth
- resnet18_cifake_pruned.pth

Na pasta modules estão as funções utilizadas para treinamento e análise dos modelos. Em globals.py estão contidas as definições de hiperparâmetros e preferências de execução. O arquivo training_function.py possui as funções de treino e validação com as métricas de avaliação do modelo. Por fim, utils.py possui funções auxiliares para plotar imagens e realizar cálculos de consumo de memória e tempo de inferência.

A pasta old contém versões antigas dos modelos treinados. Os modelos finais estão salvos na raiz do projeto com nomes no padrão "resnet18_cifake_..."

Toda a lógica de pré-processamento e execução do modelo está detalhada em main.ipynb, assim como análise de métricas e comparações dos modelos criados, além de reflexões e conclusões sobre o projeto.

