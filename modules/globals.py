# Altere para o caminho que se encontra o dataset na sua máquina
DATASET_PATH = r'C:\\Users\\masar\\.cache\\kagglehub\\datasets\\birdy654\\cifake-real-and-ai-generated-synthetic-images\\versions\\3\\'

NUM_SAMPLES_TRAIN_DEBUGGER = 100
NUM_SAMPLES_TEST_DEBUGGER = 50

CLASS_NAMES = ['FAKE', 'REAL']
IMG_HIGHT = 32
IMG_WIDTH = 32

DEVICE = "cpu"
EPOCHS_FIRST_PHASE = 2
EPOCHS_SECOND_PHASE = 3
LEARNING_RATE_CLASSIFIER = 1e-3
LEARNING_RATE_FEATURES = 1e-5
WEIGHT_DECAY = 8e-4
BATCH_SIZE = 30
