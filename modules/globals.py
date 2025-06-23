# Altere para o caminho que se encontra o dataset na sua máquina
DATASET_PATH = r'C:\\Users\\masar\\.cache\\kagglehub\\datasets\\birdy654\\cifake-real-and-ai-generated-synthetic-images\\versions\\3\\'

NUM_SAMPLES_TRAIN_DEBUGGER = 10000
NUM_SAMPLES_TEST_DEBUGGER = 4000

CLASS_NAMES = ['FAKE', 'REAL']
IMG_HIGHT = 32
IMG_WIDTH = 32

DEVICE = "cpu"
EPOCHS_FIRST_PHASE = 1
EPOCHS_SECOND_PHASE = 1
LEARNING_RATE_CLASSIFIER = 3e-3
LEARNING_RATE_FEATURES = 3e-5
WEIGHT_DECAY = 8e-4
BATCH_SIZE = 8
