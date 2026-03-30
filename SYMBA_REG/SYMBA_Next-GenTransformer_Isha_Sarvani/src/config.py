import os

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, 'data')
WEIGHTS_DIR = os.path.join(ROOT, 'weights')
PLOTS_DIR   = os.path.join(ROOT, 'plots')

REAL_LABEL  = os.path.join(DATA_DIR, 'task1_1_tokenized_postfix.json')
REAL_CLOUD  = os.path.join(DATA_DIR, 'clouds', 'isha_data_clouds.json')
VOCAB_PATH  = os.path.join(DATA_DIR, 'vocab',  'vocab.json')
ENC_PATH    = os.path.join(WEIGHTS_DIR, 'encoder.pth')
DEC_PATH    = os.path.join(WEIGHTS_DIR, 'decoder.pth')
PLOT_PATH   = os.path.join(PLOTS_DIR,   'training.png')

MAX_D       = 10
N_SAMPLE    = 50

EMBED_DIM   = 96
GPT_DIM     = 192
N_LAYERS    = 2
N_HEADS     = 4
DROPOUT     = 0.20

N_SYNTH     = 10_000
N_POINTS    = 200

SYN_EPOCHS  = 60
SYN_LR      = 3e-4
SYN_BATCH   = 32
SYN_PATIENCE= 15

FT_EPOCHS   = 120
FT_LR       = 1e-4
FT_BATCH    = 8
FT_PATIENCE = 30

BEAM_SIZE      = 16

REFINER_EPOCHS = 50
REFINER_LR     = 3e-4
REFINER_MASK   = 0.15
REFINER_PASSES = 3

BFGS_RESTARTS  = 8
BFGS_FIT_PTS   = 150

GP_POP         = 100
GP_GENS        = 80
GP_SEED_RATIO  = 0.4
GP_TOURNAMENT  = 5
GP_CROSSOVER   = 0.7
GP_MUTATION    = 0.3
GP_ELITISM     = 5
