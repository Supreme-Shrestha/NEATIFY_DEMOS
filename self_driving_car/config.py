
import os
import pygame
from neatify import EvolutionConfig

# Constants
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
UI_FONT = "Arial"
UI_FONT_SIZE = 24
UI_COLOR = (220, 220, 220)
UI_HIGHLIGHT = (255, 215, 0)
SAVE_INTERVAL_MINUTES = 5
TRAINING_GENERATIONS = 10
LAPS_TO_COMPLETE = 5

# Get absolute path to script directory (to be used by other modules)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Track definitions
TRACKS = {
    "track1": {
        "name": "Track 1",
        "start_pos": (640, 471),
        "model_prefix": "track1",
        "border_color": (255, 255, 255)
    },
    "track2": {
        "name": "Track 2", 
        "start_pos": (412, 295),
        "model_prefix": "track2",
        "border_color": (255, 255, 255)
    },
    "track3": {
        "name": "Track 3",
        "start_pos": (408, 483),
        "model_prefix": "track3",
        "border_color": (255, 255, 255)
    },
    "track4": {
        "name": "Track 4",
        "start_pos": (496, 387),
        "model_prefix": "track4",
        "border_color": (255, 255, 255)
    }
}

DATA_FOLDER = os.path.join(SCRIPT_DIR, 'data')

def create_config():
    config = EvolutionConfig()
    config.population_size = 30
    config.prob_mutate_weight = 0.8
    config.prob_add_connection = 0.3
    config.prob_add_node = 0.1
    config.elitism_count = 5
    config.weight_mutation_power = 0.5
    return config
