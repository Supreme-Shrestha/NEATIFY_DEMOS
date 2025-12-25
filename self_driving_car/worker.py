
import os
import sys

# Set headless mode for pygame BEFORE importing it (inside simulation)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import neatify
from neatify import Worker
from neatify import NeatModule
from simulation import SimulationManager, Car, evaluate_car_fitness
from config import TRACKS

def main():
    parser = argparse.ArgumentParser(description="NEATify Worker Node")
    parser.add_argument("--master", type=str, default="127.0.0.1", help="Master node IP address")
    parser.add_argument("--port", type=int, default=5000, help="Port to connect to")
    parser.add_argument("--auth", type=str, default="secret", help="Authentication key")
    
    args = parser.parse_args()
    
    print(f"Initializing Worker...")
    print(f"Connecting to {args.master}:{args.port}")
    
    # Initialize simulation resources
    sim_manager = SimulationManager()
    
    # We need to know which track to train on. 
    # In a real distributed setup, the config or a separate message should convey the environment ID.
    # For now, let's assume 'track1' or we can select based on config if we had it.
    # We'll use a default and maybe update if we receive context (neatify might support context).
    # Since neatify 0.1.1 might be simple, let's hardcode or pick random track for robustness?
    # Better: The fitness function receives the genome.
    
    current_track = "track1" 
    track_surface = sim_manager.get_track_data(current_track)
    
    def fitness_function(genome):
        # Create network
        net = NeatModule(genome)
        
        # Create car
        car = Car(current_track, track_surface)
        
        # Evaluate
        fitness = evaluate_car_fitness(net, car, max_frames=2000)
        return fitness

    # Initialize Worker
    # Note: Worker implementation in neatify likely connects and waits for tasks.
    # If neatify.Worker expects a fitness function to be registered or just runs:
    try:
        worker = Worker(
            master_address=args.master, 
            port=args.port,
            auth_key=args.auth.encode('utf-8')
        )
        worker.start(fitness_function)
    except KeyboardInterrupt:
        print("Worker stopped.")
    except Exception as e:
        print(f"Worker error: {e}")

if __name__ == "__main__":
    main()
