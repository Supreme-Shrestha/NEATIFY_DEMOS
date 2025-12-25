
import time
import argparse
from neatify import DistributedPopulation
from neatify.distributed.config import DistributedConfig
from config import create_config, TRACKS
# SAVED_GENOME_PREFIX = "best_genome"

def main():
    parser = argparse.ArgumentParser(description="NEATify Master Node")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--auth", type=str, default="secret", help="Authentication key")
    parser.add_argument("--generations", type=int, default=20, help="Generations to train")
    
    args = parser.parse_args()
    
    # Create configuration
    evo_config = create_config()
    
    print(f"Initializing Master Node on port {args.port}...")
    
    # Initialize Distributed Population using Defaults
    population = DistributedPopulation(
        pop_size=evo_config.population_size,
        num_inputs=5, # 4 radars + speed
        num_outputs=2, # Turn L/R
        config=evo_config
    )
    
    print("Waiting for workers to connect...")
    # It seems it waits during run_generation or implicitly?
    # In step 113 it said "Waiting for 1 workers..." during init? No, it was just running?
    # Actually Step 111 status was RUNNING "Waiting for 1 workers...".
    
    try:
        for gen in range(args.generations):
            print(f"\n--- Generation {gen} ---")
            
            # Master calls run_generation. 
            # In distributed mode, it likely sends genomes to workers.
            # We don't pass an eval function here because the workers execute it?
            # Or do we pass a dummy? 
            # Standard neatify: `run_generation(fitness_fn)`.
            # If distributed, maybe fitness_fn is serialized and sent?
            # But we want workers to run their OWN code (simulation.py).
            # So maybe we pass `None` or a function that just returns?
            # Or maybe neatify expects the function to be sent.
            # If we pass a function, pickle might fail if it relies on local globals.
            # Let's assume providing a function is required, and neatify sends it.
            
            # If we need workers to run local code, usually we define a function name or similar.
            # Let's try passing the function from simulation.py, assuming pickle works.
             
            population.run_generation(None) 
            
            # Get best
            best = population.best_genome
            print(f"Gen {gen}: Best Fitness = {best.fitness:.2f}")
            
    except KeyboardInterrupt:
        print("Stopping master...")
    finally:
        # population.shutdown() # if available
        pass

if __name__ == "__main__":
    main()
