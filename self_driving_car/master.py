
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
    
    # Create distributed configuration
    dist_config = DistributedConfig(
        auth_key=args.auth.encode('utf-8'),
        enable_fault_tolerance=True
    )
    
    print(f"Initializing Master Node on port {args.port}...")
    
    # Initialize Distributed Population
    # pop_size must be divisible by workers ideally, or neatify handles it.
    population = DistributedPopulation(
        max_workers=4, # Limit concurrent if needed, or dynamic
        # neatify DistributedPopulation might take config as arg?
        # Based on earlier inspection: DistributedPopulation(config, num_inputs, num_outputs) 
        # But for distributed specific, maybe it uses DistributedConfig?
        # Let's check init signature from history. 
        # (self, pop_size: int, num_inputs: int, num_outputs: int, config: EvolutionConfig = None, distributed_config: DistributedConfig = None) (Inferred)
        # Actually Step 83 showed: (self, pop_size: int, num_inputs: int, num_outputs: int, config: EvolutionConfig = None)
        # It didn't start with distributed config.
        # But neatify.distributed.config exists.
        # Maybe config has distributed fields?
        # Let's assume standard pop init for now but we need to bind to address.
        # DistributedPopulation likely inherits from Population but overrides methods.
        # Wait, if DistributedPopulation signature is (pop_size, inputs, outputs, config), 
        # how do we set port/auth?
        # Maybe via the `config` object if it has distributed fields?
        # OR maybe there's a `.start_server()` method?
        # Let's assume we pass distributed_config if the library supports it, or check simple usage.
        # History Step 52: dir(DistributedPopulation) -> ['shutdown', 'speciate', 'run_generation'...]
        # It didn't show start_server explicitly.
        # Let's look at `inspect_pop.py` output again... it timed out waiting for workers.
        # It was initialized as `DistributedPopulation(150, 2, 1)`.
        # I suspect DistributedPopulation constructs the server in `__init__`.
        # We might need to look closer or just try to pass kwargs if it accepts `**kwargs`.
        
        pop_size=evo_config.population_size,
        num_inputs=5, # 4 radars + speed
        num_outputs=2, # Turn L/R
        config=evo_config
    )
    
    # If DistributedConfig is not passed in init, maybe we attach it?
    # Or maybe we need to rely on defaults and hope 5000 is open.
    # The `worker.py` uses `Worker` class.
    
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
