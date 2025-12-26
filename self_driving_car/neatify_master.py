"""
Neatify Master Node for Self-Driving Car Training
Uses neatify's built-in DistributedPopulation for coordinating workers.
"""
import argparse
import random
import logging

# Configure logging to see distributed communication
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from neatify import DistributedPopulation, EvolutionConfig
from config import create_config, TRACKS

def main():
    parser = argparse.ArgumentParser(description="NEATify Distributed Master")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--track", type=str, default="track1", choices=list(TRACKS.keys()), help="Track to train on")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--workers", type=int, default=1, help="Minimum workers to wait for")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"NEATIFY DISTRIBUTED TRAINING - SELF-DRIVING CAR")
    print(f"{'='*60}")
    print(f"📍 Track: {TRACKS[args.track]['name']}")
    print(f"🔢 Generations: {args.generations}")
    print(f"👥 Waiting for {args.workers} worker(s)")
    print(f"🌐 Port: {args.port}")
    print(f"{'='*60}\n")
    
    # Create evolution config
    config = create_config()
    
    # Create distributed configuration
    from neatify.distributed.config import DistributedConfig
    dist_config = DistributedConfig(
        host='0.0.0.0',  # Listen on all interfaces
        port=args.port,
        min_workers=args.workers
    )
    
    # Create distributed population
    print("🚀 Initializing distributed population...")
    population = DistributedPopulation(
        pop_size=config.population_size,
        num_inputs=5,  # 4 radars + speed
        num_outputs=2,  # Turn left/right
        config=config,
        distributed_config=dist_config
    )
    
    print(f"✅ Population initialized ({config.population_size} genomes)")
    print(f"⏳ Waiting for workers to connect...\n")
    
    try:
        for gen in range(args.generations):
            print(f"🏁 Generation {gen + 1}/{args.generations} | Track: {TRACKS[args.track]['name']}")
            
            # Run generation - DistributedPopulation will send genomes to workers
            population.run_generation(lambda genomes: None)
            
            # Get statistics
            best = max(population.genomes, key=lambda g: g.fitness if g.fitness else 0)
            avg = sum(g.fitness for g in population.genomes if g.fitness) / len(population.genomes)
            
            print(f"📊 Best: {best.fitness:.1f} | Avg: {avg:.1f}\n")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE!")
        print(f"Final Best Fitness: {best.fitness:.1f}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    finally:
        if hasattr(population, 'shutdown'):
            population.shutdown()

if __name__ == "__main__":
    main()
