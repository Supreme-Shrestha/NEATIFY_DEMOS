"""
Neatify Worker Node for Self-Driving Car Training
Uses neatify's built-in WorkerNode for distributed evaluation.
Supports dynamic track switching.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless mode

import argparse
import logging
from neatify.distributed import WorkerNode
from neatify import NeatModule
from simulation import SimulationManager, Car, evaluate_car_fitness

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize simulation manager globally
sim_manager = SimulationManager()

def create_evaluation_function(track_name):
    """Create an evaluation function for a specific track."""
    def evaluation_function(genomes):
        """Evaluate a batch of genomes as required by neatify-ai 0.1.4."""
        print(f"📋 Evaluating batch of {len(genomes)} genomes on {track_name}...")
        for i, genome in enumerate(genomes):
            if i % 10 == 0 and i > 0:
                print(f"  ...processed {i}/{len(genomes)} genomes")
            # Create network
            net = NeatModule(genome)
            
            # Create car
            track_surface = sim_manager.get_track_data(track_name)
            car = Car(track_name, track_surface)
            
            # Evaluate and set fitness
            genome.fitness = evaluate_car_fitness(net, car, max_frames=2000)
    
    return evaluation_function

def main():
    parser = argparse.ArgumentParser(description="NEATify Worker Node")
    parser.add_argument("--master", type=str, default="127.0.0.1", help="Master server IP")
    parser.add_argument("--port", type=int, default=5000, help="Master server port")
    parser.add_argument("--track", type=str, default="track1", help="Default track (can be overridden by master)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting NEATify Worker Node")
    print(f"📡 Connecting to {args.master}:{args.port}")
    print(f"📍 Default Track: {args.track}\n")
    
    # Create evaluation function with default track
    eval_func = create_evaluation_function(args.track)
    
    try:
        # Create and start worker
        # Signature: (master_host, master_port, worker_id, fitness_function, capacity)
        worker = WorkerNode(
            master_host=args.master,
            master_port=args.port,
            worker_id=os.getpid(),  # Use process ID as worker ID
            fitness_function=eval_func,
            capacity=50
        )
        
        print("✅ Worker connected and ready!")
        print("⏳ Waiting for tasks from master...\n")
        
        # Worker runs indefinitely until interrupted
        worker.start()
        
    except KeyboardInterrupt:
        print("\n⚠️  Worker stopped by user")
    except Exception as e:
        print(f"\n❌ Worker error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
