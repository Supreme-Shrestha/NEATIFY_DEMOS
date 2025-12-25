"""
Neatify Worker Node for Self-Driving Car Training
Uses neatify's built-in WorkerNode for distributed evaluation.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless mode

import argparse
from neatify.distributed import WorkerNode
from neatify import NeatModule
from simulation import SimulationManager, Car, evaluate_car_fitness

# Initialize simulation manager globally
sim_manager = SimulationManager()
current_track = "track1"

def evaluation_function(genome):
    """Evaluate a single genome."""
    # Create network
    net = NeatModule(genome)
    
    # Create car
    track_surface = sim_manager.get_track_data(current_track)
    car = Car(current_track, track_surface)
    
    # Evaluate and return fitness
    fitness = evaluate_car_fitness(net, car, max_frames=2000)
    return fitness

def main():
    global current_track
    
    parser = argparse.ArgumentParser(description="NEATify Worker Node")
    parser.add_argument("--master", type=str, default="127.0.0.1", help="Master server IP")
    parser.add_argument("--port", type=int, default=5000, help="Master server port")
    parser.add_argument("--track", type=str, default="track1", help="Track name (will be set by master)")
    
    args = parser.parse_args()
    current_track = args.track
    
    print(f"🚀 Starting NEATify Worker Node")
    print(f"📡 Connecting to {args.master}:{args.port}")
    print(f"📍 Track: {current_track}\n")
    
    try:
        # Create and start worker
        # Signature: (master_host, master_port, worker_id, fitness_function, capacity)
        worker = WorkerNode(
            master_host=args.master,
            master_port=args.port,
            worker_id=os.getpid(),  # Use process ID as worker ID
            fitness_function=evaluation_function,
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
