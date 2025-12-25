"""
Distributed Worker Client for Self-Driving Car Training
Connects to master and evaluates genomes using headless simulation.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless mode

import socket
import sys
import time
from simulation import SimulationManager, Car, evaluate_car_fitness
from neatify import NeatModule
from distributed_protocol import *

class DistributedWorker:
    """Worker client that evaluates genomes."""
    
    def __init__(self, master_host, master_port, worker_id=None):
        self.master_host = master_host
        self.master_port = master_port
        self.worker_id = worker_id or f"worker-{os.getpid()}"
        self.socket = None
        self.running = False
        self.sim_manager = None
        self.track_name = None
        
    def connect(self):
        """Connect to master server."""
        print(f"🔌 Connecting to master at {self.master_host}:{self.master_port}...")
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.master_host, self.master_port))
        
        # Register with master
        send_message(self.socket, Message.REGISTER, {
            'worker_id': self.worker_id
        })
        
        # Wait for acknowledgment
        msg = receive_message(self.socket)
        if msg and msg['type'] == Message.REGISTER and msg['data']['status'] == 'accepted':
            self.track_name = msg['data']['track_name']
            print(f"✅ Connected! Assigned track: {self.track_name}")
            
            # Initialize simulation
            self.sim_manager = SimulationManager()
            return True
        
        return False
    
    def run(self):
        """Main worker loop."""
        self.running = True
        print(f"🏃 Worker '{self.worker_id}' ready for tasks\n")
        
        while self.running:
            try:
                # Wait for task
                self.socket.settimeout(1.0)
                msg = receive_message(self.socket)
                
                if not msg:
                    continue
                
                if msg['type'] == Message.TASK:
                    # Evaluate genome
                    genome_id = msg['data']['genome_id']
                    genome_data = msg['data']['genome_data']
                    
                    print(f"📋 Evaluating genome {genome_id}...")
                    
                    # Deserialize genome
                    genome = deserialize_genome(genome_data)
                    
                    # Create network
                    net = NeatModule(genome)
                    
                    # Create car
                    track_surface = self.sim_manager.get_track_data(self.track_name)
                    car = Car(self.track_name, track_surface)
                    
                    # Evaluate
                    fitness = evaluate_car_fitness(net, car, max_frames=2000)
                    
                    print(f"✅ Fitness: {fitness:.1f}")
                    
                    # Send result back
                    send_message(self.socket, Message.RESULT, {
                        'genome_id': genome_id,
                        'fitness': fitness
                    })
                    
                elif msg['type'] == Message.SHUTDOWN:
                    print("🛑 Shutdown signal received")
                    self.running = False
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"❌ Error: {e}")
                self.running = False
    
    def shutdown(self):
        """Shutdown worker."""
        print("\n🛑 Shutting down worker...")
        self.running = False
        if self.socket:
            self.socket.close()
        print("✅ Worker shutdown complete")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Worker Client")
    parser.add_argument("--master", type=str, default="127.0.0.1", help="Master server IP")
    parser.add_argument("--port", type=int, default=5000, help="Master server port")
    parser.add_argument("--id", type=str, help="Worker ID (optional)")
    
    args = parser.parse_args()
    
    worker = DistributedWorker(
        master_host=args.master,
        master_port=args.port,
        worker_id=args.id
    )
    
    try:
        if worker.connect():
            worker.run()
        else:
            print("❌ Failed to connect to master")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        worker.shutdown()

if __name__ == "__main__":
    main()
