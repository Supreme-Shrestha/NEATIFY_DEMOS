"""
Distributed Master Server for Self-Driving Car Training
Coordinates workers and manages the evolution process.
"""
import socket
import threading
import time
from neatify import Population, EvolutionConfig
from config import create_config, TRACKS
from distributed_protocol import *

class DistributedMaster:
    """Master server that coordinates distributed training."""
    
    def __init__(self, port=5000, track_name="track1"):
        self.port = port
        self.track_name = track_name
        self.workers = {}  # worker_id -> socket
        self.worker_lock = threading.Lock()
        self.results = {}  # genome_id -> fitness
        self.results_lock = threading.Lock()
        self.running = False
        
        # Create evolution config
        self.config = create_config()
        self.population = None
        self.generation = 0
        
    def start_server(self):
        """Start the master server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(5)
        self.running = True
        
        print(f"🚀 Master server started on port {self.port}")
        print(f"📍 Track: {TRACKS[self.track_name]['name']}")
        print(f"⏳ Waiting for workers to connect...\n")
        
        # Start accept thread
        accept_thread = threading.Thread(target=self._accept_workers, daemon=True)
        accept_thread.start()
        
    def _accept_workers(self):
        """Accept incoming worker connections."""
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_socket, address = self.server_socket.accept()
                
                # Receive registration
                msg = receive_message(client_socket)
                if msg and msg['type'] == Message.REGISTER:
                    worker_id = msg['data']['worker_id']
                    
                    with self.worker_lock:
                        self.workers[worker_id] = client_socket
                    
                    print(f"✅ Worker '{worker_id}' connected from {address[0]}")
                    
                    # Send acknowledgment with track info
                    send_message(client_socket, Message.REGISTER, {
                        'status': 'accepted',
                        'track_name': self.track_name
                    })
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ Error accepting worker: {e}")
    
    def distribute_genomes(self, genomes):
        """Distribute genomes to workers for evaluation."""
        self.results.clear()
        
        with self.worker_lock:
            if not self.workers:
                print("⚠️  No workers connected! Waiting...")
                return False
            
            worker_list = list(self.workers.items())
            num_workers = len(worker_list)
            
            print(f"📤 Distributing {len(genomes)} genomes to {num_workers} workers...")
            
            # Distribute genomes evenly
            for i, genome in enumerate(genomes):
                worker_id, worker_socket = worker_list[i % num_workers]
                
                try:
                    # Send genome to worker
                    send_message(worker_socket, Message.TASK, {
                        'genome_id': id(genome),
                        'genome_data': serialize_genome(genome)
                    })
                except Exception as e:
                    print(f"❌ Error sending to worker '{worker_id}': {e}")
                    # Remove failed worker
                    del self.workers[worker_id]
            
            return True
    
    def collect_results(self, genomes, timeout=60):
        """Collect fitness results from workers."""
        start_time = time.time()
        expected_results = len(genomes)
        
        print(f"📥 Collecting results from workers...")
        
        while len(self.results) < expected_results:
            if time.time() - start_time > timeout:
                print(f"⏱️  Timeout! Got {len(self.results)}/{expected_results} results")
                break
            
            with self.worker_lock:
                for worker_id, worker_socket in list(self.workers.items()):
                    try:
                        worker_socket.settimeout(0.1)
                        msg = receive_message(worker_socket)
                        
                        if msg and msg['type'] == Message.RESULT:
                            genome_id = msg['data']['genome_id']
                            fitness = msg['data']['fitness']
                            
                            with self.results_lock:
                                self.results[genome_id] = fitness
                            
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"❌ Worker '{worker_id}' error: {e}")
                        del self.workers[worker_id]
        
        # Assign fitness to genomes
        for genome in genomes:
            genome_id = id(genome)
            genome.fitness = self.results.get(genome_id, 0.0)
        
        print(f"✅ Collected {len(self.results)} results\n")
        return True
    
    def train(self, generations=10):
        """Run distributed training."""
        # Initialize population
        self.population = Population(
            pop_size=self.config.population_size,
            num_inputs=5,
            num_outputs=2,
            config=self.config
        )
        
        print(f"\n{'='*60}")
        print(f"STARTING DISTRIBUTED TRAINING")
        print(f"{'='*60}\n")
        
        for gen in range(generations):
            self.generation = gen
            print(f"🏁 Generation {gen + 1}/{generations}")
            
            # Distribute genomes
            if not self.distribute_genomes(self.population.genomes):
                time.sleep(2)  # Wait for workers
                continue
            
            # Collect results
            self.collect_results(self.population.genomes)
            
            # Evolution step (selection, crossover, mutation)
            if gen < generations - 1:
                self.population.run_generation(lambda genomes: None)  # Skip eval, already done
            
            # Stats
            best = max(self.population.genomes, key=lambda g: g.fitness if g.fitness else 0)
            avg = sum(g.fitness for g in self.population.genomes if g.fitness) / len(self.population.genomes)
            
            print(f"📊 Best Fitness: {best.fitness:.1f} | Avg: {avg:.1f}")
            print()
        
        print(f"{'='*60}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*60}\n")
    
    def shutdown(self):
        """Shutdown the server."""
        print("\n🛑 Shutting down master server...")
        self.running = False
        
        with self.worker_lock:
            for worker_id, worker_socket in self.workers.items():
                try:
                    send_message(worker_socket, Message.SHUTDOWN)
                    worker_socket.close()
                except:
                    pass
        
        self.server_socket.close()
        print("✅ Server shutdown complete")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Master Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--track", type=str, default="track1", choices=list(TRACKS.keys()), help="Track to train on")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    
    args = parser.parse_args()
    
    master = DistributedMaster(port=args.port, track_name=args.track)
    
    try:
        master.start_server()
        
        # Wait for at least one worker
        while not master.workers:
            time.sleep(1)
        
        # Start training
        master.train(generations=args.generations)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        master.shutdown()

if __name__ == "__main__":
    main()
