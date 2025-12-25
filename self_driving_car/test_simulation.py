"""
Test the refactored simulation module with a simple genome.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless mode

from simulation import SimulationManager, Car, evaluate_car_fitness
from neatify import NeatModule, EvolutionConfig
from config import create_config

def test_simulation():
    print("Initializing simulation manager...")
    sim_manager = SimulationManager()
    
    print("Creating test configuration...")
    config = create_config()
    
    # Create a simple genome for testing
    from neatify.population import Genome
    genome = Genome(num_inputs=5, num_outputs=2, config=config)
    
    print("Creating neural network from genome...")
    net = NeatModule(genome)
    
    print("Creating car...")
    track_surface = sim_manager.get_track_data("track1")
    car = Car("track1", track_surface)
    
    print("Running simulation...")
    fitness = evaluate_car_fitness(net, car, max_frames=100)
    
    print(f"\n✅ Simulation completed!")
    print(f"Fitness: {fitness:.2f}")
    print(f"Distance traveled: {car.distance:.2f}")
    print(f"Laps completed: {car.laps}")
    print(f"Final status: {'Alive' if car.alive else 'Crashed'}")

if __name__ == "__main__":
    test_simulation()
