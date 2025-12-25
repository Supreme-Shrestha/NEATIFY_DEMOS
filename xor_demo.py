import neatify
import torch

# Define XOR inputs and outputs
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

def eval_genomes(genomes):
    for genome in genomes:
        # Create model from genome
        net = neatify.NeatModule(genome)

        error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            input_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            output = net(input_tensor)
            error += (output.item() - expected[0]) ** 2
        
        genome.fitness = 1.0 - error

def run():
    # Configure NEAT
    config = neatify.EvolutionConfig()
    # config.pop_size = 150 # Quick start sets it on Population init?
    # Quick start: pop = Population(pop_size=150, ...)
    config.prob_add_node = 0.1
    config.prob_add_connection = 0.2
    
    # Initialize implementation
    # Quick Start: pop = Population(pop_size=150, num_inputs=2, num_outputs=1, config=config)
    population = neatify.Population(pop_size=150, num_inputs=2, num_outputs=1, config=config)
    
    print("Starting evolution...")
    
    for generation in range(50):
        population.run_generation(eval_genomes)
        
        # Access genomes
        # Quick Start: best = max(pop.genomes, key=lambda g: g.fitness)
        # But pop.genomes might be a list or dict. 
        # Inspecting previous code I assumed dict, but Quick Start implies iterable.
        
        best_genome = max(population.genomes, key=lambda g: g.fitness if g.fitness is not None else -100)
        print(f"Gen {generation}: Best Fitness = {best_genome.fitness}")
        
        if best_genome.fitness > 0.9:
            print("Solved!")
            break

if __name__ == '__main__':
    run()
