import neatify
import torch
import numpy as np

# Define function to approximate: y = x^2
def target_function(x):
    return x ** 2

# Generate training data
x_train = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
y_train = target_function(x_train)

# Convert to tensors
inputs = torch.tensor(x_train, dtype=torch.float32)
targets = torch.tensor(y_train, dtype=torch.float32)

def eval_genomes(genomes):
    for genome in genomes:
        # Create model from genome
        net = neatify.NeatModule(genome)
        
        # evaluation
        # We process inputs item by item or we can try batching if neatify supports it
        # Based on xor_demo attempt, we used unsqueeze(0) for single item.
        # Let's try to batch process if possible, but NeatModule might be built for single instance?
        # PyTorch modules generally support batching.
        # If inputs is (N, 1), and we pass it, let's see.
        
        try:
            predictions = net(inputs)
            # Ensure predictions shape matches targets
            if predictions.shape != targets.shape:
                predictions = predictions.view_as(targets)
            
            loss = torch.mean((predictions - targets) ** 2)
            genome.fitness = 1.0 / (1.0 + loss.item()) # Maximize fitness (inverse of loss)
            
        except Exception as e:
            # Fallback if batching fails or shape mismatch
            # print(f"Error evaluating genome: {e}")
            genome.fitness = 0.0

def run():
    config = neatify.EvolutionConfig()
    config.prob_add_node = 0.1
    config.prob_add_connection = 0.2
    
    # Initialize population
    # num_inputs=1, num_outputs=1
    pop = neatify.Population(pop_size=150, num_inputs=1, num_outputs=1, config=config)
    
    print("Starting function approximation evolution...")
    
    for generation in range(50):
        pop.run_generation(eval_genomes)
        
        best_genome = max(pop.genomes, key=lambda g: g.fitness if g.fitness is not None else 0.0)
        print(f"Gen {generation}: Best Fitness = {best_genome.fitness:.4f}")
        
        if best_genome.fitness > 0.99: # Corresponds to loss < 0.01 roughly
            print("Solved (good approximation)!")
            break

if __name__ == '__main__':
    run()
