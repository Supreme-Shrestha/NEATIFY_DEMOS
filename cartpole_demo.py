"""
CartPole Balancing Demo using NEATify
Evolves a neural network to balance a pole on a cart.
"""
import numpy as np
from neatify import Population, NeatModule, EvolutionConfig
import torch

class CartPole:
    """Simple CartPole physics simulation."""
    
    def __init__(self):
        self.reset()
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        
    def reset(self):
        """Reset to initial state."""
        self.x = 0.0  # cart position
        self.x_dot = 0.0  # cart velocity
        self.theta = np.random.uniform(-0.05, 0.05)  # pole angle
        self.theta_dot = 0.0  # pole angular velocity
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        """Return current state."""
        return [self.x, self.x_dot, self.theta, self.theta_dot]
    
    def step(self, action):
        """Apply action and update physics."""
        force = self.force_mag if action > 0.5 else -self.force_mag
        
        costheta = np.cos(self.theta)
        sintheta = np.sin(self.theta)
        
        temp = (force + self.masspole * self.length * self.theta_dot**2 * sintheta) / (self.masscart + self.masspole)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / (self.masscart + self.masspole))
        )
        xacc = temp - self.masspole * self.length * thetaacc * costheta / (self.masscart + self.masspole)
        
        self.x += self.tau * self.x_dot
        self.x_dot += self.tau * xacc
        self.theta += self.tau * self.theta_dot
        self.theta_dot += self.tau * thetaacc
        
        self.steps += 1
        
        # Check if failed
        done = (
            self.x < -2.4 or self.x > 2.4 or
            self.theta < -0.209 or self.theta > 0.209 or
            self.steps >= 500
        )
        
        return self.get_state(), done

def eval_genomes(genomes):
    """Evaluate all genomes."""
    for genome in genomes:
        net = NeatModule(genome)
        env = CartPole()
        
        state = env.reset()
        total_reward = 0
        
        while True:
            # Get action from network
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = net(state_tensor).squeeze()
            
            # Step environment
            state, done = env.step(output.item())
            total_reward += 1
            
            if done:
                break
        
        genome.fitness = total_reward

def main():
    print("=" * 60)
    print("NEATIFY CARTPOLE DEMO")
    print("=" * 60)
    
    # Configure evolution
    config = EvolutionConfig()
    config.population_size = 150
    config.prob_add_node = 0.1
    config.prob_add_connection = 0.3
    
    # Create population
    pop = Population(pop_size=150, num_inputs=4, num_outputs=1, config=config)
    
    print("\nStarting evolution...")
    print("Goal: Balance pole for 500 steps\n")
    
    for generation in range(100):
        pop.run_generation(eval_genomes)
        
        best = max(pop.genomes, key=lambda g: g.fitness if g.fitness else 0)
        avg_fitness = sum(g.fitness for g in pop.genomes if g.fitness) / len(pop.genomes)
        
        print(f"Gen {generation:3d} | Best: {best.fitness:6.1f} | Avg: {avg_fitness:6.1f}")
        
        if best.fitness >= 500:
            print(f"\n✅ Solved in {generation} generations!")
            break
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
