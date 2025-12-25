# Neatify Distributed Training Bug Report

## Issue
`DistributedPopulation.run_generation()` does not distribute genome evaluation to connected workers. All fitness scores remain 0.0.

## Root Cause
In `neatify/distributed/master.py`, the `DistributedPopulation.run_generation()` method is implemented as:

```python
def run_generation(self, fitness_function):
    super().run_generation(lambda g: None)
    # But Population.run_generation expects actual fitness evaluation
```

**Problem**: The method calls the parent `Population.run_generation()` with a dummy lambda function `(lambda g: None)` instead of using the distributed fitness evaluation mechanism.

## Expected Behavior
The `run_generation` method should:
1. Distribute genomes to connected workers
2. Workers evaluate genomes using their fitness functions
3. Collect results from workers
4. Assign fitness scores to genomes
5. Proceed with selection/crossover/mutation

## Actual Behavior
- Master calls `super().run_generation(lambda g: None)`
- Dummy function returns None for all genomes
- All fitness scores remain 0.0
- Workers connect but never receive evaluation tasks
- Workers receive shutdown signal immediately

## Reproduction Steps

### Master (master.py):
```python
from neatify import DistributedPopulation, EvolutionConfig
from neatify.distributed.config import DistributedConfig

config = EvolutionConfig()
dist_config = DistributedConfig(host='0.0.0.0', port=5000)

population = DistributedPopulation(
    pop_size=30,
    num_inputs=5,
    num_outputs=2,
    config=config,
    distributed_config=dist_config
)

for gen in range(10):
    population.run_generation(lambda genomes: None)
    best = max(population.genomes, key=lambda g: g.fitness if g.fitness else 0)
    print(f"Gen {gen}: Best Fitness = {best.fitness}")  # Always 0.0
```

### Worker (worker.py):
```python
from neatify.distributed import WorkerNode

def fitness_function(genome):
    # Some evaluation logic
    return 100.0  # This never gets called

worker = WorkerNode(
    master_host='127.0.0.1',
    master_port=5000,
    worker_id=1,
    fitness_function=fitness_function,
    capacity=50
)

worker.start()  # Connects but never evaluates
```

## Suggested Fix

The `run_generation` method should call the distributed fitness evaluation instead of the dummy function:

```python
def run_generation(self, fitness_function):
    # Distribute genomes to workers and collect results
    self._distributed_fitness_evaluation(self.genomes)
    
    # Then proceed with evolution (selection, crossover, mutation)
    super().run_generation(lambda g: None)  # Skip fitness eval, already done
```

Or alternatively, override the fitness evaluation in the parent class:

```python
def run_generation(self, fitness_function):
    # Use distributed evaluation instead of local
    def distributed_eval(genomes):
        self._send_to_workers(genomes)
        self._collect_results(genomes)
    
    super().run_generation(distributed_eval)
```

## Environment
- neatify-ai: 0.1.3
- Python: 3.13.5
- OS: Windows 11

## Additional Notes
The `_distributed_fitness` method exists in the class but is never called by `run_generation`. The coordinator starts a server and workers can connect, but the task distribution mechanism is not integrated into the generation loop.

## Impact
Distributed training is completely non-functional. Users cannot leverage multiple machines for NEAT evolution, defeating the purpose of the `DistributedPopulation` class.
