import neatify.distributed.master as m
import inspect

def print_source(obj):
    try:
        source = inspect.getsource(obj)
        lines = source.splitlines()
        for i, line in enumerate(lines):
            print(f"{i+1:3d}: {line}")
    except Exception as e:
        print(f"Error: {e}")

print("--- DistributedPopulation.run_generation ---")
print_source(m.DistributedPopulation.run_generation)
print("\n--- DistributedPopulation._distributed_fitness_evaluation ---")
print_source(m.DistributedPopulation._distributed_fitness_evaluation)
