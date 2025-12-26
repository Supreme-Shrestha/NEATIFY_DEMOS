import os
import argparse
import logging
import pygame
import sys
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="NEATify Worker Node")
    parser.add_argument("--master", type=str, default="127.0.0.1", help="Master server IP")
    parser.add_argument("--port", type=int, default=5000, help="Master server port")
    parser.add_argument("--track", type=str, default="track1", help="Default track")
    parser.add_argument("--visualize", action="store_true", help="Show the cars running in a window")
    
    args = parser.parse_args()
    
    if not args.visualize:
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless mode
    
    # Imports inside main to avoid early pygame init
    from neatify.distributed import WorkerNode
    from neatify import NeatModule
    from simulation import SimulationManager, Car, evaluate_car_fitness
    from config import SCREEN_WIDTH, SCREEN_HEIGHT, TRACKS

    print(f"🚀 Starting NEATify Worker Node")
    print(f"📡 Connecting to {args.master}:{args.port}")
    if args.visualize:
        print("📺 Visualization ENABLED")
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("NEATify Worker - Evaluating Genomes")
        clock = pygame.time.Clock()
    else:
        screen = None
        clock = None

    sim_manager = SimulationManager()

    def evaluation_function(genomes):
        """Evaluate a batch of genomes, optionally with visualization."""
        track_name = args.track # In real scenario, master might send track, but let's use arg for now
        print(f"📋 Evaluating batch of {len(genomes)} genomes on {track_name}...")
        
        if args.visualize:
            # Parallel evaluation with visualization (like car_evolution.py)
            track_surface = sim_manager.get_track_data(track_name)
            cars = []
            nets = []
            
            for genome in genomes:
                net = NeatModule(genome)
                car = Car(track_name, track_surface)
                cars.append(car)
                nets.append(net)
                genome.fitness = 0.0

            frames = 0
            running = True
            while running and frames < 2000:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # Simulation step
                alive_count = 0
                for i, (car, net) in enumerate(zip(cars, nets)):
                    if car.alive:
                        alive_count += 1
                        # Get input
                        input_data = torch.tensor(car.get_data(), dtype=torch.float32).unsqueeze(0)
                        
                        # Inference
                        try:
                            with torch.no_grad():
                                 raw_output = net(input_data)
                                 output = raw_output.flatten()
                            
                            # Control
                            car.direction = 0
                            if output.numel() >= 1:
                                if output[0].item() > 0.7: car.direction = 1
                                elif output.numel() >= 2 and output[1].item() > 0.7: car.direction = -1
                        except:
                            car.alive = False
                        
                        car.update()
                        
                        # Live fitness update (optional, but good for tracking)
                        # We'll calculate final fitness at the end to be consistent
                
                if alive_count == 0:
                    break

                # Render
                screen.blit(track_surface, (0, 0))
                for car in cars:
                    if car.alive:
                        screen.blit(car.image, car.rect)
                
                # Overlay text
                font = pygame.font.SysFont("Arial", 20)
                text = font.render(f"Evaluating: {alive_count}/{len(genomes)} alive | Frame: {frames}", True, (255, 255, 0))
                screen.blit(text, (10, 10))
                
                pygame.display.flip()
                clock.tick(60) # Cap at 60 FPS for visibility
                frames += 1

            # Assign final fitness after simulation ends
            for i, (genome, car) in enumerate(zip(genomes, cars)):
                fitness = car.speed * 0.1 + car.distance * 0.01 + car.laps * 1000
                if not car.alive and not car.dead_penalized:
                    fitness -= 50
                genome.fitness = fitness
                print(f"  ✅ Genome {genome.id} fitness: {fitness:.2f}")

        else:
            # Sequential headless evaluation
            for i, genome in enumerate(genomes):
                if i % 10 == 0 and i > 0:
                    print(f"  ...processed {i}/{len(genomes)} genomes")
                
                net = NeatModule(genome)
                track_surface = sim_manager.get_track_data(track_name)
                car = Car(track_name, track_surface)
                
                try:
                    fit = evaluate_car_fitness(net, car, max_frames=2000)
                    genome.fitness = fit
                    print(f"  ✅ Genome {genome.id} fitness: {fit:.2f}")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    genome.fitness = 0.0

    try:
        worker = WorkerNode(
            master_host=args.master,
            master_port=args.port,
            worker_id=os.getpid(),
            fitness_function=evaluation_function,
            capacity=50
        )
        
        print("✅ Worker connected and ready!")
        worker.start()
        
    except KeyboardInterrupt:
        print("\n⚠️  Worker stopped by user")
    except Exception as e:
        print(f"\n❌ Worker error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
