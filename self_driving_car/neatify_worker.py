import os
import argparse
import logging
import pygame
import sys
import torch
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# UI Constants
LIGHT_BLUE = (173, 216, 230)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

def monkeypatch_evaluator():
    """
    Monkeypatch neatify's GenomeEvaluator to inject the generation number 
    into the genome objects, as the library currently fails to do so.
    """
    from neatify.distributed.worker import GenomeEvaluator
    from neatify.distributed.protocol import deserialize_genome, FitnessResult
    
    original_eval = GenomeEvaluator.evaluate_batch

    def patched_evaluate_batch(self, genome_packages, fitness_function):
        results = []
        genomes = []
        for pkg in genome_packages:
            try:
                genome = deserialize_genome(pkg.serialized_genome)
                # INJECT THE MISSING GENERATION INFO
                genome.generation = getattr(pkg, 'generation', 0)
                genomes.append(genome)
            except Exception as e:
                results.append(FitnessResult(pkg.genome_id, 0.0, evaluation_successful=False, error_message=str(e)))
        
        try:
            fitness_function(genomes)
            for genome in genomes:
                results.append(FitnessResult(genome.id, getattr(genome, 'fitness', 0.0)))
        except Exception as e:
            for genome in genomes:
                results.append(FitnessResult(genome.id, 0.0, evaluation_successful=False, error_message=str(e)))
        return results

    GenomeEvaluator.evaluate_batch = patched_evaluate_batch
    print("🛠️  Applied Library Patch: Generation tracking fixed.")

def main():
    parser = argparse.ArgumentParser(description="NEATify Worker Node")
    parser.add_argument("--master", type=str, default="127.0.0.1", help="Master server IP")
    parser.add_argument("--port", type=int, default=5000, help="Master server port")
    parser.add_argument("--visualize", action="store_true", help="Show the cars running")
    parser.add_argument("--capacity", type=int, default=30, help="Max cars to simulate at once")
    
    args = parser.parse_args()
    
    if not args.visualize:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    monkeypatch_evaluator()

    from neatify.distributed import WorkerNode
    from neatify import NeatModule
    from simulation import SimulationManager, Car, evaluate_car_fitness
    from config import SCREEN_WIDTH, SCREEN_HEIGHT, TRACKS, SCRIPT_DIR

    print(f"🚀 Starting NEATify Worker Node")
    print(f"📡 Connecting to {args.master}:{args.port}")
    
    if args.visualize:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("NEATify Worker - Distributed Training")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 20, bold=True)
    else:
        screen = None
        clock = None

    sim_manager = SimulationManager()
    tracks_list = list(TRACKS.keys())

    def evaluation_function(genomes):
        """Evaluate genomes with fixed track rotation and Light Blue UI."""
        # Use the injected generation number
        gen_num = getattr(genomes[0], 'generation', 0)
        track_key = tracks_list[gen_num % len(tracks_list)]
        track_info = TRACKS[track_key]
        
        print(f"🏁 [Gen {gen_num}] Track: {track_info['name']}")
        
        for batch_idx in range(0, len(genomes), args.capacity):
            batch_chunk = genomes[batch_idx : batch_idx + args.capacity]
            
            if args.visualize:
                track_surface = sim_manager.get_track_data(track_key)
                # Load overlay
                overlay_path = os.path.join(SCRIPT_DIR, "tracks", f"{track_key}-overlay.png")
                overlay_surface = pygame.image.load(overlay_path).convert_alpha() if os.path.exists(overlay_path) else None

                cars = [Car(track_key, track_surface) for _ in batch_chunk]
                nets = [NeatModule(g) for g in batch_chunk]

                frames = 0
                while any(car.alive for car in cars) and frames < 2000:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit(); sys.exit()

                    for car, net in zip(cars, nets):
                        if car.alive:
                            input_data = torch.tensor(car.get_data(), dtype=torch.float32).unsqueeze(0)
                            try:
                                with torch.no_grad():
                                     output = net(input_data).flatten()
                                car.direction = 0
                                if output.numel() >= 1 and output[0].item() > 0.7: car.direction = 1
                                elif output.numel() >= 2 and output[1].item() > 0.7: car.direction = -1
                                car.update()
                            except: car.alive = False
                    
                    # Render Frame
                    screen.blit(track_surface, (0, 0))
                    
                    for car in cars:
                        if car.alive:
                            # Visualize Radars (Green lines)
                            for radar_angle, dist in car.radars:
                                x = int(car.rect.center[0] + math.cos(math.radians(car.angle + radar_angle)) * dist)
                                y = int(car.rect.center[1] - math.sin(math.radians(car.angle + radar_angle)) * dist)
                                pygame.draw.line(screen, GREEN, car.rect.center, (x, y), 1)
                                pygame.draw.circle(screen, RED, (x, y), 3)
                            
                            # Visualize Car
                            screen.blit(car.image, car.rect)
                    
                    if overlay_surface:
                        screen.blit(overlay_surface, (0, 0))
                    
                    # UI Overlay (Light Blue)
                    ui_lines = [
                        f"GENERATION: {gen_num}",
                        f"TRACK: {track_info['name']}",
                        f"AGENTS: {len([c for c in cars if c.alive])}/{len(batch_chunk)}",
                        f"FRAME: {frames}/2000"
                    ]
                    
                    for i, text in enumerate(ui_lines):
                        text_surf = font.render(text, True, LIGHT_BLUE)
                        screen.blit(text_surf, (15, 15 + i * 30))
                    
                    pygame.display.flip()
                    clock.tick(60)
                    frames += 1

                for genome, car in zip(batch_chunk, cars):
                    fitness = car.speed * 0.1 + car.distance * 0.01 + car.laps * 1000
                    if not car.alive and not car.dead_penalized: fitness -= 50
                    genome.fitness = fitness
            else:
                # Headless Mode
                for genome in batch_chunk:
                    net = NeatModule(genome)
                    car = Car(track_key, sim_manager.get_track_data(track_key))
                    try: 
                        genome.fitness = evaluate_car_fitness(net, car, max_frames=2000)
                    except: 
                        genome.fitness = 0.0

    try:
        worker = WorkerNode(args.master, args.port, os.getpid(), evaluation_function, args.capacity)
        print("✅ Worker connected and ready!")
        worker.start()
    except KeyboardInterrupt:
        print("\n⚠️  Worker stopped by user")
    except Exception as e:
        print(f"\n❌ Worker error: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
