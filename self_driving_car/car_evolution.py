"""
NEATify Self-Driving Car Simulation - Multi-Generation Training with Radar Visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import math
import torch
from datetime import datetime, timedelta
from neatify import Population, NeatModule, EvolutionConfig
from neatify.checkpoint import Checkpoint

# Hide PyGame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Constants
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
UI_FONT = "Arial"
UI_FONT_SIZE = 24
UI_COLOR = (220, 220, 220)
UI_HIGHLIGHT = (255, 215, 0)
SAVE_INTERVAL_MINUTES = 5
TRAINING_GENERATIONS = 10
LAPS_TO_COMPLETE = 5

# Get absolute path to current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize pygame
print("=== Initializing PyGame ===")
try:
    pygame.init()
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("NEATify Self-Driving Car - Multi-Generation Training")
    
    os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"
    
    print(f"✓ PyGame initialized successfully")
except Exception as e:
    print(f"❌ PyGame initialization failed: {e}")
    sys.exit(1)

clock = pygame.time.Clock()

# Create backups directory
backup_dir = os.path.join(SCRIPT_DIR, 'backups')
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

# Track definitions with correct starting positions
TRACKS = {
    "track1": {
        "name": "Track 1",
        "start_pos": (640, 471),
        "model_prefix": "track1",
        "border_color": (255, 255, 255)
    },
    "track2": {
        "name": "Track 2", 
        "start_pos": (412, 295),
        "model_prefix": "track2",
        "border_color": (255, 255, 255)
    },
    "track3": {
        "name": "Track 3",
        "start_pos": (408, 483),
        "model_prefix": "track3",
        "border_color": (255, 255, 255)
    },
    "track4": {
        "name": "Track 4",
        "start_pos": (496, 387),
        "model_prefix": "track4",
        "border_color": (255, 255, 255)
    }
}

def load_track(name):
    """Load track images."""
    tracks_folder = os.path.join(SCRIPT_DIR, "tracks")
    track_path = os.path.join(tracks_folder, f"{name}.png")
    overlay_path = os.path.join(tracks_folder, f"{name}-overlay.png")
    
    print(f"\nLoading Track: {name}")
    print(f"Starting position: {TRACKS[name]['start_pos']}")
    
    if not os.path.exists(track_path):
        print(f"❌ ERROR: Track file not found!")
        pygame.quit()
        sys.exit(1)
    
    try:
        track_surface = pygame.image.load(track_path)
        track_surface = track_surface.convert()
        print(f"✓ Loaded track: {os.path.basename(track_path)}")
        
        if os.path.exists(overlay_path):
            overlay_surface = pygame.image.load(overlay_path).convert_alpha()
            print(f"✓ Loaded overlay: {os.path.basename(overlay_path)}")
        else:
            overlay_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay_surface.fill((0, 0, 0, 0))
            
        return track_surface, overlay_surface
        
    except Exception as e:
        print(f"❌ ERROR loading image: {e}")
        pygame.quit()
        sys.exit(1)

# Preload all tracks
print("\n" + "="*60)
print("LOADING TRACKS")
print("="*60)
TRACK_IMAGES = {}
for track_id in TRACKS:
    TRACK_IMAGES[track_id] = load_track(track_id)
print("="*60)
print("✓ ALL TRACKS LOADED")
print("="*60)

class Car(pygame.sprite.Sprite):
    """Self-driving car with radar sensors and lap counting."""
    
    def __init__(self, track_id, pos=None):
        super().__init__()
        track_info = TRACKS[track_id]
        start_pos = pos if pos else track_info["start_pos"]
        
        # Load car sprite
        car_path = os.path.join(SCRIPT_DIR, "tracks", "car4.png")
        
        if os.path.exists(car_path):
            try:
                self.original_image = pygame.image.load(car_path).convert_alpha()
            except Exception as e:
                print(f"❌ Error loading car sprite: {e}")
                self._create_default_car()
        else:
            self._create_default_car()
        
        self.image = self.original_image
        self.rect = self.image.get_rect(center=start_pos)
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []
        self.distance = 0
        self.speed = 0
        self.laps = 0
        self.dead_penalized = False
        self.track_id = track_id
        self.track_surface, self.track_overlay = TRACK_IMAGES[track_id]
        self.border_color = track_info["border_color"]
        
        # Lap tracking
        self.last_checkpoint_time = 0
        self.checkpoint_cooldown = 60  # frames
        self.start_line_pos = start_pos
        self.near_start_line = False
        self.passed_start_line = False
        
        # Radar visualization settings
        self.show_radar = True
        self.radar_colors = [
            (255, 100, 100),  # -60 degrees - Red
            (255, 150, 50),   # -30 degrees - Orange
            (255, 255, 50),   # 0 degrees - Yellow
            (50, 255, 100)    # 30 degrees - Green
        ]
    
    def _create_default_car(self):
        """Create a simple car graphic."""
        self.original_image = pygame.Surface((30, 15), pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, (0, 150, 255), (0, 0, 30, 15))
        pygame.draw.polygon(self.original_image, (200, 60, 60), 
                           [(30, 0), (40, 7), (30, 15)])

    def update(self):
        """Update car state and sensors."""
        if not self.alive:
            return
            
        self.radars.clear()
        self.drive()
        self.rotate()
        
        # Radar sensors with visualization
        radar_angles = (-60, -30, 0, 30)
        for i, radar_angle in enumerate(radar_angles):
            self.radar(radar_angle, i)
            
        self.collision()
        self.check_lap()
        self.distance += abs(self.speed)
        self.speed = math.sqrt(self.vel_vector.x**2 + self.vel_vector.y**2) * 2.5

    def drive(self):
        """Move car forward."""
        self.rect.center += self.vel_vector * 2.5
    
    def collision(self):
        """Check collision with track borders."""
        length = 25
        collision_point_right = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)
        ]
        collision_point_left = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)
        ]

        try:
            if (0 <= collision_point_right[0] < SCREEN_WIDTH and 
                0 <= collision_point_right[1] < SCREEN_HEIGHT):
                right_color = self.track_surface.get_at(collision_point_right)
                if right_color[:3] == self.border_color:
                    self.alive = False
            
            if self.alive and (0 <= collision_point_left[0] < SCREEN_WIDTH and 
                0 <= collision_point_left[1] < SCREEN_HEIGHT):
                left_color = self.track_surface.get_at(collision_point_left)
                if left_color[:3] == self.border_color:
                    self.alive = False
                    
        except IndexError:
            self.alive = False
        
        # Draw collision points
        if self.alive:
            pygame.draw.circle(SCREEN, (0, 255, 255), collision_point_right, 3)
            pygame.draw.circle(SCREEN, (0, 255, 255), collision_point_left, 3)

    def check_lap(self):
        """Check if car has completed a lap."""
        # Calculate distance to start line
        dx = self.rect.centerx - self.start_line_pos[0]
        dy = self.rect.centery - self.start_line_pos[1]
        distance_to_start = math.sqrt(dx*dx + dy*dy)
        
        # If close to start line
        if distance_to_start < 50:
            if not self.near_start_line:
                self.near_start_line = True
        else:
            if self.near_start_line and not self.passed_start_line:
                # Car has moved through start line
                self.passed_start_line = True
                
        # If car has moved away from start line after passing it
        if self.passed_start_line and distance_to_start > 100:
            self.laps += 1
            self.passed_start_line = False
            self.near_start_line = False

    def rotate(self):
        """Rotate car based on direction."""
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle, color_index):
        """Cast a radar beam to detect obstacles with visualization."""
        length = 0
        x, y = self.rect.center

        while length < 200:
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

            if not (0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT):
                break

            try:
                pixel_color = self.track_surface.get_at((x, y))
                if pixel_color[:3] == self.border_color:
                    break
            except IndexError:
                break

            length += 1

        dist = int(math.sqrt((self.rect.center[0] - x) ** 2 + (self.rect.center[1] - y) ** 2))
        self.radars.append([radar_angle, dist])
        
        # Draw radar visualization
        if self.show_radar and self.alive:
            # Radar beam line
            pygame.draw.line(SCREEN, self.radar_colors[color_index], 
                           self.rect.center, (x, y), 2)
            
            # Radar endpoint
            pygame.draw.circle(SCREEN, (0, 200, 0), (x, y), 4)
            
            # Radar distance text
            font = pygame.font.SysFont(UI_FONT, 14)
            dist_text = font.render(f"{dist}", True, self.radar_colors[color_index])
            text_pos = (x + 5, y - 10)
            SCREEN.blit(dist_text, text_pos)

    def get_data(self):
        """Return sensor data for neural network."""
        return [radar[1] for radar in self.radars] + [self.speed]

class Button:
    """Simple UI button."""
    
    def __init__(self, x, y, width, height, text, color=(70, 70, 100), hover_color=(100, 100, 150)):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.is_hovered = False
        self.font = pygame.font.SysFont(UI_FONT, UI_FONT_SIZE)

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, UI_COLOR, self.rect, 2, border_radius=5)
        
        text_surf = self.font.render(self.text, True, UI_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered:
                return True
        return False

class NEATifyManager:
    """Manager for NEATify evolution and training."""
    
    def __init__(self):
        self.config = EvolutionConfig()
        self.config.population_size = 30  # Good balance
        self.config.prob_mutate_weight = 0.8
        self.config.prob_add_connection = 0.3
        self.config.prob_add_node = 0.1
        self.config.elitism_count = 5
        self.config.weight_mutation_power = 0.5
        
        self.population = None
        self.best_genome = None
        self.generation = 0
        self.max_fitness = 0
        self.fitness_history = []
        self.start_time = datetime.now()
        self.last_save_time = datetime.now()
        self.save_interval = timedelta(minutes=SAVE_INTERVAL_MINUTES)
        self.current_track = "track1"
        self.training_active = False
        
    def initialize_population(self):
        """Create initial population."""
        self.population = Population(pop_size=self.config.population_size, 
                                    num_inputs=5, num_outputs=2, config=self.config)
        print(f"\n=== Starting training on {TRACKS[self.current_track]['name']} ===")
        print(f"Population: {self.config.population_size} cars")
        print(f"Target: {LAPS_TO_COMPLETE} laps or all cars crash")
        
    def load_best_model(self):
        """Load best model for current track."""
        filename = os.path.join(SCRIPT_DIR, f'best_genome_{self.current_track}.pkl')
        try:
            self.best_genome = Checkpoint.load_best(filename)
            print(f"Loaded best model with fitness: {self.best_genome.fitness:.2f}")
            return True
        except:
            print(f"No existing best model found")
            return False
            
    def save_best_model(self, genome):
        """Save best model for current track."""
        filename = os.path.join(SCRIPT_DIR, f'best_genome_{self.current_track}.pkl')
        Checkpoint.save_best(genome, filename, {
            "fitness": genome.fitness, 
            "generation": self.generation,
            "track": self.current_track
        })
        print(f"💾 Saved best model with fitness: {genome.fitness:.2f}")
            
    def train_multiple_generations(self, num_generations=TRAINING_GENERATIONS):
        """Train for multiple generations."""
        if not self.population:
            self.initialize_population()
        
        print(f"\n{'='*60}")
        print(f"STARTING {num_generations} GENERATIONS OF TRAINING")
        print(f"Track: {TRACKS[self.current_track]['name']}")
        print(f"Starting position: {TRACKS[self.current_track]['start_pos']}")
        print(f"{'='*60}")
        
        self.training_active = True
        
        for gen in range(num_generations):
            print(f"\n🏁 GENERATION {self.generation + 1}/{num_generations}")
            
            # Run one generation
            self.population.run_generation(self.eval_genomes)
            self.generation += 1
            
            # Update best genome
            current_best = max(self.population.genomes, key=lambda g: g.fitness)
            if not self.best_genome or current_best.fitness > self.best_genome.fitness:
                print(f"🚗 NEW BEST MODEL! Fitness: {current_best.fitness:.2f}")
                self.best_genome = current_best.copy()
                self.max_fitness = current_best.fitness
            
            # Save best model
            if self.best_genome:
                self.save_best_model(self.best_genome)
            
            # Calculate statistics
            avg_fitness = sum(g.fitness for g in self.population.genomes) / len(self.population.genomes)
            self.fitness_history.append(avg_fitness)
            
            print(f"✅ Generation {self.generation} completed")
            print(f"   Best fitness: {self.max_fitness:.2f}")
            print(f"   Avg fitness: {avg_fitness:.2f}")
            
            # Check if user wants to stop
            if not self.training_active:
                print("Training stopped by user")
                break
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED: {self.generation} generations")
        print(f"Best fitness achieved: {self.max_fitness:.2f}")
        print(f"{'='*60}")
        self.training_active = False
        
    def eval_genomes(self, genomes):
        """Evaluate all genomes in the population with lap counting."""
        cars = []
        nets = []
        
        track_surface, track_overlay = TRACK_IMAGES[self.current_track]

        for genome in genomes:
            net = NeatModule(genome, use_sparse=False, trainable=False)
            car_group = pygame.sprite.GroupSingle(Car(self.current_track))
            cars.append(car_group)
            nets.append(net)
            genome.fitness = 0

        running = True
        frames = 0
        max_completed_laps = 0
        
        # Show starting message
        SCREEN.fill((0, 0, 0))
        font = pygame.font.SysFont(UI_FONT, 36)
        gen_text = font.render(f"Generation {self.generation + 1}", True, (255, 255, 0))
        track_text = font.render(f"Track: {TRACKS[self.current_track]['name']}", True, (255, 255, 255))
        SCREEN.blit(gen_text, (SCREEN_WIDTH//2 - gen_text.get_width()//2, SCREEN_HEIGHT//2 - 50))
        SCREEN.blit(track_text, (SCREEN_WIDTH//2 - track_text.get_width()//2, SCREEN_HEIGHT//2))
        pygame.display.flip()
        pygame.time.wait(1000)
        
        while running:
            frames += 1
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.training_active = False
                        return  # Exit early
                    elif event.key == pygame.K_r:
                        # Toggle radar visibility for all cars
                        for car_group in cars:
                            car_group.sprite.show_radar = not car_group.sprite.show_radar
                        print(f"Radar visualization: {'ON' if cars[0].sprite.show_radar else 'OFF'}")

            # Check stopping conditions
            all_dead = all(not car.sprite.alive for car in cars)
            max_laps_reached = any(car.sprite.laps >= LAPS_TO_COMPLETE for car in cars)
            
            if all_dead or max_laps_reached or frames > 6000:  # ~100 seconds at 60 FPS
                if max_laps_reached:
                    print(f"🏁 Car completed {LAPS_TO_COMPLETE} laps! Ending generation.")
                elif all_dead:
                    print(f"💀 All cars crashed. Ending generation.")
                else:
                    print(f"⏱️ Time limit reached ({frames} frames).")
                break

            # Draw track
            SCREEN.blit(track_surface, (0, 0))

            # Update and control cars
            for i, car_group in enumerate(cars):
                car = car_group.sprite
                car_group.update()
                
                if car.alive:
                    # Get neural network output
                    inputs = torch.tensor(car.get_data(), dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = nets[i](inputs).squeeze()
                    
                    # Control the car
                    car.direction = 0
                    if output[0] > 0.7:
                        car.direction = 1
                    if output[1] > 0.7:
                        car.direction = -1
                        
                    # Update fitness with multiple rewards
                    genome_fitness = genomes[i].fitness
                    
                    # Base fitness for staying alive and moving
                    genome_fitness += car.speed * 0.1
                    
                    # Bonus for distance traveled
                    genome_fitness += car.distance * 0.01
                    
                    # Big bonus for completing laps
                    if car.laps > 0:
                        genome_fitness += car.laps * 1000
                        
                    # Update max laps
                    if car.laps > max_completed_laps:
                        max_completed_laps = car.laps
                    
                    genomes[i].fitness = genome_fitness
                    
                    # Draw car (after radar lines so car appears on top)
                    car_group.draw(SCREEN)
                else:
                    if not car.dead_penalized:
                        genomes[i].fitness -= 50  # Death penalty
                        car.dead_penalized = True

            # Draw overlay
            SCREEN.blit(track_overlay, (0, 0))
            
            # Draw stats
            self.draw_training_stats(frames, max_completed_laps, len([c for c in cars if c.sprite.alive]))
            
            pygame.display.flip()
            clock.tick(60)
                
    def draw_training_stats(self, frames, max_laps, alive_cars):
        """Draw training statistics."""
        font = pygame.font.SysFont(UI_FONT, 20)
        
        stats = [
            f"Track: {TRACKS[self.current_track]['name']}",
            f"Generation: {self.generation + 1}",
            f"Best Fitness: {self.max_fitness:.1f}",
            f"Time: {frames//60}s",
            f"Max Laps: {max_laps}/{LAPS_TO_COMPLETE}",
            f"Cars Alive: {alive_cars}/{self.config.population_size}",
            "Press ESC to stop training",
            "Press R to toggle radar"
        ]
        
        y_offset = 10
        for line in stats:
            text_surf = font.render(line, True, UI_HIGHLIGHT)
            SCREEN.blit(text_surf, (SCREEN_WIDTH - text_surf.get_width() - 10, y_offset))
            y_offset += 25

    def run_best_model(self):
        """Run the best trained model."""
        if not self.best_genome:
            if not self.load_best_model():
                print("⚠️ No best model available! Train a model first.")
                return
                
        net = NeatModule(self.best_genome, use_sparse=False, trainable=False)
        car_group = pygame.sprite.GroupSingle(Car(self.current_track))
        car = car_group.sprite
        track_surface, track_overlay = TRACK_IMAGES[self.current_track]
        
        print(f"\n=== Running best model on {TRACKS[self.current_track]['name']} ===")
        print(f"Model fitness: {self.best_genome.fitness:.2f}")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_r:
                        car.show_radar = not car.show_radar
                        print(f"Radar visualization: {'ON' if car.show_radar else 'OFF'}")

            # Draw track
            SCREEN.blit(track_surface, (0, 0))
            
            # Update car (this draws radar lines)
            car_group.update()
            
            if car.alive:
                inputs = torch.tensor(car.get_data(), dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    output = net(inputs).squeeze()
                
                car.direction = 0
                if output[0] > 0.7:
                    car.direction = 1
                if output[1] > 0.7:
                    car.direction = -1
                    
                # Draw car (after radar lines)
                car_group.draw(SCREEN)
            
            # Draw overlay
            SCREEN.blit(track_overlay, (0, 0))
            
            # Draw stats
            self.draw_demo_stats(car)
            
            pygame.display.flip()
            clock.tick(60)
            
            if not car.alive:
                pygame.time.wait(2000)
                return
            
    def draw_demo_stats(self, car):
        """Draw demo statistics."""
        font = pygame.font.SysFont(UI_FONT, 20)
        
        stats = [
            f"Track: {TRACKS[self.current_track]['name']}",
            f"Generation: {self.generation}",
            f"Best Fitness: {self.max_fitness:.1f}",
            f"Laps: {car.laps}/{LAPS_TO_COMPLETE}",
            f"Speed: {car.speed:.1f}",
            f"Distance: {car.distance:.0f}",
            f"Alive: {'Yes' if car.alive else 'No'}",
            "Press ESC to return",
            "Press R to toggle radar"
        ]
        
        y_offset = 10
        for line in stats:
            text_surf = font.render(line, True, UI_HIGHLIGHT)
            SCREEN.blit(text_surf, (SCREEN_WIDTH - text_surf.get_width() - 10, y_offset))
            y_offset += 25
            
        # Draw radar sensor values
        if car.show_radar and hasattr(car, 'radars') and len(car.radars) == 4:
            radar_font = pygame.font.SysFont(UI_FONT, 16)
            radar_text = [
                f"Radar -60°: {car.radars[0][1]}",
                f"Radar -30°: {car.radars[1][1]}", 
                f"Radar 0°: {car.radars[2][1]}",
                f"Radar 30°: {car.radars[3][1]}"
            ]
            
            y_offset = SCREEN_HEIGHT - 120
            for line in radar_text:
                text_surf = radar_font.render(line, True, (200, 200, 255))
                SCREEN.blit(text_surf, (10, y_offset))
                y_offset += 20

def main_menu():
    """Main menu interface."""
    neatify_manager = NEATifyManager()
    
    # Create buttons
    track_buttons = []
    left_x = SCREEN_WIDTH // 2 - 250
    right_x = SCREEN_WIDTH // 2 + 50
    
    # Track selection buttons
    track_buttons.append(Button(left_x, 150, 200, 40, TRACKS["track1"]["name"]))
    track_buttons.append(Button(left_x, 200, 200, 40, TRACKS["track2"]["name"]))
    track_buttons.append(Button(right_x, 150, 200, 40, TRACKS["track3"]["name"]))
    track_buttons.append(Button(right_x, 200, 200, 40, TRACKS["track4"]["name"]))
    
    # Action buttons
    train_button = Button(SCREEN_WIDTH//2 - 100, 300, 200, 40, "Train 10 Generations")
    run_button = Button(SCREEN_WIDTH//2 - 100, 350, 200, 40, "Run Best Model")
    quit_button = Button(SCREEN_WIDTH//2 - 100, 400, 200, 40, "Quit")
    
    buttons = track_buttons + [train_button, run_button, quit_button]
    
    print("\n" + "="*60)
    print("MAIN MENU - READY")
    print("="*60)
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
                
            for i, button in enumerate(buttons):
                button.check_hover(mouse_pos)
                if button.handle_event(event):
                    if i < len(track_buttons):
                        track_id = list(TRACKS.keys())[i]
                        neatify_manager.current_track = track_id
                        neatify_manager.best_genome = None
                        print(f"\n🔁 Switched to {TRACKS[track_id]['name']}")
                        print(f"Starting position: {TRACKS[track_id]['start_pos']}")
                        neatify_manager.load_best_model()
                    elif button == train_button:
                        print("\n🏁 Starting 10-generation training...")
                        neatify_manager.train_multiple_generations(TRAINING_GENERATIONS)
                        print("✅ Training completed!")
                    elif button == run_button:
                        print("\n🏎️ Running best model...")
                        neatify_manager.run_best_model()
                        print("🏁 Demo completed!")
                    elif button == quit_button:
                        running = False
        
        # Draw UI
        SCREEN.fill((40, 40, 60))
        
        # Title
        title_font = pygame.font.SysFont(UI_FONT, 48)
        title = title_font.render("NEATify Self-Driving Car", True, UI_HIGHLIGHT)
        SCREEN.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 50))
        
        # Track info
        info_font = pygame.font.SysFont(UI_FONT, 28)
        track_info = f"Current: {TRACKS[neatify_manager.current_track]['name']}"
        track_text = info_font.render(track_info, True, (0, 255, 0))
        SCREEN.blit(track_text, (SCREEN_WIDTH//2 - track_text.get_width()//2, 100))
        
        # Starting position
        pos_font = pygame.font.SysFont(UI_FONT, 18)
        pos_text = f"Start: {TRACKS[neatify_manager.current_track]['start_pos']}"
        pos_surf = pos_font.render(pos_text, True, (200, 200, 255))
        SCREEN.blit(pos_surf, (SCREEN_WIDTH//2 - pos_surf.get_width()//2, 130))
        
        # Draw buttons
        for button in buttons:
            button.draw(SCREEN)
            
            # Highlight current track
            if button in track_buttons:
                track_id = list(TRACKS.keys())[track_buttons.index(button)]
                if track_id == neatify_manager.current_track:
                    pygame.draw.rect(SCREEN, UI_HIGHLIGHT, button.rect, 3, border_radius=5)
        
        # Draw training info
        help_font = pygame.font.SysFont(UI_FONT, 18)
        help_text = [
            f"Training: {TRAINING_GENERATIONS} generations per click",
            f"Target: {LAPS_TO_COMPLETE} laps or all cars crash",
            f"Population: {neatify_manager.config.population_size} cars per generation",
            "Press ESC during training to stop early",
            "Press R to toggle radar visualization"
        ]
        
        for i, text in enumerate(help_text):
            text_surf = help_font.render(text, True, (180, 255, 180))
            SCREEN.blit(text_surf, (SCREEN_WIDTH//2 - text_surf.get_width()//2, 460 + i*25))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    print("="*60)
    print("NEATIFY SELF-DRIVING CAR - MULTI-GENERATION TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Generations per training: {TRAINING_GENERATIONS}")
    print(f"  Laps to complete: {LAPS_TO_COMPLETE}")
    print(f"  Population size: 30 cars")
    print(f"  Tracks: 4 (with custom starting positions)")
    print(f"  Radar visualization: ON (Press R to toggle)")
    
    main_menu()