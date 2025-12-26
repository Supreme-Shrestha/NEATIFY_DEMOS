
import os
import math
import sys
import pygame
import torch
from config import SCREEN_WIDTH, SCREEN_HEIGHT, TRACKS, SCRIPT_DIR

# Initializing pygame just for image handling if not already done
if not pygame.get_init():
    pygame.init()

class SimulationManager:
    """Manages track resources and headless simulation."""
    
    def __init__(self):
        self.track_surfaces = {}
        self.track_overlays = {}
        self.load_all_tracks()
        
    def load_all_tracks(self):
        """Preload all track images."""
        tracks_folder = os.path.join(SCRIPT_DIR, "tracks")
        for name in TRACKS:
            track_path = os.path.join(tracks_folder, f"{name}.png")
            
            if not os.path.exists(track_path):
                print(f"Error: Track {name} not found at {track_path}")
                continue
                
            try:
                # Load image - convert() usually requires a display, but strictly 
                # for pixel access we might be okay, or we assume a dummy video driver 
                # is set by the worker if headless.
                surface = pygame.image.load(track_path)
                self.track_surfaces[name] = surface
                
                # We don't necessarily need overrides for collision, but good to have
            except Exception as e:
                print(f"Failed to load track {name}: {e}")

    def get_track_data(self, track_name):
        return self.track_surfaces.get(track_name)


class Car(pygame.sprite.Sprite):
    """Self-driving car logic."""
    
    def __init__(self, track_name, track_surface, start_pos=None):
        super().__init__()
        track_info = TRACKS[track_name]
        self.start_pos = start_pos if start_pos else track_info["start_pos"]
        self.border_color = track_info["border_color"]
        self.track_surface = track_surface
        
        # Load car sprite specifically
        car_path = os.path.join(SCRIPT_DIR, "tracks", "car4.png")
        if os.path.exists(car_path):
             self.original_image = pygame.image.load(car_path)
             # Scale down if needed? The original code didn't scale explicitely unless loaded differently
        else:
             self.original_image = pygame.Surface((30, 15))
             self.original_image.fill((0, 150, 255))
        
        self.image = self.original_image
        self.rect = self.image.get_rect(center=self.start_pos)
        
        # Physics setup
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
        
        # Lap tracking code
        self.start_line_pos = self.start_pos
        self.near_start_line = False
        self.passed_start_line = False

    def update(self):
        if not self.alive:
            return
            
        self.radars.clear()
        self.drive()
        self.rotate()
        self.check_sensors()
        self.check_collision()
        self.check_lap()
        
        self.distance += abs(self.speed)
        self.speed = math.sqrt(self.vel_vector.x**2 + self.vel_vector.y**2) * 2.5

    def drive(self):
        self.rect.center += self.vel_vector * 2.5

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def check_sensors(self):
        """Radar logic."""
        radar_angles = (-60, -30, 0, 30)
        for angle in radar_angles:
            self.cast_radar(angle)

    def cast_radar(self, radar_angle):
        length = 0
        x, y = self.rect.center
        
        while length < 200:
            # Calculate beam end point
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

            # Check bounds
            if not (0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT):
                break
            
            # Check pixel color (collision)
            if self.track_surface:
                try:
                    pixel_color = self.track_surface.get_at((x, y))
                    if pixel_color[:3] == self.border_color:
                        break
                except IndexError:
                    break
            
            length += 1
            
        dist = int(math.sqrt((self.rect.center[0] - x) ** 2 + (self.rect.center[1] - y) ** 2))
        self.radars.append([radar_angle, dist])

    def check_collision(self):
        length = 25 # car half length roughly
        # Check two corners
        corners = [18, -18]
        for d_angle in corners:
            cx = int(self.rect.center[0] + math.cos(math.radians(self.angle + d_angle)) * length)
            cy = int(self.rect.center[1] - math.sin(math.radians(self.angle + d_angle)) * length)
            
            if not (0 <= cx < SCREEN_WIDTH and 0 <= cy < SCREEN_HEIGHT):
                self.alive = False
                return

            if self.track_surface:
                 try:
                    if self.track_surface.get_at((cx, cy))[:3] == self.border_color:
                        self.alive = False
                        return
                 except IndexError:
                    self.alive = False

    def check_lap(self):
        dx = self.rect.centerx - self.start_line_pos[0]
        dy = self.rect.centery - self.start_line_pos[1]
        desc_dist = math.sqrt(dx*dx + dy*dy)
        
        if desc_dist < 50:
            if not self.near_start_line:
                self.near_start_line = True
        else:
            if self.near_start_line and not self.passed_start_line:
                self.passed_start_line = True
        
        if self.passed_start_line and desc_dist > 100:
            self.laps += 1
            self.passed_start_line = False
            self.near_start_line = False

    def get_data(self):
        """Return normalized sensor data."""
        # Radars are 0-200. Speed is approx 0-6.
        return [r[1] for r in self.radars] + [self.speed]

def evaluate_car_fitness(net, car, max_frames=2000):
    """Run a single car simulation until crash or timeout."""
    frames = 0
    while car.alive and frames < max_frames:
        # Get input
        input_data = torch.tensor(car.get_data(), dtype=torch.float32).unsqueeze(0)
        
        # Inference
        try:
            with torch.no_grad():
                 raw_output = net(input_data)
                 output = raw_output.flatten()
        except Exception as e:
            print(f"\n❌ FORWARD PASS ERROR: {e}")
            print(f"  Input Shape: {input_data.shape} (Expected [1, 5])")
            import traceback
            traceback.print_exc()
            raise e
        
        # Control - handle any number of outputs safely
        car.direction = 0
        if output.numel() >= 1:
            if output[0].item() > 0.7: 
                car.direction = 1
            elif output.numel() >= 2 and output[1].item() > 0.7: 
                car.direction = -1
        
        car.update()
        frames += 1
        
        # Stop if stuck (optional, not in original but good practice)
    
    # Calculate Fitness
    fitness = 0
    fitness += car.speed * 0.1
    fitness += car.distance * 0.01
    fitness += car.laps * 1000
    if not car.alive and not car.dead_penalized:
        fitness -= 50
        
    return fitness
