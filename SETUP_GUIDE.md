# NEATIFY Demos - Setup Guide

## Quick Setup on New Machine

### 1. Clone the Repository
```bash
git clone https://github.com/Supreme-Shrestha/NEATIFY_DEMOS.git
cd NEATIFY_DEMOS
```

### 2. Install Dependencies
```bash
pip install neatify-ai pygame torch numpy
```

### 3. Available Demos

#### A. XOR Problem (Simple)
```bash
python xor_demo.py
```

#### B. Function Approximation
```bash
python function_approx_demo.py
```

#### C. Self-Driving Car (Single Machine)
```bash
python self_driving_car/car_evolution.py
```

#### D. Distributed Training (Multiple Machines)

**On Master Machine:**
```bash
python self_driving_car/neatify_master.py --port 5000 --generations 50
```

**On Worker Machines:**
```bash
python self_driving_car/neatify_worker.py --master <MASTER_IP> --port 5000
```

Replace `<MASTER_IP>` with the IP address of your master machine.

---

## Merge Into Existing Repository

If you already have a local copy and want to update it:

```bash
# Navigate to your existing repo
cd path/to/NEATIFY_DEMOS

# Pull latest changes
git pull origin main
```

---

## Distributed Training Setup

### Network Requirements
- All machines must be on the same LAN
- Master machine's firewall must allow incoming connections on port 5000
- All machines need the `self_driving_car/` folder with track images

### File Structure Required on Each Machine
```
NEATIFY_DEMOS/
├── self_driving_car/
│   ├── neatify_master.py      # Run on master only
│   ├── neatify_worker.py      # Run on workers
│   ├── simulation.py
│   ├── config.py
│   └── tracks/                # Must be present on all machines
│       ├── track1.png
│       ├── track1-overlay.png
│       ├── track2.png
│       ├── ...
│       └── car4.png
```

### Testing Locally First
Before deploying to multiple machines, test on one machine:

**Terminal 1 (Master):**
```bash
python self_driving_car/neatify_master.py --port 5000 --generations 5
```

**Terminal 2 (Worker):**
```bash
python self_driving_car/neatify_worker.py --master 127.0.0.1 --port 5000
```

You should see the worker connect and training begin!

---

## Troubleshooting

### "Connection Refused"
- Master might not be ready yet - wait 3-5 seconds after starting master
- Check firewall settings on master machine
- Verify port 5000 is not already in use

### "Module not found"
```bash
pip install neatify-ai pygame torch numpy
```

### Worker crashes immediately
- Ensure `tracks/` folder exists on worker machine
- Check that pygame can run in headless mode (SDL_VIDEODRIVER=dummy is set)

---

## Repository Link
https://github.com/Supreme-Shrestha/NEATIFY_DEMOS
