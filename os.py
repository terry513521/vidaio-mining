import shutil
import subprocess
import psutil


def bytes_to_human(num):
    """Convert bytes to a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"


print("=" * 60)
print("SYSTEM RESOURCES")
print("=" * 60)

# --------------------------------------------------
# CPU
# --------------------------------------------------
print("\nCPU")
print("-" * 60)

print(f"Physical Cores : {psutil.cpu_count(logical=False)}")
print(f"Logical Cores  : {psutil.cpu_count(logical=True)}")
print(f"CPU Usage      : {psutil.cpu_percent(interval=1):.1f}%")

freq = psutil.cpu_freq()
if freq:
    print(f"Current Freq   : {freq.current:.0f} MHz")
    print(f"Max Freq       : {freq.max:.0f} MHz")

# --------------------------------------------------
# RAM
# --------------------------------------------------
mem = psutil.virtual_memory()

print("\nRAM")
print("-" * 60)

print(f"Total          : {bytes_to_human(mem.total)}")
print(f"Used           : {bytes_to_human(mem.used)}")
print(f"Available      : {bytes_to_human(mem.available)}")
print(f"Usage          : {mem.percent:.1f}%")

# --------------------------------------------------
# SWAP
# --------------------------------------------------
swap = psutil.swap_memory()

print("\nSWAP")
print("-" * 60)

print(f"Total          : {bytes_to_human(swap.total)}")
print(f"Used           : {bytes_to_human(swap.used)}")
print(f"Free           : {bytes_to_human(swap.free)}")
print(f"Usage          : {swap.percent:.1f}%")

# --------------------------------------------------
# DISK
# --------------------------------------------------
disk = psutil.disk_usage("/")

print("\nDISK (/)")
print("-" * 60)

print(f"Total          : {bytes_to_human(disk.total)}")
print(f"Used           : {bytes_to_human(disk.used)}")
print(f"Free           : {bytes_to_human(disk.free)}")
print(f"Usage          : {disk.percent:.1f}%")

# --------------------------------------------------
# GPU
# --------------------------------------------------
print("\nGPU")
print("-" * 60)

if shutil.which("nvidia-smi"):
    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]).decode().strip().splitlines()

        for i, line in enumerate(result):
            name, total, used, free, util = [x.strip() for x in line.split(",")]

            print(f"GPU {i}")
            print(f"  Name         : {name}")
            print(f"  VRAM Total   : {int(total):,} MB")
            print(f"  VRAM Used    : {int(used):,} MB")
            print(f"  VRAM Free    : {int(free):,} MB")
            print(f"  GPU Usage    : {util}%")

    except Exception as e:
        print("Unable to query NVIDIA GPU.")
        print(e)
else:
    print("No NVIDIA GPU detected.")

# --------------------------------------------------
# Docker
# --------------------------------------------------
print("\nDOCKER")
print("-" * 60)

if shutil.which("docker"):
    try:
        running = subprocess.check_output(
            ["docker", "ps", "-q"]
        ).decode().splitlines()

        total = subprocess.check_output(
            ["docker", "ps", "-aq"]
        ).decode().splitlines()

        print(f"Running Containers : {len(running)}")
        print(f"Total Containers   : {len(total)}")

        if running:
            print("\nContainer Usage")
            print("-" * 60)

            stats = subprocess.check_output([
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
            ]).decode()

            print(stats)

    except Exception:
        print("Docker is installed but cannot be queried.")
else:
    print("Docker is not installed.")

print("\n" + "=" * 60)
