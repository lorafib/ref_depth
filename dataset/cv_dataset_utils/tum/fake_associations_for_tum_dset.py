import os
import sys

fps = 15.0
delta_per_frame = 1.0/fps

num_images  = int(sys.argv[1])

start_time = 1674552821

with open(f"./associations{num_images}.txt", "w") as f:
    for i in range(num_images):
        f.write(f"{i:04d} {start_time+i*delta_per_frame}\n")





