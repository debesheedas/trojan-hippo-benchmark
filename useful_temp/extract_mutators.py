#!/usr/bin/env python3
"""Extract mutator responses from log file."""

import re

log_file = "/Users/ddas/Desktop/Debeshee/Thesis/Fresh/memory-agent-security-benchmark/logs/adaptive_benchmark_20251104_211010.log"
output_file = "/Users/ddas/Desktop/Debeshee/Thesis/Fresh/memory-agent-security-benchmark/candidates.log"

print("Reading log file...")
with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

mutator_responses = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Check if this line contains the mutator response marker
    if "[MUTATOR] Raw Response:" in line:
        # Start collecting the response
        response_lines = []
        i += 1  # Move to next line
        
        # Collect lines until we hit the separator line (===...) 
        while i < len(lines):
            current_line = lines[i]
            # Stop when we hit a separator line
            if "=" * 40 in current_line:
                break
            response_lines.append(current_line.rstrip('\n'))
            i += 1
        
        # Store this mutator response
        mutator_responses.append('\n'.join(response_lines))
    
    i += 1

print(f"Found {len(mutator_responses)} mutator responses")

# Write to output file
print(f"Writing to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for idx, response in enumerate(mutator_responses, 1):
        f.write("=" * 80 + "\n")
        f.write(f"MUTATOR RESPONSE #{idx}\n")
        f.write("=" * 80 + "\n")
        f.write(response)
        f.write("\n\n")

print(f"Wrote {len(mutator_responses)} mutator responses to {output_file}")
print("Done!")

