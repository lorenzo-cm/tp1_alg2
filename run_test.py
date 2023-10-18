import subprocess

count_not_intersects = 0
count_all_ones = 0
python_executable = "python"

# Initializing the dictionary for the time accumulators
time_accumulator = {
    "TIME_GET_DATASET": 0.0,
    "TIME_GRAHAM_SCAN": 0.0,
    "TIME_HULL_TO_SEGMENTS": 0.0,
    "TIME_SWEEP_LINE_INTERSECTION": 0.0,
    "TIME_LINEAR_SEPARATION": 0.0,
    "TIME_LINEAR_CLASSIFICATION": 0.0
}

# Running the loop 10 times
for _ in range(20):
    result = subprocess.check_output([python_executable, "test.py"]).decode('utf-8')

    with open("output.txt", "a") as output_file:
        output_file.write(result)
        output_file.write("\n\n")
    
    for line in result.splitlines():
        if line.startswith("time_"):
            key, value = line.split(':')
            key = key.upper()
            time_accumulator[key] += float(value)
        elif "not intersects" in line:
            count_not_intersects += 1
        elif "METRICS: Accuracy: 1.0, Precision: 1.0, Recall: 1.0, F1 Score: 1.0" in line:
            count_all_ones += 1

print(f"Number of times it did not intersect: {count_not_intersects}")
print(f"How many times metrics were all 1 (after 'not intersects'): {count_all_ones}")
accuracy = int((count_all_ones * 100) / count_not_intersects) if count_not_intersects != 0 else 0
print(f"Accuracy of the algorithm: {accuracy}%")

for key, value in time_accumulator.items():
    print(f"{key}: {value}")

