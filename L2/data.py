import subprocess
import csv
import time
from datetime import datetime

def run_astar(mode, n):
    results = []
    for i in range(n):
        print(f"Running mode {mode}, iteration {i+1}/{n}")
        start_time = time.time()
        try:
            # Run the Go program and capture output
            result = subprocess.run(["./15puzzle", str(mode)], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            output = result.stdout
            
            # Parse the output
            data = {
                "mode": mode,
                "iteration": i+1,
                "timestamp": datetime.now().isoformat(),
                "heuristic": None,
                "solution_length": None,
                "visited_states": None,
                "time_taken": None,
                "solution_steps": None
            }
            
            # Process each heuristic's results
            for heuristic in ["Manhattan distance", "Linear Conflict"]:
                if heuristic in output:
                    lines = output.split('\n')
                    for j, line in enumerate(lines):
                        if heuristic in line:
                            current_data = data.copy()
                            current_data["heuristic"] = heuristic
                            
                            # Extract solution length
                            if "Solution found in" in lines[j+1]:
                                parts = lines[j+1].split()
                                current_data["solution_length"] = int(parts[3])
                            
                            # Extract visited states
                            if "Visited" in lines[j+2]:
                                parts = lines[j+2].split()
                                current_data["visited_states"] = int(parts[1])
                            
                            # Extract time taken
                            if "Time taken:" in lines[j+3]:
                                parts = lines[j+3].split(": ")
                                current_data["time_taken"] = parts[1].strip()
                            
                            # Extract solution steps
                            if "Solution steps:" in lines[j+4]:
                                parts = lines[j+4].split(": ")
                                current_data["solution_steps"] = parts[1].strip()
                            
                            results.append(current_data)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running mode {mode}: {e}")
            continue
        
    return results

def main():
    # Ask for number of iterations
    n = int(input("Enter the number of iterations to run for each mode: "))
    
    # Prepare CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"astar_results_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'mode', 
            'iteration', 
            'timestamp', 
            'heuristic', 
            'solution_length', 
            'visited_states', 
            'time_taken',
            'solution_steps'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Run for each mode
        for mode in [3, 4, 5]:
            results = run_astar(mode, n)
            for result in results:
                writer.writerow(result)
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()