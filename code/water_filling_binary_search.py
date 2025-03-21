import numpy as np
from matplotlib import pyplot as plt


def calculate_target(alpha, x): # calculate the target function
    return -(np.log2(alpha + x).sum())

def fill_water(alpha, total_water, precision, track = False):
    lower_bound = 1/(total_water + max(alpha)) # Initialize the lower bound of nu
    upper_bound = 1/min(alpha) # Initialize the upper bound of nu
    iteration = 0
    track_x = []
    trakc_target = []

    while upper_bound - lower_bound > precision:
        nu = (lower_bound + upper_bound)/2 # Update nu with middle of the bounds
        x = np.maximum(0, 1/nu - alpha) # Update x, ensure x > 0
        water_sum = np.sum(x) # Calculate the water sum
        iteration += 1
        # Update nu
        if track:
            track_x.append(x)
            trakc_target.append(calculate_target(alpha, x))
        if water_sum > total_water:
            lower_bound = nu 
        else:
            upper_bound = nu
            
    nu_opt = (upper_bound + lower_bound)/2 # The optimal nu
    x_opt = np.maximum(0, 1/nu_opt - alpha) # the optimal x
    print(f"Iterations : {iteration}")
    
    if track:
        return track_x, trakc_target
    
    return np.array(x_opt), iteration

def visualize_water(alpha, x, horizontal_line, iteration):
    alpha = alpha.squeeze() # Ensure alpha only has 1-axis
    x = x.squeeze() # Ensure x only has 1-axis
    x_range = range(1, x.shape[0]+1) # num of x
    plt.xticks(x_range)
    plt.bar(x_range, alpha, color='#ff9966',
    width=1.0, edgecolor='#ff9966')
    plt.bar(x_range, x, bottom=alpha, color='#4db8ff', width=1.0)
    plt.axhline(y=horizontal_line,linewidth=1, color='k')
    plt.text(0, horizontal_line , f"Horizontal line : {horizontal_line}")
    plt.text(0, 0, f"Sum of x : {x.sum()}")
    plt.text(8, 0, f"iterations : {iteration}")
    plt.text(5, 0.08, f"precision : {precision}")
    plt.title("Visualization of Binary Search")
    plt.show()

def monkey_search(alpha):
    # random return x s.t. 1^T x = 1 and x > 0
    while True:
        monkey_solution = np.random.dirichlet(np.ones(dimension),size=1).reshape(-1,1)
        if np.less(monkey_solution, 0).any():
            continue
        return monkey_solution

def visualize_monkey_search(alpha, monkey_amount, optimal):
    monkey_solutions = [calculate_target(alpha, monkey_search(alpha)) \
                        for _ in range(monkey_amount)]
    plt.scatter(range(monkey_amount), monkey_solutions)
    plt.axhline(y=optimal,linewidth=1, color='r')
    plt.text(0, optimal, f"Target : {optimal}")
    plt.title("Mokey Search VS Binary Search")
    plt.show()

def visualize_track(track_x, track_target):
    x = range(len(track_x))
    s = np.sum(track_x, axis=1)
    plt.plot(x, track_target, label = "Target")
    plt.plot(x, s, label = "sum of x")
    plt.xlabel("iterations")
    plt.legend()
    plt.show()    



if __name__ == "__main__":
    alpha_range = [0.0, 1.0]
    total_water = 1.0
    dimension = 10
    precision = 1e-8

    np.random.seed(12345)

    alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=(dimension, 1))

    print(alpha)
    
    x, iteration = fill_water(alpha=alpha, total_water=total_water, precision=precision)
    print(x)

    print("iterations : ", iteration)
    print("sum of x : ", np.sum(x))
    
    horizontal_line = np.min(alpha + x)
    print(horizontal_line)

    target = calculate_target(alpha, x)
    print("target : ", target)

    visualize_water(alpha, x, horizontal_line, iteration)
    
    visualize_monkey_search(alpha, 1000, target)

    track_x, track_target = fill_water(alpha, total_water, precision, track=True)
    visualize_track(track_x, track_target)
    
    



