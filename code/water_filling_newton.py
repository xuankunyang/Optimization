import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track


def calculate_target(alpha, x):
    return -(np.log2(alpha + x).sum())

def backtracking_line_search(x, direction ,total_waterm, beta = 0.5, max_iter = 1000):
    step = 0.01

    for i in range(max_iter):
        if np.sum(x + step * direction) > total_water :
            step = step * beta
        else:
            break
        
    return step

            
def fill_water(alpha, total_water, precision, max_iter, F_track = False):
    n = len(alpha)

    x = np.ones((n, 1))*(total_water / n) # Initialize x

    track_x = []
    track_target = []
    
    for t in track(range(max_iter), description = "进度..."):
        nu = 1 / np.min(alpha + x) # Calculate nu

        # for i in range(n):
        #     if nu < 1 / alpha[i]:
        #         x[i] = 1 / nu - alpha[i]
        #     else: 
        #         x[i] = 0
        x = np.maximum(0, 1 / nu - alpha) # Update x with nu
        if F_track:
            track_x.append(x)
            track_target.append(calculate_target(alpha, x))
        
        grad = -1 / (alpha + x) # Calculate gradient

        nt = alpha + x # The Newton step
        
        step = backtracking_line_search(x, nt, total_water) # Calculate the step
        x = x + step * nt # Update x
    
    if F_track:
        return track_x, track_target
        
    return np.array(x)
        


def visualize_water(alpha, x, horizontal_line, iteration):
    alpha = alpha.squeeze()
    x = x.squeeze()
    x_range = range(1, x.shape[0]+1)
    plt.xticks(x_range)
    plt.bar(x_range, alpha, color='#ff9966',
    width=1.0, edgecolor='#ff9966')
    plt.bar(x_range, x, bottom=alpha, color='#4db8ff', width=1.0)
    plt.axhline(y=horizontal_line,linewidth=1, color='k')
    plt.text(0, horizontal_line, f"Horizontal line : {horizontal_line}")
    plt.text(0, 0, f"Sum of x : {x.sum()}")
    plt.text(8, 0, f"iterations : {iteration}")
    plt.title("Visualization of Newton Method")
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
    plt.title("Mokey Search VS Newton Method")
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
    precision = 1e-9
    iteration = 200


    np.random.seed(12345)

    alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=(dimension, 1))

    print(alpha)
    
    x = fill_water(alpha=alpha, total_water=total_water, precision=precision, max_iter=iteration)
    print(x)
    print("sum of x : ", np.sum(x))
    
    horizontal_line = np.min(alpha + x)
    print(horizontal_line)

    target = calculate_target(alpha, x)
    print("target : ", target)

    visualize_water(alpha, x, horizontal_line, iteration)
    
    visualize_monkey_search(alpha, 1000, target)
    
    track_x, track_target = fill_water(alpha=alpha, total_water=total_water, precision=precision, max_iter=iteration, F_track=True)
    visualize_track(track_x, track_target)


