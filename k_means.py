# this script reads the labels directory and loads all labels from all txt files therein
# it assumes all files have a format where each line is a 1-d YOLO box described as
# <class index> <center x> <width>

# it then performs k means and prints out the anchor widths


import torch
from glob import glob
from sys import argv


# for each point calculate the distance to each center and find the min (closest center for each point)
# return a parallel array (lenght num_points) containing a center assignment for each point
def assign_points(points, centers):

    k = len(centers)
    num_points = len(points)

    assignments = torch.empty(num_points)
    dist = torch.empty(k)

    for p in range(num_points):
        for c in range(k):
            dist[c] = abs((points[p]-centers[c]).item())
        assignments[p] = torch.argmin(dist)
    return assignments

# for each center, set it to the mean of its points
def update_centers(assignments, points, centers):

    k = len(centers)
    num_points = len(points)

    for c in range(k):
        # get the points belonging to this center
        pts = points[(assignments == c)]
        if len(pts) > 0:
            centers[c] = pts.mean()


# finds the squared distance from the centers to the points assigned to them to determine how well we did
def get_squared_distance(assignments, points, centers):

    k = len(centers)
    num_points = len(points)

    # a running sum
    s = 0

    for c in range(k):
        # get the points belonging to this center
        pts = points[(assignments == c)]
        dist_sqr = (pts - centers[c])**2
        s += dist_sqr.sum()

    return s


def k_means(points, k, num_iterations, print_everything):

    if print_everything:
        print(f"Points: {points}")

    # get the range of the points (max - min)
    mn = points.min()
    mx = points.max()
    rng = mx - mn

    # make k random centers within the range
    centers = mn + torch.rand(k)*rng

    if print_everything:
        print(f"Initial Centers: {centers}")

    assignments = assign_points(points, centers)

    initial_dist = get_squared_distance(assignments, points, centers)
    print(f"Initial distance: {initial_dist}")

    if print_everything:
        print("Starting k means...")

    for i in range(num_iterations):
        assignments = assign_points(points, centers)
        update_centers(assignments, points, centers)

    if print_everything:
        print("Done")

        print(f"Final Centers: {centers}")

    final_dist = get_squared_distance(assignments, points, centers)
    print(f"Final distance: {final_dist}")

    return centers


def run_simulation(k, num_points, num_iterations, print_everything):
    # make random points
    points = torch.rand(num_points)

    # run k means
    k_means(points, k, num_iterations, print_everything)





if __name__ == "__main__":

    filenames = glob("./data/labels/*.txt")
    widths = []

    for fname in filenames:
        with open(fname, "r") as f:
            lines = f.readlines()
            for line in lines:
                widths.append(float(line.strip('\n').split(' ')[1]))

    points = torch.tensor(widths)

    if len(argv) > 1:
        k = int(argv[1])
    else:
        k = 9

    results, _ = k_means(points, k, 100, False).sort()
    print(results)


