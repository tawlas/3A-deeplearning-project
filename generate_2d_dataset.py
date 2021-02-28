import numpy as np
from tqdm import tqdm
import cv2
from ompl import base as ob
from ompl import geometric as og
import matplotlib.pyplot as plt
import h5py

# This script generates a dataset for the 2d path planning problem.
# Two main parameters have to be chosen:
#   1) The number of data in the dataset (n_problems)
#   2) Where to save the dataset (datasetpath)
#
# Put visualization to True if you want to visualize the generated problem
# and the associated solution.

def main(n_images=1000, output):
	n_problems = n_images
	datasetpath  = os.path.join(output, "2d_dataset.hdf5")
	visualization = False

	map_size = [198,155]
	x_dim = map_size[0]*map_size[1]+4
	ξ_dim = 2*100

	f = h5py.File(datasetpath,'w')
	dataset = f.create_dataset("dataset", (n_problems, x_dim+ξ_dim), compression= "lzf")

	image = cv2.imread("2d_dataset_generation/4th_floor_images/4th_floor_edited.jpg", 0)
	_, binary_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

	for index_data in tqdm(range(n_problems)):
		_, binary_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

		n_obstacles = np.random.randint(1,11)
		for i in range(n_obstacles):
			pos = np.random.random(2)*0.6+0.2 # biais to avoid obstacles outside the map 
			pos[0] = int(np.floor(map_size[0]*pos[0]))
			pos[1] = int(np.floor(map_size[1]*pos[1]))

			binary_img[int(pos[1]-5):int(pos[1]+5), int(pos[0]-5):int(pos[0]+5)] = 0
			
		def isStateValid(state):
			return binary_img[int(np.floor(map_size[1]*state[1])), int(np.floor(map_size[0]*state[0]))] > 0

		# Use OMPL to find a trajectory solution.
		# If no solution is found, new start and goal are sampled.
		
		path_found = False 
		while(not(path_found)):
			space = ob.RealVectorStateSpace(2)
	
			bounds = ob.RealVectorBounds(2)
			bounds.setLow(0)
			bounds.setHigh(1)
			space.setBounds(bounds)

			start = ob.State(space)
			goal  = ob.State(space)
		
			start_found = False
			while(not(start_found)):
				start_x = np.random.random()
				start_y = np.random.random()
				start_found = binary_img[int(np.floor(start_y * map_size[1])), int(np.floor(start_x * map_size[0]))] > 0
			
			start[0] = start_x
			start[1] = start_y

			goal_found = False
			while(not(goal_found)):
				goal_x = np.random.random()
				goal_y = np.random.random()
				goal_found = binary_img[int(np.floor(goal_y * map_size[1])), int(np.floor(goal_x * map_size[0]))] > 0
		
			goal[0] = goal_x
			goal[1] = goal_y

			si= ob.SpaceInformation(space)
			si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
			si.setup()
			
			pdef = ob.ProblemDefinition(si)
			pdef.setStartAndGoalStates(start, goal)
			planner = og.RRTConnect(si)
			planner.setProblemDefinition(pdef)
			planner.setup()
			solved = planner.solve(1.0)

			# If a solution is found, it is simplified and interpolated
			# to have 100 waypoints.
			if solved:
				simplifier = og.PathSimplifier(si)
				path = pdef.getSolutionPath()
				simplifier.simplify(path,1)
				path.interpolate(100)

				ξ = np.zeros((2,100))
				for k in range(100):
					ξ[0,k]=path.getState(k)[0]
					ξ[1,k]=path.getState(k)[1]

				start_goal = np.asarray([start[0], start[1], goal[0], goal[1]])
				
				x = np.concatenate([binary_img.reshape(map_size[0]*map_size[1]), start_goal])
				dataset[index_data] = np.concatenate([x,ξ.reshape(ξ_dim)])

				path_found = True

				if visualization:
					plt.imshow(binary_img)
					print(binary_img.shape)
					plt.scatter([map_size[0]*start[0], map_size[0]*goal[0]], [map_size[1]*start[1], map_size[1]*goal[1]])
					plt.plot(map_size[0]*ξ[0], map_size[1]*ξ[1])
					plt.show()

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--n_images",
        "-n",
        dest="n_images",
        required=True,
        help="Number of images to generate"
    )
	arg_parser.add_argument(
        "--output",
        "-o",
        dest="output",
        required=True,
        help="Path where dataset is stored."
    )

    args = arg_parser.parse_args()
    main(args.n_images, args.output)