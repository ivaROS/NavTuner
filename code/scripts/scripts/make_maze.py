from __future__ import print_function
from builtins import range
from builtins import object
import random
import numpy as np
import matplotlib.pyplot as plt
# import skimage.transform.resize as imresize
# import cv2.resize as imresize
from PIL import Image
import cv2
import yaml


class Maze(object):
    def __init__(self, h, w, seed=0, path=None, hm=0, wm=0):
        self.random = random.Random()
        self.random.seed(seed)
        self.nprandom = np.random.RandomState(seed=seed)
        self.walls = np.ones((2 * w + 1, 2 * h + 1))
        self.walls[1::2, 1::2] = 0.
        self.visited = [[False] * w + [True] for _ in range(h)] + [[True] * (w + 1)]
        self.walk(2 * self.random.randrange(w) + 1, 2 * self.random.randrange(h) + 1)
        maze_image = Image.fromarray(np.array(1 - self.walls[1:-1, 1:-1], dtype=np.float))
        maze_image = maze_image.resize((40, 40), resample=Image.NEAREST)
        maze_image = np.array(maze_image)
        kernel = np.ones((3, 3))
        maze_image = cv2.erode(1 - maze_image, kernel, iterations=1)
        self.walls = maze_image
        '''
        if path is not None:
            maze_image = Image.fromarray(np.array(1 - self.walls, dtype=np.float))
            # maze_image = np.ones((int(wm/0.1), int(hm/0.1)))
            # points = np.where(self.walls == 1)
            # maze_image[]
            maze_image = maze_image.resize((4 * w + 2, 4 * h + 2), resample=Image.NEAREST)
            maze_image = np.array(maze_image)
            kernel = np.ones((2, 2))
            maze_image = cv2.erode(1 - maze_image, kernel, iterations=1)
            # maze_image[maze_image>0.3] = 1.
            maze_image = 205 * (1 - maze_image)
            maze_image = np.array(maze_image, dtype=np.uint8)
            maze_image = np.transpose(maze_image)
            maze_image = np.flipud(maze_image)
            # maze_image = Image.fromarray(np.array(np.array(maze_image, dtype=np.float)/255., dtype=np.bool))
            # maze_image.mode = '1'
            # maze_image.save(path + '.pgm')
            cv2.imwrite(path + '.pgm', maze_image)
            maze_yaml = {'image': path + '.pgm', 'resolution': 0.1,
                         'origin': [-9.7500000, -9.7500000, 0.000000], 'negate': 0,
                         'occupied_thresh': 0.65, 'free_thresh': 0.196}
            yaml.dump(maze_yaml, open(path + '.yaml', 'wb'))
        '''

    def walk(self, x, y):
        self.visited[y // 2][x // 2] = True
        d = [(x - 2, y), (x, y + 2), (x + 2, y), (x, y - 2)]
        self.random.shuffle(d)
        for (xx, yy) in d:
            if self.visited[yy // 2][xx // 2]:
                continue
            if xx == x:
                # RUR.we.toggle_wall(x + 1, min(y, yy) + 1, "north")
                self.walls[x, min(y, yy) + 1] = 0
            elif yy == y:
                # RUR.we.toggle_wall(min(x, xx) + 1, y + 1, "east")
                self.walls[min(x, xx) + 1, y] = 0
            self.walk(xx, yy)


# '''
if __name__ == '__main__':
    maze = Maze(20, 20, 10)
    maze_image = Image.fromarray(255 * np.array(1 - maze.walls, dtype=np.uint8))
    maze_image.save('test.pgm')
    print()
# '''
