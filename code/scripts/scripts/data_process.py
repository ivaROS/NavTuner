from builtins import range
import pickle
import numpy as np


read = '/home/haoxin/data/training/max_depth/'
save = '/home/haoxin/data/training/combined/'

best_config = pickle.load(open(read + 'best_configs.pickle'))

suffix = 'classifier'
filename = read + suffix + '_alldata.pickle'
data = pickle.load(open(filename, 'r'))

# classifier: 0.75 (1, 4); 1.0 (3, 3); 1.25(5, 2); 1.5(6, 2)
# regressor: 0.75 (1.5, 1.); 1.0 (2.5, 0.5); 1.25(3.5, 0.25); 1.5(4.0, 0.25)
lookup = {'classifier': {1: 4, 3: 3, 5: 2, 6: 2}, 'regressor': {1.5: 1., 2.5: .5, 3.5: .25, 4.:.25}}
v1 = data['max_depth']['Y']
v2 = np.array(v1)
for i in range(len(v2)):
    v2[i] = lookup[suffix][v2[i]]
new = {'X': data['max_depth']['X'], 'Y1': v1, 'Y2': v2}
filename = save + suffix + '_alldata.pickle'
new = {'combined': new}
pickle.dump(new, open(filename, 'w'))
