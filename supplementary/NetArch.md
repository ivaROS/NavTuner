# NavTuner Architecture

## IROS 2021 

The workflow of the NavTuner pipeline is as follows.

![workflow of NavTuner](https://github.com/ivaROS/NavTuner/blob/main/supplementary/images/workflow.png?raw=true)

We use different learning algorithms as our NavTuner, and compared their performance in the paper. Below is the details of the algorithms we use. In our paper, we use the laserscan as the input signal, and we tune different hyperparameters, including egocircle radius, global planning frequency, selection cost hysteresis, path switching blocking period, selection prefers initial plan, inflation distance, and the number of poses in the feasibility check.

- Linear Models

For linear models, we use only 1 linear layer, and the output dimension depends on the model type. For classifier models, it's the number of classes; and for regressor models, it's 1.

- Neural Network Models

For neural network models, our NavTuner has the following architecture. The numbers in the parenthesis indicates the output dimension of that layer, and the output dimension for the last linear layer depends on the model type. For classifier models, it's the number of classes; and for regressor models, it's 1.
![NN model architecture](https://github.com/ivaROS/NavTuner/blob/main/supplementary/images/NN.png?raw=true)

- Convolutional Neural Network Models

For convolutional neural network models, our NavTuner has the following architecture. The numbers in the parenthesis indicates the output dimension of that layer, and the output dimension for the last linear layer depends on the model type. For classifier models, it's the number of classes; and for regressor models, it's 1.
![CNN model architecture](https://github.com/ivaROS/NavTuner/blob/main/supplementary/images/CNN.png?raw=true)

- Deep Q Network Models

For deep Q network models, our NavTuner has the following architecture. Note that this is the case for 1D DQN, where we only tune 1 hyper-parameter. For multi-hyper-parameter cases, we use branches. The numbers in the parenthesis indicates the output dimension of that layer, and the output dimension for the last linear layer depends on the number of bins of that hyper-parameter.
![DQN model architecture](https://github.com/ivaROS/NavTuner/blob/main/supplementary/images/DQN.png?raw=true)

