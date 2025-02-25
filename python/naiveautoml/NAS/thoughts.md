# Loading the CIFAR-10 dataset
I trained 27 structures with early stop when for 50 epochs there was no increase found in the validation accuracy. Then I tested the same architectures with one single epoch and checked the same metric. 
The results were interesting, out of the top 5 architectures found in the 50-epochs-without-increase training, 4 were identified as well as the 5 most accurate in the One-Epoch approach. This supports the method of using such a strategy to reduce computational load since the 50 epochs training took aproximmately 4 hours, against 4 minutes in the One-Epoch.
Perhaps those architectures with the highest accuracies could be used as the basis for next steps in training. 

# Ideas
- Perhaps an early stopping of testing the architectures could be implemented once a certain % of all the pre-defined architectures are tested and none achieve higher results than the best one found yet. 
- If the top 5 architectures are later on tested with different parameters, isn't that a sort of overly simplified evolutionary algorithm?
- What will happen if each parameter found for the best architecture is isolated? 
- What's the most time expensive parameter? What is its tradeoff? 
- How to define the search space? 
- Is there any property from the training set that could be used to delimit the search space?
- What about the solver and the activation functions?