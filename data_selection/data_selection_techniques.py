import torch
import math

# https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py

 # ** Calculate confidence from prob_dist **
def least_confidence(prob_dist, sorted=False):
    """ 
    Returns the uncertainty score of an array using
    least confidence sampling in a 0-1 range where 1 is the most uncertain
    
    Assumes probability distribution is a pytorch tensor, like: 
        tensor([0.0321, 0.6439, 0.0871, 0.2369])
                
    Keyword arguments:
        prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
        sorted -- if the probability distribution is pre-sorted from largest to smallest
    """
    if sorted:
        simple_least_conf = prob_dist.data[0] # most confident prediction
    else:
        simple_least_conf = torch.max(prob_dist) # most confident prediction
                
    num_labels = prob_dist.numel() # number of labels
        
    normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels -1))
    
    return normalized_least_conf.item()

def least_confidence_batch(prob_dists):
    max_probs = torch.max(prob_dists, dim=1)[0]
    num_labels = prob_dists.size(1)
    return (1 - max_probs) * (num_labels / (num_labels - 1))

 # ** Calculate entropy from prob_dist **
def entropy_based(prob_dist):
    """ 
    Returns the uncertainty score of a probability distribution using
    entropy 
    
    Assumes probability distribution is a pytorch tensor, like: 
        tensor([0.0321, 0.6439, 0.0871, 0.2369])
                
    Keyword arguments:
        prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
        sorted -- if the probability distribution is pre-sorted from largest to smallest
    """
    log_probs = prob_dist * torch.log2(prob_dist) # multiply each probability by its base 2 log
    raw_entropy = 0 - torch.sum(log_probs)

    normalized_entropy = raw_entropy / math.log2(prob_dist.numel())
    
    return normalized_entropy.item()

def entropy_based_batch(prob_dist):
    eps = 1e-10
    prob_dist = prob_dist + eps
    
    # Compute entropy for entire batch at once
    log_probs = prob_dist * torch.log2(prob_dist)  # shape: (batch_size, num_classes)
    raw_entropy = -torch.sum(log_probs, dim=1)     # shape: (batch_size,)
    
    # Normalize by log2(num_classes)
    num_classes = prob_dist.size(1)
    normalized_entropy = raw_entropy / math.log2(num_classes)
    
    return normalized_entropy


if __name__ == "__main__":
    test_tensor = torch.tensor([[0.5, 0.4, 0.1], [0.4, 0.4, 0.2]])
    print(least_confidence(test_tensor[0]))
    print(least_confidence(test_tensor[1]))
    print(least_confidence_batch(test_tensor))


    print(entropy_based(test_tensor[0]))
    print(entropy_based(test_tensor[1]))
    print(entropy_based_batch(test_tensor))
