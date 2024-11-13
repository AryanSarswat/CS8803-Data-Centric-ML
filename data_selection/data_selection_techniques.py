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

if __name__ == "__main__":
    print(least_confidence(torch.tensor([0.5, 0.4, 0.1])))
    print(entropy_based(torch.tensor([0.5, 0.4, 0.1])))