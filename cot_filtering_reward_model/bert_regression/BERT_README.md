This is a simplier implementation of the RL trained reward model 
The RL reward model is trained using GPRO with verifiable rewards to reason about a response, and then output a score between 0-1, which is compared to the similarity score between the predicted and true NL from bird. 
We use RL here because it allows us to train it to reason in a way that produces a correct final result.

However, if reasoning doesn't benefit the model, then we can train a much lighter model to do the same task by putting a sigmoid on a bert model and finetuning to embed the input and output a score. This approach is much less computationally expensive, so it makes sense to check both and see what is better 