# Generative language model notes (Coursera)

## Week 1

### Transformer for Generative Evaluations (or translation schema)
- As always, refer to Attention is All you Need for the theoretical basics
- From [machine learning mastery](https://machinelearningmastery.com/the-transformer-model/): 
	- In a nutshell, the task of the encoder, on the left half of the Transformer architecture, is to map an input sequence to a sequence of continuous representations, which is then fed into a decoder. 

	- The decoder, on the right half of the architecture, receives the output of the encoder together with the decoder output at the previous time step to generate an output sequence.

### Shots
- Within some context window (amount of characters):
	- Zero-Shot: Asking to complete task with no sample evaluations in context window [larger, well-trained LLM]
	- One-Shot: Asking to complete task with one sample evaluation (same format) in context window 
	- Few-Shot: Asking to complete task with a few sample evaluations (same format) in context window [smaller, well-trained LLM]
	- If not completing to satisfaction: retraining may be necessary

### Tuning Evaluation
- These hyperparameters tuned in parallel
	- Max new tokens: look at N many tokens for next spot
	- Temperature: Generate narrower distribution of softmax outputs:
	$softmax(\frac{x_i}{T}, i=1,...,N$ positions
	- k-largest: select from only k-highest prob characters for next char. output
	- p-highest: select from only p-highest (cumulative) probability characters
