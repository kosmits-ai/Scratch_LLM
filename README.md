#  Building an LLM from Scratch (Educational Project)

This project is an **illustration of the inner workings of Large Language Models (LLMs)**, implemented entirely in **Python and PyTorch**.  
It was inspired by the **[@Vizuara YouTube playlist — “Build LLMs from Scratch”](https://www.youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)**, which provides a detailed, step-by-step breakdown of how modern transformer-based models like GPT are built.

---

##  Motivation

Artificial Intelligence — and especially **Large Language Models** — are evolving at an incredible pace.  
The field of **cybersecurity** is rapidly adapting these advancements to build more intelligent, automated, and resilient defense systems.  

To truly harness the power of LLMs, I believe it’s essential to **understand how they work from the inside out**, not just how to use them.  
That’s why I decided to dive deep into their architecture and build my own minimal GPT-like model from scratch.

---

##  Learning Journey

I was fortunate to find **@Vizuara’s "Build LLMs from Scratch" playlist**, which I consider a *gold-standard resource* for learning how LLMs operate under the hood.

Through the lessons, I explored each core stage of the LLM pipeline:

1. **Tokenization** – Converting raw text into tokens (words, subwords, or characters) that the model can process.  
2. **Embeddings** – Combining word embeddings with positional embeddings to form input representations.  
3. **Transformer Architecture** – The heart of modern LLMs, made of stacked layers including:
   - Layer Normalization for stable gradient flow  
   - Multi-Head Self-Attention for contextual understanding  
   - Residual Connections for better training stability  
   - Feed-Forward Networks with activation functions (GELU/ReLU)  
   - Dropout for regularization and generalization  
4. **Decoder Output** – Producing a matrix of probabilities for every token in the vocabulary, predicting the most likely next word.
