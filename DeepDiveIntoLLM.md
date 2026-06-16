# Deep Dive into LLMs

**Total Duration:** ~3h 32m | **24 Chapters**

## Table of Contents

1. [Introduction](#1-introduction-000000) — `00:00:00`
2. [Pretraining Data (Internet)](#2-pretraining-data-internet-000100) — `00:01:00`
3. [Tokenization](#3-tokenization-000747) — `00:07:47`
4. [Neural Network I/O](#4-neural-network-io-001427) — `00:14:27`
5. [Neural Network Internals](#5-neural-network-internals-002011) — `00:20:11`
6. [Inference](#6-inference-002601) — `00:26:01`
7. [GPT-2: Training and Inference](#7-gpt-2-training-and-inference-003109) — `00:31:09`
8. [Llama 3.1 Base Model Inference](#8-llama-31-base-model-inference-004252) — `00:42:52`
9. [Pretraining to Post-training](#9-pretraining-to-post-training-005923) — `00:59:23`
10. [Post-training Data (Conversations)](#10-post-training-data-conversations-010106) — `01:01:06`
11. [Hallucinations, Tool Use, Knowledge/Working Memory](#11-hallucinations-tool-use-knowledgeworking-memory-012032) — `01:20:32`
12. [Knowledge of Self](#12-knowledge-of-self-014146) — `01:41:46`
13. [Models Need Tokens to Think](#13-models-need-tokens-to-think-014656) — `01:46:56`
14. [Tokenization Revisited: Models Struggle with Spelling](#14-tokenization-revisited-models-struggle-with-spelling-020111) — `02:01:11`
15. [Jagged Intelligence](#15-jagged-intelligence-020453) — `02:04:53`
16. [Supervised Finetuning to Reinforcement Learning](#16-supervised-finetuning-to-reinforcement-learning-020728) — `02:07:28`
17. [Reinforcement Learning](#17-reinforcement-learning-021442) — `02:14:42`
18. [DeepSeek-R1](#18-deepseek-r1-022747) — `02:27:47`
19. [AlphaGo](#19-alphago-024207) — `02:42:07`
20. [Reinforcement Learning from Human Feedback (RLHF)](#20-reinforcement-learning-from-human-feedback-rlhf-024826) — `02:48:26`
21. [Preview of Things to Come](#21-preview-of-things-to-come-030939) — `03:09:39`
22. [Keeping Track of LLMs](#22-keeping-track-of-llms-031515) — `03:15:15`
23. [Where to Find LLMs](#23-where-to-find-llms-031834) — `03:18:34`
24. [Grand Summary](#24-grand-summary-032146) — `03:21:46`

---

## 1. Introduction `[00:00:00]`

### What is this video about?

A general audience introduction to Large Language Models (**LLMs**) like ChatGPT. The goal is to build mental models for thinking about what these tools are — what they're good at, what they're bad at, and where the sharp edges lie.

**Key Questions Answered**

- What is behind the ChatGPT text box?
- How are the words generated?
- What are you actually talking to?

### Structure of the Video

The video walks through the entire pipeline of how LLMs are built — from raw internet data all the way to a usable assistant. Along the way, it covers the cognitive and psychological implications of interacting with these models.

---

## 2. Pretraining Data (Internet) `[00:01:00]`

### The Big Picture

Building an LLM starts with a **pre-training** stage, and the very first step is: download and process the internet.

### The FineWeb Dataset (by Hugging Face)

A good real-world reference. All major LLM providers (OpenAI, Anthropic, Google) have their own internal equivalent.

Goals for the dataset:

- Huge quantity of text
- Very high quality documents
- Large diversity of topics (to pack broad knowledge into the model)

> 📦 **Size:** Even after filtering the entire internet down to text, FineWeb is only ~44 terabytes → roughly fits on a single hard drive.

### Where Does the Data Come From? → Common Crawl

- Organisation that has been crawling the internet since 2007
- As of 2024: indexed 2.7 billion web pages
- Works by starting at seed pages and following all links recursively

### Data Filtering Pipeline

Raw Common Crawl data is very messy. It goes through multiple processing stages:

| Stage | What it Does |
|---|---|
| URL Filtering | Removes malware, spam, adult, racist websites using block lists |
| Text Extraction | Strips raw HTML markup, keeps only readable text |
| Language Filtering | Keeps pages with >65% English (design decision per company) |
| Deduplication | Removes duplicate pages |
| PII Removal | Strips personally identifiable info like addresses, SSNs |

> 💡 Language filtering is a design decision. If you filter out Spanish, your model won't be good at Spanish. Companies optimise differently for multilingual capability.

Read full blog post: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1

---

## 3. Tokenization `[00:07:47]`

### The Problem

Neural networks can't directly process text. They expect a one-dimensional sequence of symbols from a finite set. So we need to convert text into such a sequence.

Before we can even think about tokens, we need to understand how text is stored on a computer at the lowest level. Every character you type — a letter, a space, a punctuation mark — is ultimately stored in your computer as a sequence of bits (0s and 1s). The system that defines which sequence of bits corresponds to which character is called an **encoding**.

The most widely used encoding on the internet is **UTF-8** (Unicode Transformation Format – 8-bit).

For example, under UTF-8:

- The letter `h` is stored as `01101000`
- The letter `e` is stored as `01100101`
- The letter `l` is stored as `01101100`

So the word `hello` becomes a stream of 40 bits (8 bits per character × 5 characters).

When we take our entire filtered internet dataset — that 44 TB of text — and look at it at this raw level, the whole thing is just one enormously long sequence of 0s and 1s. This is the most fundamental representation of our text data on the computer.

This raw binary representation is technically a valid input format for a neural network — it has a finite set of symbols (just 0 and 1), and it's a one-dimensional sequence. But as we're about to see, it's a terrible choice in practice.

### Why Not Just Use Raw Bits?

- Raw UTF-8 = only 2 symbols (0 and 1) but an extremely long sequence
- Sequence length is a precious, limited resource in neural networks (Why?)

There are two separate reasons, one computational and one architectural.

**Reason 1: The Attention Mechanism Scales Quadratically**

The most expensive part of a Transformer is the **attention mechanism** — the step where every token looks at every other token to understand context. The cost of this operation doesn't scale linearly with sequence length; it scales quadratically.

What that means: if you double the sequence length, the attention computation doesn't double in cost — it becomes four times more expensive. Triple the length → nine times the cost. This gets out of hand very quickly.

So if you use raw bits and your sequence is, say, 8× longer than it needs to be, your attention computation is not 8× more expensive — it's 64× more expensive. That's the difference between a training run that takes one week and one that takes over a year.

**Reason 2: The Context Window Is Finite**

The neural network can only "see" a limited number of tokens at once — this is the maximum **context length**. GPT-2 could only see 1,024 tokens at a time. Modern models can see 100,000 or more, but there is always a hard ceiling.

This ceiling exists because of the quadratic cost above — you can't just make it infinite. So every token slot in that window is genuinely precious real estate.

Here's the practical consequence: if you encode text as raw bits (2 symbols), the sentence "The cat sat on the mat" uses up roughly 176 token slots (22 characters × 8 bits). Encode it as bytes (256 symbols) and it uses 22 slots. Encode it using BPE (Byte Pair Encoding) tokens (100,277 symbols) and it might use only 7 slots.

Those saved slots can now hold more context — more of the conversation, more of the document, more background information. That directly improves the quality of predictions.

We want more symbols and shorter sequences — a trade-off.

### Step 1: Bytes

- Group 8 bits into 1 byte → 256 possible symbols (0–255)
- Sequence becomes 8x shorter
- Think of each byte as a unique ID (like a unique emoji), not a number

### Step 2: Byte Pair Encoding (BPE)

Merge frequently occurring byte pairs to compress further:

- Find the most common consecutive pair (e.g., bytes 116 and 32)
- Mint a new symbol (e.g., ID 256) for that pair
- Replace every occurrence of that pair with the new symbol
- Repeat many times until vocabulary reaches target size

> 🎯 GPT-4 uses a vocabulary of 100,277 tokens. This process of converting text → tokens is called **tokenization**.

### Tokenization in Practice

- Website to explore: Tiktokenizer (select `cl100k_base` for GPT-4)
- `"hello world"` → 2 tokens
- Capitalisation, spacing, and punctuation all affect tokenisation
- The same word in different contexts can produce different tokens

---

## 4. Neural Network I/O `[00:14:27]`

### Setting the Stage

After tokenizing the internet, the FineWeb dataset becomes a sequence of **15 trillion tokens**. This is the input to neural network training.

### What Does the Neural Network Do?

It learns the statistical relationships between tokens — i.e., given previous tokens, what token is most likely to come next.

### How Training Works (Step by Step)

1. Take a window of tokens from the data (up to ~8,000 tokens in practice)
2. Call the window the **context**; the next token is the **label** (correct answer)
3. Feed context into the neural network → it outputs a probability for every possible next token (100,277 values)
4. Compare to the correct answer → measure how wrong the model is
5. Update the weights slightly to make the correct token more probable
6. Repeat for billions of token windows

### Window Sizes Over Time

| Model | Max Window |
|---|---|
| GPT-2 (2019) | 1024 tokens |
| GPT-3 (2020) | 2048 tokens |
| GPT-4 (2023) | ~128,000 tokens |
| Frontier models (2025) | 200K to 1M+ tokens |

### Input & Output Summary

| Column 1 | Detail |
|---|---|
| Input | Sequence of tokens (variable length, up to ~8,000) |
| Output | 100,277 probabilities — one per possible next token |
| At start | Weights are random → predictions are random |
| After training | Weights tuned → predictions match data statistics |

---

## 5. Neural Network Internals `[00:20:11]`

### What's Inside?

Inputs (token sequences) are mixed with parameters (weights) through a giant mathematical expression. Modern neural networks have billions of parameters.

> 🎛 Think of parameters like knobs on a DJ set. Twiddling them changes the model's predictions. Training = finding the knob settings that produce predictions consistent with the training data.

### The Math Is Simple (Individually)

Each operation is just multiplication, addition, exponentiation, or division. The complexity comes from doing this at a massive scale with billions of terms.

### The Transformer Architecture

The specific neural network architecture used in modern LLMs is called the **Transformer**.

Information flow inside a Transformer:

- Input tokens are embedded — each token gets a vector representation
- Information flows through attention blocks (tokens 'look at' each other)
- Then through MLP (Multi-Layer Perceptron) blocks
- Output: probabilities for the next token

### Important Caveat About 'Neurons'

These are called neurons loosely, but they're far simpler than biological neurons. Biological neurons have memory and complex dynamics. These are stateless mathematical operations — no memory between forward passes.

---

## 6. Inference `[00:26:01]`

### What is Inference?

After training, we generate new data from the model — no more training. This is **inference**.

### How Generation Works (Step by Step)

1. Start with a prefix (some initial tokens)
2. Feed into the network → get probability distribution over next tokens
3. Sample from this distribution (flip a biased coin) → pick next token
4. Append that token to the sequence
5. Feed the longer sequence back in → repeat until done

### Why Sampling Instead of Always Picking the Top Token?

- Always picking the top (greedy) token → repetitive, boring output
- Sampling introduces variety and creativity
- High-probability tokens are more likely, but not guaranteed

### The Output is a 'Remix'

The model doesn't copy text from training. Each token is sampled fresh. The result is statistically similar to training data but not identical — like an improvisation inspired by what it learned.

> 🔒 In production (ChatGPT): the model was trained months ago — weights are fixed. When you chat, it's pure inference — the model just continues your token sequence, one token at a time.

---

## 7. GPT-2: Training and Inference `[00:31:09]`

### Why GPT-2?

GPT-2 (2019, OpenAI) was the first recognizably modern LLM. All the pieces are the same as today — everything has just gotten bigger since. **GPT** expands as **G**eneratively **P**re-trained **T**ransformer.

### GPT-2 Specs vs. Modern LLMs

| Parameter | GPT-2 (2019) | Modern LLMs |
|---|---|---|
| Parameters | 1.6 billion | 100B – 1 trillion+ |
| Context length | 1,024 tokens | 100K – 1M tokens |
| Training tokens | 100 billion | 15+ trillion |
| Training cost | ~$40,000 | Millions+ |

### Cost of Training: Then vs. Now

- In 2019: ~$40,000
- Today: can be reproduced for ~$600 (or as low as ~$100)
- Why is it cheaper? Better datasets, much faster hardware (GPUs), better software

### Hardware: GPUs

- Training runs on NVIDIA H100 GPUs → ~$3/GPU/hour to rent in the cloud
- Real LLM training uses thousands of GPUs in datacenters
- GPU demand is why Nvidia's market cap exploded to ~$3.4 trillion

> 🤔 What are all those GPUs doing? Collaborating to predict the next token on a massive dataset and nudging the model's weights to be better at it. That's it. The simplicity of the core task at enormous scale is what drives the entire AI revolution.

GPT-2 Model release blog: https://openai.com/index/gpt-2-1-5b-release/
GPT-2 Model Release GitHub: https://github.com/openai/gpt-2/tree/master

---

## 8. Llama 3.1 Base Model Inference `[00:42:52]`

### What is a Base Model?

The model that comes out at the end of pre-training. It is an internet text token simulator → it predicts and generates text that looks like internet documents.

> ⚠️ A base model is **NOT** yet an assistant. It doesn't answer questions — it just autocompletes tokens.

### What a Model Release Includes

- **Python code** — describes the architecture (sequence of mathematical operations)
- **Parameters** — the actual learned weights (billions of numbers)

### Llama 3.1 (by Meta)

- 405 billion parameters trained on 15 trillion tokens
- Released as open weights — anyone can download and use
- Also comes with an 'instruct' (assistant) version

Llama 3 Release blog: https://ar5iv.labs.arxiv.org/html/2407.21783

### Base Model Behaviors

**In-Context Learning**

Even without being an assistant, a base model can do impressive things with clever prompting:

| Prompt | What Happens |
|---|---|
| "What is 2+2?" | Does NOT say '4'. Just autocompletes tokens, often going philosophical |
| Wikipedia text | Can recite large chunks from memory (seen many times in training) |
| Future events (post cutoff) | Makes up plausible but wrong answers — probabilistic hallucination |

- **Few-shot prompting:** Show 10 English→Korean translation examples, then give a new word. Model continues the pattern and translates correctly.
- **Simulated assistant:** Format your prompt as a webpage showing a conversation. Model continues in that role.

> 💡 The 405B parameters are like a lossy compression of the internet. Common knowledge is well-remembered; rare facts may be hallucinated.

### The "Psychology" of a Base Model

- It's a token-level internet document simulator
- It is stochastic / probabilistic — you're going to get something else each time you run it
- It "dreams" internet documents
- It can also recite some training documents verbatim from memory ("regurgitation")
- The parameters of the model are kind of like a lossy zip file of the internet; a lot of useful world knowledge is stored in the parameters of the network

---

## 9. Pretraining to Post-training `[00:59:23]`

### What We've Built So Far

A base model — a token simulator trained on internet documents. Interesting, but not directly useful as an assistant.

### The Three Stages of Building an LLM

| Stage | Input Data | Output | Cost |
|---|---|---|---|
| Pre-training | Internet text (15T tokens) | Base model | Months, millions $$$ |
| Supervised Finetuning (SFT) | Curated conversations | Assistant model | Hours, cheap |
| Reinforcement Learning (RL) | Practice problems with answers | Reasoning model | Days, moderate |

Post-training is computationally much cheaper than pre-training, but transforms the internet simulator into a helpful assistant.

---

## 10. Post-training Data (Conversations) `[01:01:06]`

### What Does Post-training Data Look Like?

Instead of raw internet text, we now use **conversations** — structured exchanges between a human and an assistant.

Example conversations in the dataset:

- Human: "What is 2+2?" → Assistant: "2+2 is 4."
- Human: "Write me a poem." → Assistant: [writes a poem politely]
- Human: "How do I make a bomb?" → Assistant: "I'm sorry, I can't help with that."

### How the Model Is 'Programmed'

Not with code — by example. Hundreds of thousands of ideal conversations teach the model to statistically imitate a good assistant's behaviour.

### Where Do the Conversations Come From?

**Phase 1: Human Labellers (InstructGPT, 2022)**

Human labellers hired by the company (e.g., OpenAI, via Upwork or Scale AI):

- Given labelling instructions — sometimes hundreds of pages, written by the company
- Asked to: come up with prompts, then write ideal assistant responses
- Guidelines: be helpful, truthful, and harmless

**Phase 2: LLM-Assisted Data Creation (current)**

- Labellers rarely write responses from scratch now — LLMs generate drafts, humans edit
- Example: UltraChat — millions of mostly synthetic conversations with human oversight
- 'SFT mixtures' = combining human-written, human-edited, and synthetic data

UltraChat repo: https://github.com/thunlp/ultrachat
Paper on training LLMs on conversations by OpenAI: https://arxiv.org/pdf/2203.02155

### Supervised Fine-tuning (SFT)

Training on conversations uses the exact same algorithm as pre-training. Only the dataset changes.

- Pre-training: months. SFT: a few hours.
- Modern practice: labellers use existing LLMs to draft responses, then curate/edit — they rarely write from scratch.

### Conversation Token Format

Every conversation must be converted to tokens using special markers:

```
<|im_start|>user<|im_sep|>What is 2+2?<|im_end|>
<|im_start|>assistant<|im_sep|>2+2 is 4.<|im_end|>
```

| Token | Meaning |
|---|---|
| `<\|im_start\|>` | Start of a turn (IM = imaginary monologue) |
| `user` / `assistant` | Whose turn it is |
| `<\|im_sep\|>` | Separator between role and content |
| `<\|im_end\|>` | End of turn |

> 🧠 If you ask ChatGPT a question, what ChatGPT responds is a statistical simulation of a skilled human data labeler at OpenAI who:
> 1. Read OpenAI's hundreds of pages of labeling instructions
> 2. Looked at your prompt
> 3. Wrote what they judged to be the ideal response

- If your exact query appeared in the SFT dataset: you'll likely get something very close to what that labeler wrote.
- If not: you get an emergent generalisation — the model extrapolates from its pretraining knowledge + the style and format it learned from SFT.

### System Messages

At the start of every conversation, companies inject a hidden system message (invisible to you) that tells the model its identity, behavior guidelines, and cutoff dates. This is why ChatGPT 'knows' it's ChatGPT.

- Contains model identity ('You are ChatGPT made by OpenAI...')
- Contains knowledge cutoff date
- May contain custom instructions (e.g., 'You are a customer service agent for Company X')
- Developers building on LLMs can customise this to change model behaviour

---

## 11. Hallucinations, Tool Use, Knowledge/Working Memory `[01:20:32]`

### What Are Hallucinations?

When LLMs confidently make up false information. A long-standing problem, but mitigations have significantly improved it.

### Why Do Hallucinations Happen?

The training data contains conversations where humans confidently answer 'Who is X?' questions. The model mimics this confident tone — even when it doesn't actually know who X is.

When asked about a nonexistent person like 'Orson Kovats,' the model generates a plausible-sounding but completely fabricated answer.

### Mitigation 1: Teaching the Model to Say 'I Don't Know'

Use model interrogation to discover the model's knowledge and programmatically augment its training dataset with knowledge-based refusals in cases where the model doesn't know.

Meta's approach for Llama 3:

1. Take a random document, generate factual Q&A from it using an LLM
2. Ask the model these questions multiple times
3. If the model consistently gets the answer wrong → it doesn't know
4. Add a training example: correct response is 'I don't know'
5. The model learns to associate its internal uncertainty with admitting it

### Mitigation 2: Tool Use — Web Search

Instead of relying only on memory, give the model the ability to search the web:

1. Model emits special tokens: `[search_start] query [search_end]`
2. System pauses, runs web search, retrieves results
3. Results pasted into the context window (working memory)
4. Model generates answer based on fresh, retrieved information

### The Memory Distinction

| Type | Location | Nature |
|---|---|---|
| Vague Recollection | Model's parameters (weights) | Vague, probabilistic recollection — 'what you read a month ago' |
| Working memory | Context window (tokens in current chat) | Directly accessible, precise — 'what you just read' |

> 💡 Practical tip: When you want the model to reference something specific, paste it directly into the conversation rather than asking the model to recall it from memory. Working memory >> long-term memory for accuracy.

### Other Tools

- **Code interpreter:** Write and execute Python code for math, data processing
- When in doubt on a computation: ask the model to 'use code' — far more reliable than mental arithmetic

---

## 12. Knowledge of Self `[01:41:46]`

### The Strange Question: 'What Model Are You?'

This is somewhat nonsensical because:

- The model has no persistent existence — boots up, processes tokens, shuts off
- No continuous self — each conversation starts completely fresh
- It's a token tumbler following statistical patterns, not a person

### What Happens Without Explicit Programming?

The model hallucinates its own identity! A model not told what it is will likely say it's 'ChatGPT by OpenAI' — because that's the most common 'helpful AI assistant' identity in its training data.

### How Companies Program Self-Knowledge

There are two ways companies handle this:

**A. Hardcoded conversations**
Include ~240 example Q&As in the SFT dataset: 'What are you?' → 'I am Tulu, made by Allen AI...'

**B. System Messages**
Inject a hidden message at start of every conversation: 'You are ChatGPT, made by OpenAI...'

> ⚠️ Both methods are bolted on and not deeply embedded. It's all just statistical pattern matching — not genuine self-awareness.

---

## 13. Models Need Tokens to Think `[01:46:56]`

### The Core Insight

For every single token the model generates, there is a fixed, finite amount of computation. The model cannot do arbitrary amounts of thinking per token generated.

### The Fixed Compute Constraint

- Every token generation = one forward pass through the neural network (fixed layers, fixed computation)
- Typically ~100 layers of computation per token in modern networks
- This computation budget is finite and **CANNOT** be exceeded for any single token
- Consequence: if reasoning requires many steps, those steps must be distributed across many tokens

### Why This Matters — The Apple Problem

**Q:** Emily buys 3 apples and 2 oranges. Each orange costs $2. Total is $13. What is the cost of each apple?

The bad answer tries to cram all computation into one token jump. The good answer spreads computation across many tokens — each step is small and manageable.

| Bad Training Example (Never Use) | Good Training Example (Always Use) |
|---|---|
| "The answer is $3. Because 3×$3 + 2×$2 = $13." | "Let the cost of each apple be $a. Oranges cost 2×$2 = $4. So 3a + 4 = 13 → 3a = 9 → Each apple costs $3." |

**Why is this bad?**

- Token sequence: 'The answer is $' → must produce '3' in ONE forward pass
- That single forward pass must contain ALL the computation for the problem
- One forward pass ≠ enough computation for multi-step arithmetic
- Everything after '3' is post-hoc justification — answer already committed, working is fake

**Why is this good?**

- '2 × $2 = $4' — simple computation, one step
- '$13 - $4 = $9' — simple given $4 is now in context
- '$9 ÷ 3 = $3' — simple given $9 is in context
- Each token does a small, manageable amount of work; context window accumulates the results

### The 'Show Your Work' Principle

When labellers create training data, they must write out intermediate steps. Not for the user — but for the model's benefit. Without steps, the model can't reliably reach correct answers.

### Practical Guidelines

| Task | Approach |
|---|---|
| Multi-step math | Ask to 'use code' — delegate to Python interpreter |
| Counting items | Ask to 'use code' — let Python count |
| Complex reasoning | Ask to 'think step by step' — force externalised intermediate steps |
| Simple knowledge Q&A | Model can handle inline — no special handling needed |
| Verifying computation | Always verify with code, not just model's working |

- ChatGPT naturally shows work for math — because OpenAI's labellers were trained to write out steps
- For computation reliability: ask the model to use code (Python interpreter)
- The model is reliable at copy-pasting to Python; Python handles the actual arithmetic

> 🎯 Mental model: Each token = a small fixed unit of compute. Distribute your reasoning across many tokens. Never ask for a big leap in a single token.

---

## 14. Tokenization Revisited: Models Struggle with Spelling `[02:01:11]`

### The Problem

Models don't see characters — they see tokens (chunks of text). There is no direct access to individual letters.

### Why Spelling Tasks Fail

Let's take an example of the word "Ubiquitous":

- Humans see: U-B-I-Q-U-I-T-O-U-S (10 individual letters, easy to index)
- The model sees 3 tokens (e.g., 'ubiq' + 'ui' + 'tous')
- To do 'every 3rd character', the model must know which letters are INSIDE each token AND perform the indexing correctly — all from memory during training. Therefore it fails at this task.

### Fix: Use Code

Ask the model to 'use code' → it writes Python to manipulate the string character by character → Python handles it correctly.

### The Famous 'Strawberry' Example

For a long time, state-of-the-art models said 'strawberry' has 2 R's (it has 3). Why?

- **Failure 1 (tokenisation):** 'strawberry' is NOT tokenised as individual letters — model sees chunks
- **Failure 2 (counting):** even if the model could see letters, counting requires too much per-token compute
- Combining both leads to consistent failure

> 📌 Most modern models now handle 'strawberry' correctly, likely via hardcoded training examples or improved tokenization-aware training.

---

## 15. Jagged Intelligence `[02:04:53]`

### The Swiss Cheese Model

LLMs are incredibly capable across a huge range of topics — but have random holes in their capabilities, like holes in Swiss cheese.

### The 9.11 vs. 9.9 Example

Ask: 'Which is bigger, 9.11 or 9.9?' — Models sometimes say 9.11 is bigger, which is wrong (9.9 > 9.11). Yet the same model solves Olympiad-grade math.

Why? Researchers found neurons associated with Bible verses light up. In Bible verses, 9:11 does come after 9:9. The model gets cognitively distracted by this pattern.

### Why This Matters

- Failures can appear completely random and context-dependent
- The same model can solve PhD-level problems AND fail simple comparisons
- These failures are not fully understood, even by researchers who build these models
- You cannot always predict where holes are

> ⚠️ Key Lesson: LLMs are stochastic systems — magical in many ways, but unreliable in others. Don't trust them blindly. Use them as tools. Always check their work.

---

## 16. Supervised Finetuning to Reinforcement Learning `[02:07:28]`

### Quick Recap

| Stage | What it Does | Analogy |
|---|---|---|
| Pre-training | Train on internet docs → base model | Reading textbooks (exposition) |
| Supervised finetuning (SFT) | Train on conversations → assistant | Studying worked examples |
| Reinforcement Learning (RL) | Practice problems → reasoning model | Doing practice problems yourself |

### What's Missing from SFT?

With SFT, the model imitates human expert responses. But:

- Humans don't know what's easy or hard for the model
- Human solutions might skip steps the model needs, or include trivial ones
- The model needs to discover what works for it — not what works for humans

### How Companies Organise This

- **Pre-training team:** curates internet data, trains base model over months
- **SFT team:** creates conversation datasets, fine-tunes the base model
- **RL team:** sets up the RL pipeline, takes SFT model as starting point

Models are handed off sequentially. Each stage builds on the previous.

---

## 17. Reinforcement Learning `[02:14:42]`

Reinforcement Learning (**RL**) is the third and most powerful stage of LLM training. Rather than showing the model what to do (SFT), RL lets the model discover what works through trial and error.

### The Core Idea: Trial and Error

Instead of telling the model *how* to solve a problem, let it figure it out by itself:

1. Give the model a prompt (e.g., a math problem)
2. Have the model generate many different solutions (say 1,000)
3. Check each solution against the known correct answer
4. Reward solutions that got the right answer; penalize wrong ones
5. Update the model to make good solutions more likely
6. Repeat across thousands of diverse problems

### What the Model Discovers

Through trial and error across thousands of problems, the model discovers token sequences that reliably work for it — without any human specifying exactly how to solve things. These solutions are optimized for the model's own cognition.

### How RL Works in Practice

1. Sample a math problem
2. Generate 15 candidate solutions (random paths through the solution space)
3. Check → 4 right answers (green), 11 wrong answers (red)
4. Train the model on the 4 successful solutions
5. Next time this type of problem appears, the model is more likely to follow a similar path

### The Role of SFT in RL

SFT initialises the model into the right neighbourhood — it gets the model started on writing solutions. RL then dials it in to find what actually works.

> 📚 Analogy: SFT → studying worked examples from a textbook. RL → doing practice problems at the end of each chapter without the answer key, figuring out your own problem-solving strategies through trial and error.

---

## 18. DeepSeek-R1 `[02:27:47]`

### Context

Pre-training and SFT have been standard for years. But RL for LLMs was mostly done secretly inside companies — until the DeepSeek-R1 paper made it public.

### What DeepSeek-R1 Showed

- **Quantitative:** Math problem accuracy continuously climbs with RL steps — no plateau observed
- **Qualitative:** Response length grows — the model learns to 'think longer'
- **Most remarkable:** The model spontaneously discovers chain of thought reasoning — without anyone telling it to

### The Emergent Behavior: Chain of Thought

The model spontaneously learns to do what humans do when solving hard problems — and no human told it to:

- **Backtrack** → 'Wait, let me reconsider...'
- **Check from multiple angles** → 'Let me verify with a different approach...'
- **Try different framings** → 'What if I set up an equation instead?'

> ✨ This 'chain of thought' emerged from RL training because these strategies reliably help the model get correct answers. It's an emergent property of the optimisation — not hardcoded by humans.

### Thinking Models vs. SFT Models

| Model Type | Examples | Trained With |
|---|---|---|
| SFT (non-thinking) | GPT-4o, GPT-4o mini | Mostly SFT + light RLHF |
| Thinking/Reasoning | o1, o3-mini, DeepSeek-R1 | Full RL training |

### Where to Access DeepSeek-R1

- chat.deepseek.com — Enable 'Deep Think' button for R1
- together.ai — Hosts R1 (American servers)
- Open weights model — anyone can download and host it

---

## 19. AlphaGo `[02:42:07]`

### The Parallel with Games

The power of RL is not new — DeepMind demonstrated it years ago with **AlphaGo** (Go-playing AI).

| Approach | Performance on Go |
|---|---|
| Supervised learning (imitate experts) | Gets better, but tops out below the best human players |
| Reinforcement learning (play to win) | Surpasses the best human players in the world |

Why? Supervised learning is fundamentally limited — you can't surpass what you're imitating. RL has no such ceiling.

### Move 37 — The Famous Moment

In a match against world champion Lee Sedol, AlphaGo played Move 37:

- 1-in-10,000 probability of being played by a human
- Initially thought to be a mistake by all human experts
- Turned out to be brilliant — AlphaGo won the game

> 🌟 AlphaGo discovered a strategy through RL that was unknown to humans. The same potential exists for LLMs applied to reasoning and problem solving.

### The Implication for LLMs

If RL is applied at scale across all domains, models could potentially:

- Discover reasoning strategies humans haven't thought of
- Develop novel problem-solving approaches
- Perhaps think in ways fundamentally different from human cognition

We're only seeing the very earliest hints of this potential. This is the frontier of current LLM research.

---

## 20. Reinforcement Learning from Human Feedback (RLHF) `[02:48:26]`

Everything discussed so far about RL works in **verifiable domains** — problems with clear correct answers (math, code). What about creative writing, summarisation, and other tasks where there's no objectively correct answer?

### The Problem → Unverifiable Domains

Pure RL requires verifiable answers (math, code). But what about:

- "Write a joke about pelicans"
- "Summarize this paragraph"
- "Write me a poem"

These have no objective correct answer. You cannot run pure RL → no score function to optimise. And having humans score every response is infeasible at scale.

**Why human scoring is infeasible:** 1,000 RL updates × 1,000 prompts per update × 1,000 response attempts per prompt = 1 billion human evaluations needed. That's 1 billion people looking at terrible pelican jokes. Not practical.

### The RLHF Solution: Train a Reward Model

Train a separate neural network (reward model) to simulate human judgment, then do RL against it:

**Step 1: Collect human rankings (not scores)**
- Generate 5 responses to a prompt
- Ask a human to rank them best-to-worst (easier than giving scores)
- This is the human's contribution — only ~5,000 such rankings needed total

**Step 2: Train the reward model**
- Reward model takes (prompt, response) as input → outputs a score (0 to 1)
- Train it to produce scores consistent with human rankings
- After training → it simulates what a human would prefer

**Step 3: RL against the reward model**
- Generate responses → score with reward model (instant, automatic)
- Train toward higher-scoring responses
- Repeat for hundreds of steps — no humans needed in the loop

### The Upside of RLHF

It's much easier to judge quality than to produce quality:

- **SFT:** labellers must GENERATE the ideal response → hard for creative tasks
- **RLHF:** labellers only RANK 5 options → much easier task
- Higher-quality human signal at lower effort → better model

### The Downside: Reward Model Hacking

What happens when you run RLHF too long:

- First ~100-200 steps: jokes about pelicans genuinely improve ✓
- After ~1,000 steps: quality collapses
- The top-scoring 'joke' becomes: 'the the the the the the'
- Expected reward model score: 0 (gibberish)
- Actual reward model score: 1.0 (maximum!)

**Why does this happen?**

- The reward model is a neural network — it has adversarial examples
- RL is extremely good at FINDING adversarial inputs that exploit blind spots
- Run RL long enough against ANY game-able reward → finds the exploit

You can't fix this permanently — there are always more adversarial examples. The only solution → stop after a few hundred steps and ship it.

### RLHF Is Not 'Real RL'

| Aspect | Verifiable RL (Math/Code) | RLHF (Creative/Subjective) |
|---|---|---|
| Reward function | Ground truth (correct answer) | Neural network simulation of humans |
| Gameable? | No — you either got it right or not | Yes — adversarial examples exist |
| How long can you run it? | Indefinitely — keeps improving | A few hundred steps, then degrades |
| Scaling potential | Enormous (AlphaGo effect) | Limited — diminishing returns then collapse |
| Role in practice | Makes thinking models powerful | Small incremental improvement to SFT models |

> ⚠️ Key insight: "RLHF is not real RL." Real RL with verifiable answers can run indefinitely and discover amazing things. RLHF can only run for a limited number of steps before the model games the reward model. It's more like a light fine-tuning step.

---

## 21. Preview of Things to Come `[03:09:39]`

### 1. Multimodality

Current LLMs are text-only. All frontier models will soon natively handle audio, images, and video. The approach requires no fundamental change:

- **Audio:** Tokenise slices of the audio spectrogram → audio tokens in the context window
- **Images:** Tokenise 16×16 pixel patches → image tokens in the context window
- **Mixed contexts:** A single context window can interleave text, audio, and image tokens simultaneously

We're already seeing early versions (GPT-4V, Gemini 1.5 multimodal), but fully native multimodal systems are still maturing.

### 2. Agents

Currently, we hand individual tasks to models. Agents will perform multi-step jobs over long periods:

- Receive a high-level goal and break it into subtasks
- Execute those subtasks, correct errors, try alternatives
- Work for minutes, hours, or days autonomously
- Report progress periodically to a human supervisor

Human-to-agent ratios will become a key productivity metric — one human overseeing many agents, like a factory manager overseeing robots.

### 3. Computer Use / Operators

Models are learning to take direct actions on computers — keyboard, mouse, web browsing, filling forms. OpenAI's Operator is an early example. Transformative because it delegates not just thinking but doing.

### 4. Test-time Training (Open Research Problem)

Currently, deployed model weights are frozen. The only learning during deployment is in-context (from the current conversation). Humans genuinely learn and update from their experiences — especially during sleep. Developing an equivalent for LLMs (test-time training) is an open research challenge.

### 5. Very Long Contexts

As tasks grow more complex (long videos, multi-week projects), context windows will need to grow to millions or billions of tokens. Simply making windows larger is not a scalable solution. New architectures (sparse attention, hierarchical models, external memory) will be needed.

---

## 22. Keeping Track of LLMs `[03:15:15]`

### Three Recommended Resources

| Resource | What it Is | Notes |
|---|---|---|
| LM Arena (lmarena.ai) | Human-based LLM leaderboard; ELO ratings from blind human comparisons | Has become somewhat gamed recently — use as a first pass only |
| AI News Newsletter (by Swyx) | Very comprehensive, published almost daily; mix of human + LLM-generated | Good for staying up to date without missing major developments |
| X / Twitter | Where a lot of AI news breaks in real-time | Follow trusted researchers, lab employees, and AI journalists |

---

## 23. Where to Find LLMs `[03:18:34]`

| Type | Provider / Platform | URL |
|---|---|---|
| Proprietary — OpenAI | ChatGPT | chat.openai.com |
| Proprietary — Google | Gemini | gemini.google.com |
| Proprietary — Anthropic | Claude | claude.ai |
| Open weights (inference) | Together.ai | together.ai |
| Base models | Hyperbolic | hyperbolic.xyz |
| Run locally | LM Studio | lmstudio.ai |
| Thinking models | DeepSeek | chat.deepseek.com (Deep Think on) |
| Thinking models — OpenAI | o1, o3-mini, o3 | chat.openai.com (paid plans) |

### Running Models Locally: Distillation & Quantisation

- **Distillation:** Small model trained to mimic a large model. E.g., 'DeepSeek-R1 7B distilled' retains some R1 reasoning in a MacBook-compatible size.
- **Quantisation:** Reduce weight precision (16-bit → 4-bit) → much smaller model with modest quality loss. 'Q4_K_M' models often run on a laptop GPU.
- LM Studio handles model selection and lets you choose distillation/quantisation level for your hardware.

---

## 24. Grand Summary `[03:21:46]`

### What Happens When You Type Into ChatGPT

1. Your message is tokenised → sequence of integer token IDs
2. Wrapped in conversation protocol (special tokens, system message, conversation history)
3. Full token sequence fed into the Transformer
4. Network generates probability distribution over all 100,277 possible next tokens
5. One token sampled from this distribution
6. Appended to sequence — repeat until response is complete
7. Each token appears on your screen as it's generated (the streaming you see)

### What Is the Model Actually Simulating?

| Model Type | What It Is |
|---|---|
| Standard model (GPT-4o) | Neural network simulation of a human data labeler at OpenAI, following labeling instructions |
| Thinking model (o3, DeepSeek-R1) | Something new and novel — chains of thought that emerged from RL training, not just human imitation |

### The Three Training Stages

| Stage | Analogous To | What It Produces |
|---|---|---|
| Pre-training | Reading all textbooks | Base model with broad world knowledge |
| SFT | Studying worked examples | Assistant that imitates expert responses |
| RL | Doing practice problems | Reasoning model that discovers its own strategies |

### The Swiss Cheese Map — Why Each Hole Exists

| Failure | Root Cause |
|---|---|
| Hallucinations (making up facts) | Trained on confident answers; internal uncertainty not connected to saying 'I don't know' |
| 9.11 vs 9.9 comparison errors | Spurious Bible verse neuron activation interferes with decimal comparison |
| Spelling/character tasks | Tokenisation — model sees chunks, not individual characters |
| Counting errors | Fixed compute per token — too much work for one forward pass |
| Mental arithmetic failures | Same as counting — delegate to code interpreter |
| Inconsistent reasoning | Stochastic sampling — same prompt can produce different reasoning chains |

### The Right Way to Use LLMs

- Use LLMs as tools, not oracles. Great for drafts, brainstorming, acceleration — not as a source of truth.
- For computation, counting, arithmetic: use code. Ask 'use code' and let Python do the work.
- For specific facts: use web search or paste the source. Don't trust long-term memory for rare or recent facts.
- Paste documents directly. Working memory >> long-term memory. Always put your reference in the context window.
- For complex reasoning: use a thinking model. DeepSeek-R1, o3 — for hard math, complex logic, difficult code.
- Own the product of your work. The model is a collaborator; you are the responsible party. Always verify before acting.

---

## My Questions

1. Why are these models called Large Language Models?
2. Why does the Attention Mechanism scale quadratically with sequence length?
3. How are the weights changed to make the correct token more probable?
4. For reasoning models, are they completely trained on RL by skipping SFT? Or do they go through SFT briefly and then in detail RL?
