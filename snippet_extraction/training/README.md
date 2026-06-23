Inspiration:
https://research.perplexity.ai/articles/query-aware-context-compression-for-better-snippets
https://exa.ai/blog/highlights-for-agents
https://arxiv.org/html/2501.16214v1

### High-level: "Context Surgeon: A Small Query-Aware Compressor That Cuts RAG Tokens 10×"

1. **Title / one-liner**: "Context Surgeon: An Open-Source Query-Aware Context Compressor for RAG — Reproducing Perplexity's Compression"
6. **Description**: Build an open-source clone of Perplexity's query-aware extractive compressor. Architecture: take a small bidirectional encoder (start from `pplx-embed-v1-0.6B` or `bge-base-en` if pplx-embed weights are too new) and bolt on a per-sentence keep/drop head. 
Generate ~50k–100k (query, document, kept_sentences) triples by prompting an LLM as the judge (Perplexity's paper used 750k LLM-judge-labeled pairs; he'll be 1–2 orders of magnitude smaller but with the same recipe). 

Train with BCE + distillation from a larger teacher. 

Evaluate on **SimpleQA** and a public RAG benchmark like **BEIR-NaturalQuestions** or **HotpotQA**: target a token compression ratio of ≥10× with no statistically significant drop in downstream QA accuracy (vs. an uncompressed baseline served by a small open model).

Step by step of what the algorithm will look like (AFAIK):

### Training

Making per token predictions but aggregating those tokens into sentences.

Even though Perplexity doesn’t mention it how implement this, Provenance does.

1. Independent of the BERT model, we need to split the sentences of the document. They will be needed for step 8. Sentence boundary detection is an NLP problem in of itself. We can leverage a package like NLTK for it.
2. Input the query + document as tokens to the model.
3. Get one output vector per token (standard BERT)
4. Use the linear layer (finetuned head) to turn those output vectors into a single scalar. Therefore, the linear layer has a shape of (model_size, 1).
5. Pass those scalars through a sigmod to get values between 0 and 1.
6. Next, use a threshold value T to decide which tokens should be kept. This threshold is adjustable and it depends on how much tokens you want to keep. E.g. If you want to be more aggressive with the compression, decrease the threshold value.
7. At this point we have a mask for which tokens to keep. Instead of just sending the raw tokens based on the mask, the algorithm groups the tokens into sentences.
8. At this point the sentences are already splitted. For each sentence we want the percentage of tokens that are included by the mask. We’d keep the sentences that are above a certain thresold. Note that this sentence threshold is different from the T token thresold. For example, if 50% of the tokens in the sentence are kept, then the sentence is kept.
9. Send the snippet — Concatenate the surviving sentences following the original order, that’s the compressed snippet :)


### Generating data
To get started and to avoid the cost and pain of scraping thousands of websites, I could leverage province recipes where they use the Ms Marco document ranking dataset. They just used the documents and then generated the relevant sentences using a language model. 

Further down the line, I could scrape my own websites. Maybe even focusing on the things that I mostly use agents for, such as code documentation for code, research papers, that sort of thing. For now, I guess I can start simple and avoid, or rather skip, the scraping part and just focus on generating the data. 


