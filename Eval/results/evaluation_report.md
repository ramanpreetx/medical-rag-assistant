# Medical RAG Evaluation Report

## 1. Evaluation Overview

The Medical RAG system was put to the test across four key areas: retrieval performance, reranker performance, threshold calibration, and response faithfulness. The goal was to ensure that the system not only pulls up relevant medical information but also effectively filters out unsupported queries and generates responses that are firmly based on the evidence retrieved.

For the evaluation, two benchmark datasets were utilized. The first benchmark featured **100 manually reviewed medical questions** and was aimed at assessing the retrieval pipeline and reranker. To generate candidate relevant chunks for each question, a pooling approach was employed, pulling from various retrieval configurations, and these were then automatically labeled for relevance. Every AI-generated label underwent a manual review, and any inaccuracies were corrected before calculating the retrieval metrics.

A second benchmark was specifically designed for threshold calibration and RAGAS evaluation. This dataset included **150 questions**, broken down into **50 medical questions**, **50 out-of-domain questions**, and **50 out-of-domain but medically related questions**. The threshold calibration process utilized all 150 questions to establish an appropriate reranker score threshold, which would determine whether a query should be sent to the language model. RAGAS evaluation focused solely on the 50 medical questions, as metrics like faithfulness are meant to assess responses to valid in-domain queries.

To evaluate retrieval and reranker performance, standard information retrieval metrics were employed, including Recall@k, Precision@k, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (nDCG). The threshold calibration examined the distribution of reranker scores across various query categories to identify a threshold that maximized recall for valid medical queries while filtering out unsupported requests. Lastly, RAGAS was utilized to assess response faithfulness, checking whether the generated answers were backed by the retrieved context.


## 2. Retrieval Evaluation

The retrieval component was evaluated to find the best retrieval strategy for the Medical RAG system. Different retrieval approaches were tested, including FAISS dense retrieval, BM25 sparse retrieval, hybrid retrieval with different FAISS/BM25 weight combinations, and MMR-based retrieval.

The evaluation was performed on 100 medical questions with the metrics Recall@5, Recall@10, Precision@5, Precision@10, MRR and nDCG.

The results showed that FAISS-based Similarity Search achieved the best overall performance. The highest-performing configuration was FAISS-only retrieval (FAISS weight = 1.0, BM25 weight = 0.0), achieving:

- Recall@5: 0.7567
- Recall@10: 0.9543
- Precision@5: 0.5900
- Precision@10: 0.4040
- MRR: 0.9660
- nDCG@5: 0.8444
- nDCG@10: 0.8994

The results show that dense semantic retrieval worked very well on the medical knowledge base. A Recall@10 score of 0.9543 means that most of the relevant chunks are in the top 10 results. The high MRR score means that the relevant information was generally ranked high.

Hybrid retrieval was also tested with increased contribution of BM25. However, adding BM25 did not lead to improved retrieval performance. When the BM25 weight was increased, the retrieval quality decreased. For example, pure BM25 retrieval only achieves 0.3478 Recall@5 and 0.6176 MRR in comparison to 0.7567 Recall@5 and 0.9660 MRR for pure FAISS retrieval. This indicates that semantic similarity was more effective than keyword-based matching for medical queries.

MMR retrieval was tested to determine whether increasing diversity among retrieved chunks could improve results. However, MMR performed worse than standard similarity search with maximum Recall@10 of 0.6498 compared to 0.9543 for FAISS similarity search. This indicates that the decrease of similarity between retrieved chunks removed useful related information, which is important for medical questions often requiring multiple chunks from the same topic.

Based on these experiments the final retrieval strategy was chosen as FAISS Similarity Search without BM25 contribution. It achieved the best trade-off between recall, ranking quality and retrieval accuracy and was used as the baseline retriever for the next evaluation phase.

## 3. Reranker Evaluation

After selecting FAISS Similarity Search as the retrieval strategy, a cross encoder reranker was evaluated, to see if reranking those retrieved chunks could help , with the quality of the final context that gets passed to the LLM.

Several configurations were tried by changing how many candidate chunks were first pulled (`retrieve_k`) and then how many were kept after reranking (`rerank_k`).

As a baseline, the retriever was used without any reranking (`retrieve_k=10`) and it achieved:

- Recall@5: 0.7567
- Recall@10: 0.9543
- Precision@5: 0.5900
- MRR: 0.9660
- nDCG@5: 0.8444

Applying reranking with `retrieve_k=10` and `rerank_k=5` produced:

- Recall@5: 0.7596
- Precision@5: 0.5860
- MRR: 0.9412
- nDCG@5: 0.8379

The reranker gave a small improvement in recall but at the same time a little lower ranking metrics compared to the original similarity retriever, like it helped in one place and then slipped a bit somewhere else. This suggests that the cross-encoder wasn’t really boosting retrieval accuracy much but it could, rather gently, shuffle the results by looking at deeper query document relevance.

When I increased the retrieval pool ( `retrieve_k=20` ) performance went down. The setup with retrieve_k=20 and rerank_k=5 got Recall@5 of 0.7083 and MRR of 0.9233. But if selecting just the top 3 reranked chunks, precision came up to 0.6800 while recall dropped to 0.5779. So there’s a kind of trade off here between grabbing more potentially relevant info and then offering fewer but more pointed chunks.

Based on the results, reranking was not used to replace the original retriever but was used as a context optimization step before generation. The final RAG pipeline retrieves candidate chunks using FAISS similarity search and applies the cross-encoder reranker to select the most relevant chunks before passing context to the LLM.

The `retrieve_k=10` and `rerank_k=5` configuration was selected for downstream evaluation, including threshold calibration and RAGAS evaluation, because it provided a good balance between retrieval coverage and context reduction.

Even though the reranker did not improve retrieval metrics in any big way, it stayed in the final RAG pipeline because the evaluation metrics mostly track retrieval performance, not the actual quality of the final generated answer. A slight drop in the number of retrieved chunks can still be helpful, since it gives the LLM a more focused context, sort of. So the reranker ended up being used as a context refinement step.

The main drawback though is the extra latency that comes from the cross encoder reranking process.

## 4. Threshold Calibration

A threshold calibration experiment was done to figure out the best reranker score cutoff, for filtering out those kinda irrelevant queries before they ever get pushed into the RAG pipeline. A total of 150 queries were used, consisting of:

- 50 medical queries covered by the knowledge base
- 50 out-of-domain queries unrelated to medical topics
- 50 medically related queries not covered by the knowledge base

The score distribution looked like it had a clear split between supported medical queries and weird irrelevant ones. Medical queries had noticeably higher reranker scores, with a median score of **0.9885**, while out-of-domain queries were down around **0.0004**. For medical-related questions that were outside the knowledge base, the scores sat in the middle, a median of **0.0286** which suggests the model can spot medical intent, though not every time it can also find the relevant info available.

Based on the score distribution, a threshold of **0.045** was selected. The threshold was chosen to maintain high acceptance of valid medical queries while rejecting most irrelevant queries.

The selected threshold achieved the following results:

- Medical queries (in knowledge base): 50/50 accepted (100% acceptance)
- Out-of-domain queries: 4/50 accepted and 46/50 rejected (92% rejection)
- Medical-related queries not covered by the knowledge base: 20/50 accepted and 30/50 rejected (60% rejection)

The results show that the threshold successfully retained all valid medical queries while filtering the majority of irrelevant queries. Some medical-related queries outside the knowledge base were accepted because they were semantically similar to available medical content, which is expected since the reranker evaluates relevance rather than whether the exact information exists in the knowledge base.

The threshold of **0.045** was selected for the final pipeline to reduce unnecessary LLM calls for irrelevant queries while maintaining high recall for supported medical questions.

## 5. Faithfulness Evaluation

Faithfulness was assessed with 50 medical questions, basically to see how well the generated responses were supported by the retrieved context. The evaluation was performed using both top-3 and top-5 retrieved chunks after reranking.

From the results, it looked like both setups got really high faithfulness scores. Most answers were sitting near 1, which suggests the responses were strongly anchored in the medical documents that came back.

The handful of lower-scoring cases seemed to pop up mostly because of a couple things: adding little details or interpretations that the retrieved context didn’t actually state, and also mixing information from several retrieved chunks where some of those chunks were relevant but not strictly needed for the question.

So in general, this evaluation suggests the RAG pipeline keeps good grounding. The remaining hiccups are more about context choice, and small response expansion, rather than bigger hallucination issues.

## 6. Overall Findings

The evaluation showed that the Medical RAG system was able to retrieve, relevant medical information, filter unsupported queries and generate responses that stayed strongly grounded in the retrieved context

FAISS based dense retrieval gave the best retrieval performance compared to BM25 , hybrid retrieval, and MMR style approaches. The reranker did not really boost retrieval metrics by a lot, though it was still kept as a context refinement step before generation. This made the context more focused for the LLM while it also kept high faithfulness. The threshold calibration also looked promising , because it could identify unsupported queries well, and cut down on unnecessary LLM calls

The final pipeline configuration was

- Retriever: FAISS Similarity Search
- Embedding model: BAAI/bge-base-en-v1.5
- Retrieval candidates: Top 10 chunks
- Reranker: Cross-Encoder reranking
- Final context: Top 5 chunks
- Reranker threshold: 0.045

### Limitations

The primary limitation of this evaluation is kinda the benchmark dataset. Since no real user query logs or publicly available benchmarks aligned with the MedlinePlus Health Topics knowledge base were available, the evaluation questions were manually made up by sampling documents from the corpus and turning them into representative questions. Even if the questions were manually reviewed, they might not really capture the full diversity, the ambiguity, or the actual distribution you’d see in real-world user queries.

Also the evaluation was run on a relatively small benchmark, which is ok for comparing retrieval configurations ,but it does not ensure that the measured performance will generalize to basically all possible medical questions.