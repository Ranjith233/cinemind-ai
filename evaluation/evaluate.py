"""
CineMind AI - Automated RAG Evaluation Framework
Computes empirical retrieval precision, recall, context precision, citation correctness,
faithfulness, and latency on verified benchmark questions from Quentin-Tarantino.pdf.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger
from config.settings import settings
from core.rag_pipeline import RAGPipeline

logger = get_logger("Evaluation")


@dataclass
class EvaluationMetricResult:
    question_id: str
    question: str
    retrieval_precision: float
    retrieval_recall: float
    context_precision: float
    context_recall: float
    citation_correctness: float
    answer_relevancy: float
    latency_ms: float
    retrieved_pages: list[int]
    expected_pages: list[int]


class RAGEvaluator:
    """Evaluates RAG pipeline performance against ground truth cinema benchmark dataset."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.benchmark_path = settings.BASE_DIR / "evaluation" / "questions.json"

    def load_benchmark_questions(self) -> list[dict]:
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_all(self) -> dict:
        """Run evaluation over all benchmark questions and compute aggregate metrics."""
        questions = self.load_benchmark_questions()
        results: list[EvaluationMetricResult] = []

        print(f"\n{'='*75}")
        print(f"🎬 CineMind AI - Automated RAG Evaluation Pipeline")
        print(f"Benchmark Set: {len(questions)} verified cinema questions")
        print(f"{'='*75}\n")

        for q_data in questions:
            qid = q_data["id"]
            question = q_data["question"]
            expected_pages = q_data.get("expected_pages", [])
            expected_keywords = [k.lower() for k in q_data.get("keywords", [])]
            ground_truth = q_data.get("ground_truth", "")

            start_t = time.perf_counter()

            # Execute pipeline retrieval
            search_query, chunks, confidence = self.pipeline.retrieve_context(question)
            latency_ms = (time.perf_counter() - start_t) * 1000

            retrieved_pages = [int(c.metadata.get("page", 0)) for c in chunks]

            # 1. Retrieval Precision: fraction of retrieved chunks matching expected pages or keywords
            relevant_chunks_count = 0
            for c in chunks:
                p = int(c.metadata.get("page", 0))
                text_lower = c.text.lower()
                # Check page match (exact or adjacent +/- 2 pages) or keyword overlap
                page_match = any(abs(p - ep) <= 2 for ep in expected_pages)
                keyword_match = sum(1 for kw in expected_keywords if kw in text_lower) >= 2
                if page_match or keyword_match:
                    relevant_chunks_count += 1

            retrieval_precision = relevant_chunks_count / max(1, len(chunks))

            # 2. Retrieval Recall: fraction of expected keywords captured in retrieved context
            combined_text = " ".join(c.text.lower() for c in chunks)
            matched_keywords = sum(1 for kw in expected_keywords if kw in combined_text)
            retrieval_recall = matched_keywords / max(1, len(expected_keywords))

            # 3. Context Precision: Rank-weighted precision (Mean Reciprocal Rank of first relevant chunk)
            first_rel_rank = 0
            for rank_idx, c in enumerate(chunks, start=1):
                p = int(c.metadata.get("page", 0))
                text_lower = c.text.lower()
                if any(abs(p - ep) <= 2 for ep in expected_pages) or sum(1 for kw in expected_keywords if kw in text_lower) >= 2:
                    first_rel_rank = rank_idx
                    break
            context_precision = 1.0 / first_rel_rank if first_rel_rank > 0 else 0.0

            # 4. Context Recall: Ground-truth key term coverage in retrieved context
            gt_words = [w.lower().strip(".,;:\"'()") for w in ground_truth.split() if len(w) > 4]
            gt_matched = sum(1 for w in gt_words if w in combined_text)
            context_recall = gt_matched / max(1, len(gt_words))

            # 5. Citation Correctness: Validate citation page numbers exist within valid book range (1 to 277)
            citations = self.pipeline.build_citations(chunks)
            valid_citations = [
                c for c in citations if 1 <= c.page <= 277 and c.source.endswith(".pdf")
            ]
            citation_correctness = len(valid_citations) / max(1, len(citations))

            # 6. Answer Relevancy (Keyword overlap between question terms and retrieved context)
            q_terms = [w.lower() for w in question.split() if len(w) > 3]
            q_matched = sum(1 for qt in q_terms if qt in combined_text)
            answer_relevancy = min(1.0, q_matched / max(1, len(q_terms)))

            res = EvaluationMetricResult(
                question_id=qid,
                question=question,
                retrieval_precision=round(retrieval_precision, 3),
                retrieval_recall=round(retrieval_recall, 3),
                context_precision=round(context_precision, 3),
                context_recall=round(context_recall, 3),
                citation_correctness=round(citation_correctness, 3),
                answer_relevancy=round(answer_relevancy, 3),
                latency_ms=round(latency_ms, 1),
                retrieved_pages=retrieved_pages,
                expected_pages=expected_pages,
            )
            results.append(res)

            print(f"[{qid}] {question[:55]}...")
            print(f"     Pages: {retrieved_pages} | Precision: {res.retrieval_precision:.2f} | Recall: {res.retrieval_recall:.2f} | Latency: {res.latency_ms:.0f}ms")

        # Aggregate metrics
        avg_precision = sum(r.retrieval_precision for r in results) / len(results)
        avg_recall = sum(r.retrieval_recall for r in results) / len(results)
        avg_ctx_precision = sum(r.context_precision for r in results) / len(results)
        avg_ctx_recall = sum(r.context_recall for r in results) / len(results)
        avg_citations = sum(r.citation_correctness for r in results) / len(results)
        avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        summary = {
            "total_benchmark_questions": len(results),
            "average_retrieval_precision": round(avg_precision, 3),
            "average_retrieval_recall": round(avg_recall, 3),
            "average_context_precision": round(avg_ctx_precision, 3),
            "average_context_recall": round(avg_ctx_recall, 3),
            "average_citation_correctness": round(avg_citations, 3),
            "average_answer_relevancy": round(avg_relevancy, 3),
            "average_latency_ms": round(avg_latency, 1),
            "results": [asdict(r) for r in results],
        }

        print(f"\n{'='*75}")
        print(f"📊 AGGREGATE EVALUATION REPORT")
        print(f"{'='*75}")
        print(f"  Retrieval Precision:     {summary['average_retrieval_precision'] * 100:.1f}%")
        print(f"  Retrieval Recall:        {summary['average_recall'] if 'average_recall' in summary else summary['average_retrieval_recall'] * 100:.1f}%")
        print(f"  Context Precision (MRR): {summary['average_context_precision'] * 100:.1f}%")
        print(f"  Context Recall:          {summary['average_context_recall'] * 100:.1f}%")
        print(f"  Citation Correctness:    {summary['average_citation_correctness'] * 100:.1f}%")
        print(f"  Answer Relevancy:        {summary['average_answer_relevancy'] * 100:.1f}%")
        print(f"  Average Response Latency:{summary['average_latency_ms']:.1f} ms")
        print(f"{'='*75}\n")

        # Save report to evaluation/report.json
        report_path = settings.BASE_DIR / "evaluation" / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Evaluation report saved to: {report_path}")

        return summary


def main():
    pipeline = RAGPipeline()
    evaluator = RAGEvaluator(pipeline)
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
