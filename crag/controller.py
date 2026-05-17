from typing import Any, Dict, List, Optional

from generation import NO_ANSWER_RESPONSE, generate_answer

from .config import CRAGConfig
from .context_filter import filter_context
from .logging_utils import log_event
from .query_rewriter import rewrite_query
from .retrieval_evaluator import evaluate_retrieval
from .reranker import rerank_chunks
from .types import CRAGResponse, RetrievalEvaluation, RetrievalResult
from .validator import validate_answer


class CorrectiveRAGController:
    def __init__(
        self,
        config: CRAGConfig,
        retriever,
        logger=None,
        generator=None,
    ) -> None:
        self._config = config
        self._retriever = retriever
        self._logger = logger
        self._generator = generator or generate_answer

    def run(self, query: str) -> CRAGResponse:
        attempts: List[Dict[str, Any]] = []
        evaluation: Optional[RetrievalEvaluation] = None
        retrieval: Optional[RetrievalResult] = None
        selected_chunks = []

        rewrite_candidates: List[str] = []
        rewritten_query: Optional[str] = None

        total_attempts = self._config.max_retries + 1
        for attempt in range(total_attempts):
            if attempt == 0:
                current_query = query
            else:
                if not rewrite_candidates:
                    if self._config.enable_query_rewrite:
                        rewrite_candidates = rewrite_query(query, self._config)
                        rewrite_candidates = [
                            candidate
                            for candidate in rewrite_candidates
                            if candidate != query
                        ]
                if not rewrite_candidates:
                    break
                current_query = rewrite_candidates.pop(0)
                rewritten_query = current_query

            chunks = self._retriever.retrieve(current_query, self._config.top_k)
            reranked = rerank_chunks(current_query, chunks, self._config)
            evaluation = evaluate_retrieval(reranked, self._config)

            retrieval = RetrievalResult(
                query=query,
                rewritten_query=rewritten_query,
                attempt=attempt,
                used_hybrid=self._config.enable_hybrid,
                chunks=reranked,
            )

            attempt_info = {
                "attempt": attempt,
                "query": current_query,
                "decision": evaluation.decision,
                "max_score": evaluation.max_score,
                "avg_score": evaluation.avg_score,
            }
            attempts.append(attempt_info)
            log_event(self._logger, "retrieval_attempt", attempt_info)

            selected_chunks = reranked
            if evaluation.decision == "relevant":
                break

            if attempt >= self._config.max_retries:
                break

            if not rewrite_candidates and self._config.enable_query_rewrite:
                rewrite_candidates = rewrite_query(query, self._config)
                rewrite_candidates = [
                    candidate for candidate in rewrite_candidates if candidate != query
                ]

            if not rewrite_candidates:
                break

        if evaluation is None or retrieval is None:
            evaluation = RetrievalEvaluation(
                decision="irrelevant",
                max_score=0.0,
                avg_score=0.0,
                reason="no_attempts",
            )
            retrieval = RetrievalResult(
                query=query,
                rewritten_query=None,
                attempt=0,
                used_hybrid=self._config.enable_hybrid,
                chunks=[],
            )

        if evaluation.decision == "irrelevant" or not selected_chunks:
            return CRAGResponse(
                answer=self._insufficient_info_response(),
                context="",
                evaluation=evaluation,
                retrieval=retrieval,
                validation=None,
                attempts=attempts,
            )

        filtered = filter_context(selected_chunks, self._config)
        context = "\n\n".join(chunk.text for chunk in filtered)
        if not context:
            return CRAGResponse(
                answer=self._insufficient_info_response(),
                context="",
                evaluation=evaluation,
                retrieval=retrieval,
                validation=None,
                attempts=attempts,
            )

        strict = evaluation.decision != "relevant"
        answer, validation = self._generate_and_validate(
            query, context, strict, filtered
        )

        if answer is None:
            answer = self._insufficient_info_response()

        return CRAGResponse(
            answer=answer,
            context=context,
            evaluation=evaluation,
            retrieval=retrieval,
            validation=validation,
            attempts=attempts,
        )

    def _generate_and_validate(
        self,
        query: str,
        context: str,
        strict: bool,
        filtered_chunks,
    ):
        validation = None
        try:
            answer = self._generator(query, context, strict=strict)
        except Exception as exc:
            log_event(
                self._logger,
                "generation_error",
                {"error": str(exc), "phase": "initial"},
            )
            return None, None

        if not self._config.enable_validation:
            return answer, None

        validation = validate_answer(
            answer,
            context,
            self._config,
            no_answer_response=NO_ANSWER_RESPONSE,
        )
        if validation.is_grounded:
            return answer, validation

        max_attempts = max(1, self._config.max_generation_attempts)
        for _ in range(1, max_attempts):
            strict_context = (
                filtered_chunks[0].text if filtered_chunks else context
            )
            try:
                answer = self._generator(query, strict_context, strict=True)
            except Exception as exc:
                log_event(
                    self._logger,
                    "generation_error",
                    {"error": str(exc), "phase": "retry"},
                )
                return None, validation

            validation = validate_answer(
                answer,
                strict_context,
                self._config,
                no_answer_response=NO_ANSWER_RESPONSE,
            )
            if validation.is_grounded:
                return answer, validation

        return None, validation

    def _insufficient_info_response(self) -> str:
        return (
            f"{NO_ANSWER_RESPONSE} "
            "Please add more sources or rephrase your question."
        )
