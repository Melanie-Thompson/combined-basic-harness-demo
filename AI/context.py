"""TDA-based context management — persistent homology for topic clustering,
with archival and rehydration of pruned clusters."""

import json
import logging
import os

import gudhi
import numpy as np
from langchain_core.messages import BaseMessage
from pydantic import TypeAdapter

from AI import redis_store
from AI.embeddings import embed
from AI.models.context import ArchivedCluster
from AI.tokenizer import count_messages

log = logging.getLogger("context")

DEFAULT_MAX_TOKENS = 32_768
REHYDRATION_THRESHOLD = 0.5

_clusters_adapter = TypeAdapter(list[ArchivedCluster])


class ContextManager:
    def __init__(self, session_id: str, max_tokens: int | None = None):
        self.session_id = session_id
        self.max_tokens = max_tokens or int(
            os.getenv("CONTEXT_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
        )

    async def prepare(
        self, history: list[BaseMessage], query: str
    ) -> list[BaseMessage]:
        if not history:
            return history

        token_count = count_messages(history)
        if token_count <= self.max_tokens:
            return history

        log.info(
            "context exceeds budget (%d > %d), running TDA clustering",
            token_count, self.max_tokens,
        )

        texts = [
            m.content if isinstance(m.content, str) else str(m.content)
            for m in history
        ]
        query_vec = embed([query])[0]
        all_vecs = embed(texts)

        archived = await self._load_archived()
        rehydrated_messages, rehydrated_indices_start = await self._rehydrate(
            query_vec, archived, history,
        )

        if rehydrated_messages:
            history = list(history) + rehydrated_messages
            rehydrated_texts = [
                m.content if isinstance(m.content, str) else str(m.content)
                for m in rehydrated_messages
            ]
            rehydrated_vecs = embed(rehydrated_texts)
            all_vecs = np.vstack([all_vecs, rehydrated_vecs])

        clusters = self._cluster(all_vecs)
        scored = self._score_clusters(clusters, all_vecs, query_vec)
        selected, overflow = self._pack(scored, history)

        if overflow:
            await self._archive(overflow, all_vecs, history)

        selected.sort(key=lambda m: history.index(m) if m in history else len(history))

        log.info(
            "context: %d messages selected, %d clusters archived",
            len(selected), len(overflow),
        )
        return selected

    def is_follow_up(
        self,
        query: str,
        history: list[BaseMessage],
        tool_descriptions: dict[str, str],
    ) -> bool:
        """Check if query is a topological follow-up to a tool-relevant cluster.

        Only tool-relevant messages enter the TDA. Builds a Rips complex on
        their embeddings to find the natural clustering scale via the
        persistence diagram, then checks if the query falls within that
        topological structure.
        """
        from AI.embeddings import semantic_similarity

        if not history:
            return False

        tool_texts = []
        for msg in history:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            for desc in tool_descriptions.values():
                if semantic_similarity(content, desc) >= 0.25:
                    tool_texts.append(content)
                    break

        if not tool_texts:
            return False

        tool_vecs = embed(tool_texts)
        query_vec = embed([query])[0]

        # Build Rips complex on tool-relevant messages only.
        # Use the persistence diagram to find the natural scale — the
        # largest gap separates the tight core from outliers.
        rips = gudhi.RipsComplex(points=tool_vecs, max_edge_length=2.0)
        st = rips.create_simplex_tree(max_dimension=1)
        dgm = st.persistence()

        deaths = sorted(
            d for dim, (b, d) in dgm if dim == 0 and d != float("inf")
        )

        if not deaths:
            cluster_scale = 0.0
        elif len(deaths) == 1:
            cluster_scale = deaths[0]
        else:
            best_gap = 0.0
            epsilon = deaths[-1]
            for i in range(len(deaths) - 1):
                gap = deaths[i + 1] - deaths[i]
                if gap > best_gap:
                    best_gap = gap
                    epsilon = deaths[i]
            cluster_scale = epsilon if best_gap > 1e-6 else deaths[len(deaths) // 2]

        if cluster_scale < 1e-6:
            cos = float(np.dot(tool_vecs[0], query_vec) / (
                np.linalg.norm(tool_vecs[0]) * np.linalg.norm(query_vec) + 1e-9
            ))
            connected = cos >= 0.3
            log.info("TDA follow-up: single-point, cosine=%.3f, connected=%s", cos, connected)
            return connected

        from scipy.spatial.distance import cdist
        dists_to_tools = cdist([query_vec], tool_vecs, metric="euclidean")[0]
        min_dist = float(min(dists_to_tools))
        ratio = min_dist / cluster_scale

        connected = ratio <= 1.75
        log.info(
            "TDA follow-up: min_dist=%.3f, cluster_scale=%.3f, ratio=%.3f, connected=%s",
            min_dist, cluster_scale, ratio, connected,
        )
        return connected

    def _cluster(self, embeddings: np.ndarray) -> list[list[int]]:
        n = len(embeddings)
        if n <= 1:
            return [list(range(n))]

        rips = gudhi.RipsComplex(points=embeddings, max_edge_length=2.0)
        st = rips.create_simplex_tree(max_dimension=1)
        dgm = st.persistence()
        deaths = sorted(
            d for dim, (b, d) in dgm if dim == 0 and d != float("inf")
        )

        if not deaths:
            return [list(range(n))]

        # Find the largest gap in death values to pick epsilon
        best_gap = 0.0
        epsilon = deaths[-1]
        for i in range(len(deaths) - 1):
            gap = deaths[i + 1] - deaths[i]
            if gap > best_gap:
                best_gap = gap
                epsilon = deaths[i] + gap / 2

        # If no meaningful gap, use the median death
        if best_gap < 1e-6:
            epsilon = deaths[len(deaths) // 2]

        # Build clusters using union-find at the chosen epsilon
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial.distance import pdist, squareform

        dist_matrix = squareform(pdist(embeddings))
        adjacency = (dist_matrix <= epsilon).astype(int)
        np.fill_diagonal(adjacency, 0)

        n_components, labels = connected_components(adjacency, directed=False)

        clusters: list[list[int]] = [[] for _ in range(n_components)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        return clusters

    def _score_clusters(
        self,
        clusters: list[list[int]],
        embeddings: np.ndarray,
        query_vec: np.ndarray,
    ) -> list[tuple[float, list[int]]]:
        scored = []
        query_norm = np.linalg.norm(query_vec) + 1e-9
        for members in clusters:
            centroid = embeddings[members].mean(axis=0)
            cos = np.dot(centroid, query_vec) / (
                np.linalg.norm(centroid) * query_norm + 1e-9
            )
            scored.append((float(cos), members))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _pack(
        self,
        scored: list[tuple[float, list[int]]],
        messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], list[list[int]]]:
        selected: list[BaseMessage] = []
        overflow: list[list[int]] = []
        remaining_budget = self.max_tokens

        for score, members in scored:
            cluster_msgs = [messages[i] for i in members if i < len(messages)]
            cost = count_messages(cluster_msgs)
            if cost <= remaining_budget:
                selected.extend(cluster_msgs)
                remaining_budget -= cost
            else:
                overflow.append(members)

        return selected, overflow

    async def _archive(
        self,
        overflow_clusters: list[list[int]],
        embeddings: np.ndarray,
        messages: list[BaseMessage],
    ) -> None:
        from AI.sessions import _langchain_to_models

        existing = await self._load_archived()

        for members in overflow_clusters:
            cluster_msgs_lc = [messages[i] for i in members if i < len(messages)]
            if not cluster_msgs_lc:
                continue
            centroid = embeddings[members].mean(axis=0)
            pydantic_msgs = _langchain_to_models(cluster_msgs_lc)
            token_count = count_messages(cluster_msgs_lc)
            existing.append(ArchivedCluster(
                centroid=centroid.tolist(),
                messages=pydantic_msgs,
                token_count=token_count,
            ))

        data = _clusters_adapter.dump_json(existing).decode()
        await redis_store.set_clusters(self.session_id, data)

    async def _load_archived(self) -> list[ArchivedCluster]:
        raw = await redis_store.get_clusters(self.session_id)
        if raw is None:
            return []
        try:
            return _clusters_adapter.validate_json(raw)
        except Exception:
            return []

    async def _rehydrate(
        self,
        query_vec: np.ndarray,
        archived: list[ArchivedCluster],
        current_history: list[BaseMessage],
    ) -> tuple[list[BaseMessage], int]:
        if not archived:
            return [], len(current_history)

        from AI.sessions import _model_to_langchain

        rehydrated: list[BaseMessage] = []
        remaining: list[ArchivedCluster] = []
        query_norm = np.linalg.norm(query_vec) + 1e-9

        for cluster in archived:
            centroid = np.array(cluster.centroid)
            cos = np.dot(centroid, query_vec) / (
                np.linalg.norm(centroid) * query_norm + 1e-9
            )
            if cos >= REHYDRATION_THRESHOLD:
                log.info("rehydrating cluster (similarity=%.3f)", cos)
                for msg in cluster.messages:
                    rehydrated.append(_model_to_langchain(msg))
            else:
                remaining.append(cluster)

        if len(remaining) != len(archived):
            data = _clusters_adapter.dump_json(remaining).decode()
            await redis_store.set_clusters(self.session_id, data)

        return rehydrated, len(current_history)
