"""
SCM LOCOMO Benchmark Runner
Evaluates SCM against the LoCoMo (Long Conversation Memory) benchmark.

LoCoMo is the industry-standard benchmark for evaluating long-term
conversational memory (ACL 2024, Maharana et al.).

QA Categories:
  1 = single-hop (direct fact retrieval)
  2 = temporal (when did X happen)
  3 = multi-hop (requires combining multiple facts)
  4 = open-ended (free-form reasoning about preferences/events)
  5 = adversarial (questions that swap speaker identities to test confusion)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from src.core.models import Concept, Episode, ImportanceVector, ConceptType
from src.core.encoder import MeaningEncoder
from src.core.value_tagger import ValueTagger
from src.core.working_memory import WorkingMemory
from src.core.long_term_memory import LongTermMemory
from src.sleep.sleep_cycle import SleepCycleOrchestrator
from src.llm import LLMExtractor
from src.core.time_utils import utc_now

CAT_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-ended",
    5: "adversarial",
}


class SCMLocomoEvaluator:
    def __init__(self, data_path: str, use_llm: bool = True):
        self.data_path = data_path
        self.use_llm = use_llm
        self.results = {}

        if use_llm:
            self.llm = LLMExtractor()
            self.encoder = MeaningEncoder(llm=self.llm)
        else:
            self.llm = None
            self.encoder = None

        from sentence_transformers import SentenceTransformer
        if self.encoder and hasattr(self.encoder, 'embedding_model'):
            self.embedding_model = self.encoder.embedding_model
        else:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self) -> List[Dict]:
        with open(self.data_path) as f:
            return json.load(f)

    def _get_embedding(self, text: str) -> List[float]:
        emb = self.embedding_model.encode(text)
        # encode() may return either a numpy array (sentence-transformers)
        # or a plain list (HashEmbeddingModel fallback).
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def _ingest_conversation(self, conv: Dict) -> Tuple[LongTermMemory, WorkingMemory, Dict]:
        ltm = LongTermMemory()
        wm = WorkingMemory(capacity=7)
        tagger = ValueTagger()
        sleep_orch = SleepCycleOrchestrator()

        conv_data = conv['conversation']
        speaker_a = conv_data.get('speaker_a', 'Speaker A')
        speaker_b = conv_data.get('speaker_b', 'Speaker B')

        session_keys = sorted(
            [k for k in conv_data.keys()
             if k.startswith('session_') and not k.endswith('_date_time')],
            key=lambda x: int(x.split('_')[1])
        )

        stats = {
            'total_turns': 0,
            'total_concepts': 0,
            'sessions': len(session_keys),
            'sleep_cycles': 0,
        }

        for session_key in session_keys:
            session_turns = conv_data[session_key]

            for turn in session_turns:
                speaker = turn.get('speaker', '')
                text = turn.get('text', '')
                if not text.strip():
                    continue

                stats['total_turns'] += 1
                full_text = f"{speaker}: {text}"

                concepts = self._extract_concepts(full_text, speaker_a, speaker_b)

                for concept in concepts:
                    concept.importance = tagger.tag(concept)
                    if not concept.embedding:
                        concept.embedding = self._get_embedding(concept.description)
                    ltm.add_concept(concept)

                if concepts:
                    ep = Episode(
                        concept_ids=[c.id for c in concepts],
                        raw_content=full_text,
                        importance=concepts[0].importance,
                        source=speaker.lower(),
                    )
                    wm.store(ep)
                    stats['total_concepts'] += len(concepts)

            # Sleep between sessions
            if wm.size() >= 3:
                concepts_list = ltm.get_all_concepts(include_suppressed=False)
                try:
                    success, cycle, cycle_stats = sleep_orch.begin_sleep_cycle(
                        concepts=concepts_list,
                        relations=[],
                        episodes=wm.get_all(),
                        force=True,
                    )
                    if success:
                        stats['sleep_cycles'] += 1
                        if 'updated_concepts' in cycle_stats:
                            surviving = {c.id for c in cycle_stats['updated_concepts']}
                            forgotten = set(ltm._concept_cache.keys()) - surviving
                            for fid in forgotten:
                                if fid in ltm._concept_cache:
                                    del ltm._concept_cache[fid]
                                if fid in ltm.graph:
                                    ltm.graph.remove_node(fid)
                        wm.clear()
                except Exception as e:
                    print(f"    Sleep error: {e}")

        return ltm, wm, stats

    def _extract_concepts(self, text: str, speaker_a: str, speaker_b: str) -> List[Concept]:
        if self.llm and self.use_llm:
            try:
                concepts = self.encoder.extract(text)
                if concepts:
                    return concepts
            except Exception:
                pass
        return self._extract_heuristic(text, speaker_a, speaker_b)

    def _extract_heuristic(self, text: str, speaker_a: str, speaker_b: str) -> List[Concept]:
        concepts = []
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue

            c_type = ConceptType.FACT
            if any(w in sent.lower() for w in ['prefer', 'like', 'love', 'enjoy', 'hate', 'favorite']):
                c_type = ConceptType.PREFERENCE
            elif any(w in sent.lower() for w in ['went to', 'visited', 'attended', 'joined', 'participated']):
                c_type = ConceptType.EVENT
            elif any(w in sent.lower() for w in ['work', 'job', 'career', 'study', 'school', 'counseling']):
                c_type = ConceptType.FACT

            concepts.append(Concept(
                type=c_type,
                description=sent,
                importance=ImportanceVector(novelty=0.6, task_relevance=0.7),
            ))

        if not concepts:
            concepts.append(Concept(
                type=ConceptType.FACT,
                description=text[:200],
                importance=ImportanceVector(),
            ))

        return concepts[:5]

    def _answer_question(self, question: str, ltm: LongTermMemory) -> str:
        query_emb = np.array(self._get_embedding(question))
        all_concepts = list(ltm._concept_cache.values())
        if not all_concepts:
            return ""

        scored = []
        for concept in all_concepts:
            if concept.embedding:
                c_emb = np.array(concept.embedding)
                norm_c = np.linalg.norm(c_emb)
                norm_q = np.linalg.norm(query_emb)
                if norm_c > 0 and norm_q > 0:
                    sim = float(np.dot(c_emb, query_emb) / (norm_c * norm_q))
                else:
                    sim = 0.0
            else:
                sim = 0.0

            q_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
            d_words = set(re.findall(r'\b\w{3,}\b', concept.description.lower()))
            keyword_overlap = len(q_words & d_words) / len(q_words) if q_words else 0

            final_score = 0.5 * sim + 0.3 * keyword_overlap + 0.2 * concept.importance.overall
            scored.append((concept, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        top_k = min(5, len(scored))
        relevant = [c.description for c, s in scored[:top_k]]
        context = "\n".join(f"- {r}" for r in relevant)

        if self.llm:
            try:
                prompt = (
                    "Based on the following memories, answer the question briefly. "
                    "If the memories don't contain the answer, say 'I don't know'.\n\n"
                    f"Memories:\n{context}\n\n"
                    f"Question: {question}\n\nAnswer:"
                )
                response = self.llm._chat(prompt, num_predict=128)
                return response.strip()
            except Exception:
                pass

        return relevant[0] if relevant else ""

    def _score_answer(self, predicted: str, ground_truth, category: int) -> float:
        # LOCOMO ground truths can be ints (years, ages) or floats — coerce to str.
        ground_truth = "" if ground_truth is None else str(ground_truth)
        predicted = "" if predicted is None else str(predicted)
        if not predicted or predicted.lower().strip() in (
            "i don't know", "unknown", "not sure", "n/a", ""
        ):
            return 0.0

        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()

        if pred_lower == gt_lower:
            return 1.0

        pred_tokens = set(re.findall(r'\b\w{2,}\b', pred_lower))
        gt_tokens = set(re.findall(r'\b\w{2,}\b', gt_lower))

        if not gt_tokens:
            return 0.0

        overlap = pred_tokens & gt_tokens
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        recall_threshold = {1: 0.5, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.5}.get(category, 0.5)
        cat_multiplier = {2: 0.8, 3: 0.9}.get(category, 1.0)

        if recall >= recall_threshold:
            return 1.0
        return f1 * cat_multiplier

    def evaluate_conversation(self, conv: Dict, conv_idx: int) -> Dict:
        speakers = f"{conv['conversation'].get('speaker_a', '?')}/{conv['conversation'].get('speaker_b', '?')}"
        print(f"\n{'='*60}")
        print(f"  Conversation {conv_idx}: {speakers}")
        print(f"{'='*60}")

        print("  Ingesting conversation...")
        ingest_start = time.time()
        ltm, wm, stats = self._ingest_conversation(conv)
        ingest_time = time.time() - ingest_start
        print(f"  Ingested: {stats['total_turns']} turns, "
              f"{stats['total_concepts']} concepts, "
              f"{stats['sleep_cycles']} sleeps ({ingest_time:.1f}s)")

        qa_pairs = conv.get('qa', [])
        print(f"  Evaluating {len(qa_pairs)} QA pairs...")

        cat_scores = defaultdict(list)
        all_scores = []

        for i, qa in enumerate(qa_pairs):
            question = qa['question']
            category = qa.get('category', 0)

            if category == 5:
                continue

            answer = qa.get('answer', '')
            predicted = self._answer_question(question, ltm)
            score = self._score_answer(predicted, answer, category)

            cat_scores[category].append(score)
            all_scores.append(score)

            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(qa_pairs)}")

        result = {
            'conv_idx': conv_idx,
            'speakers': speakers,
            'ingest_stats': stats,
            'ingest_time_s': round(ingest_time, 1),
            'total_qa': len(qa_pairs),
            'evaluated_qa': len(all_scores),
            'overall_score': round(sum(all_scores) / len(all_scores), 4) if all_scores else 0,
            'category_scores': {},
        }

        for cat in sorted(cat_scores.keys()):
            scores = cat_scores[cat]
            avg = sum(scores) / len(scores) if scores else 0
            result['category_scores'][cat] = {
                'name': CAT_NAMES.get(cat, f'cat_{cat}'),
                'count': len(scores),
                'score': round(avg, 4),
            }
            print(f"  Cat {cat} ({CAT_NAMES.get(cat, '?')}): {avg:.3f} ({len(scores)} Q)")

        print(f"  Overall: {result['overall_score']:.3f}")
        return result

    def run(self, max_conversations: int = None) -> Dict:
        data = self.load_data()
        if max_conversations:
            data = data[:max_conversations]

        print(f"\n{'#'*60}")
        print(f"# SCM LOCOMO BENCHMARK")
        print(f"# Conversations: {len(data)}")
        print(f"# LLM: {'llama3.2 (Ollama)' if self.use_llm else 'heuristic'}")
        print(f"# Time: {utc_now().isoformat()}")
        print(f"{'#'*60}")

        start_time = time.time()
        conv_results = []

        for i, conv in enumerate(data):
            result = self.evaluate_conversation(conv, i)
            conv_results.append(result)

        total_time = time.time() - start_time

        all_scores = [r['overall_score'] for r in conv_results]
        cat_aggregate = defaultdict(list)
        for r in conv_results:
            for cat, info in r['category_scores'].items():
                cat_aggregate[cat].append(info['score'])

        report = {
            'timestamp': utc_now().isoformat(),
            'benchmark': 'LoCoMo (ACL 2024)',
            'num_conversations': len(data),
            'total_time_s': round(total_time, 1),
            'overall_score': round(sum(all_scores) / len(all_scores), 4) if all_scores else 0,
            'category_scores': {},
            'conversations': conv_results,
            'baselines': {
                'Mem0': 0.671,
                'MemGPT': 0.42,
                'RAG (gpt-3.5-turbo)': 0.56,
                'MemMachine': 0.899,
            },
        }

        for cat in sorted(cat_aggregate.keys()):
            scores = cat_aggregate[cat]
            report['category_scores'][cat] = {
                'name': CAT_NAMES.get(cat, f'cat_{cat}'),
                'score': round(sum(scores) / len(scores), 4),
                'num_conversations': len(scores),
            }

        return report

    def print_report(self, report: Dict):
        print(f"\n\n{'#'*60}")
        print(f"# LOCOMO BENCHMARK RESULTS")
        print(f"{'#'*60}")
        print(f"Overall Score: {report['overall_score']:.3f}")
        print(f"Conversations: {report['num_conversations']}")
        print(f"Total Time: {report['total_time_s']:.0f}s")
        print()
        print("Category Breakdown:")
        for cat in sorted(report['category_scores'].keys()):
            info = report['category_scores'][cat]
            print(f"  Cat {cat} ({info['name']:12s}): {info['score']:.3f}")
        print()
        print("Baselines (reported):")
        for name, score in report['baselines'].items():
            marker = " <-- SCM beats" if report['overall_score'] > score else ""
            print(f"  {name:25s}: {score:.3f}{marker}")
        print()
        print("Per-conversation:")
        for r in report['conversations']:
            print(f"  Conv {r['conv_idx']} ({r['speakers']:20s}): "
                  f"{r['overall_score']:.3f}  ({r['evaluated_qa']} QA, "
                  f"{r['ingest_stats']['sleep_cycles']} sleeps)")
        print(f"\n{'#'*60}")

    def save_report(self, report: Dict, filename: str = "locomo_report.json"):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {filepath}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SCM LOCOMO Benchmark')
    parser.add_argument('--data', default='data/locomo/locomo10.json')
    parser.add_argument('--no-llm', action='store_true')
    parser.add_argument('--max-conv', type=int, default=None)
    parser.add_argument('--output', default='locomo_report.json')
    args = parser.parse_args()

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.data
    )

    evaluator = SCMLocomoEvaluator(data_path=data_path, use_llm=not args.no_llm)
    report = evaluator.run(max_conversations=args.max_conv)
    evaluator.print_report(report)
    evaluator.save_report(report, args.output)
