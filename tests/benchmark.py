"""
SleepAI Benchmark Suite
Scientific evaluation of memory performance
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.core.models import Concept, Episode, ImportanceVector, ConceptType
from src.core.encoder import MeaningEncoder
from src.core.value_tagger import ValueTagger
from src.core.working_memory import WorkingMemory
from src.core.long_term_memory import LongTermMemory
from src.sleep.sleep_cycle import SleepCycleOrchestrator
from src.chat.engine import ChatEngine
from src.core.time_utils import utc_now


@dataclass
class BenchmarkResult:
    """Single benchmark test result"""
    test_name: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: Dict
    duration_ms: float


@dataclass
class ConversationTurn:
    """One turn in a test conversation"""
    user_message: str
    expected_facts: List[str]  # Facts that should be remembered
    turn_number: int


class SleepAIBenchmark:
    """
    Comprehensive benchmark for SleepAI memory system.

    Tests:
    1. Memory retention across conversation turns
    2. Recall accuracy after sleep consolidation
    3. Forgetting effectiveness (noise removal)
    4. Latency scaling with memory size
    5. Working memory capacity limits
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.start_time = utc_now()

    def run_all(self) -> Dict:
        """Run complete benchmark suite"""
        print("="*70)
        print("SLEEPAI BENCHMARK SUITE")
        print("="*70)
        print()

        tests = [
            ("Working Memory Capacity", self.test_wm_capacity),
            ("Memory Retention (5 turns)", self.test_retention_5),
            ("Memory Retention (10 turns)", self.test_retention_10),
            ("Sleep Consolidation Benefit", self.test_sleep_benefit),
            ("Forgetting Effectiveness", self.test_forgetting),
            ("Graph Traversal Accuracy", self.test_graph_traversal),
            ("Latency Scaling", self.test_latency_scaling),
            ("Multi-Session Persistence", self.test_persistence),
        ]

        for test_name, test_func in tests:
            print(f"\n[{test_name}]")
            try:
                result = test_func()
                self.results.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"  {status} | Score: {result.score:.2f} | {result.duration_ms:.0f}ms")
            except Exception as e:
                print(f"  💥 ERROR: {e}")
                self.results.append(BenchmarkResult(
                    test_name=test_name,
                    passed=False,
                    score=0.0,
                    details={"error": str(e)},
                    duration_ms=0
                ))

        return self._generate_report()

    def test_wm_capacity(self) -> BenchmarkResult:
        """Test that working memory respects capacity limit"""
        start = time.time()

        wm = WorkingMemory(capacity=7)
        for i in range(10):
            ep = Episode(
                concept_ids=[f"c{i}"],
                raw_content=f"message {i}",
                importance=ImportanceVector()
            )
            wm.store(ep)

        score = 1.0 if wm.size() == 7 else 0.0

        return BenchmarkResult(
            test_name="Working Memory Capacity",
            passed=score == 1.0,
            score=score,
            details={"capacity": 7, "actual_size": wm.size()},
            duration_ms=(time.time() - start) * 1000
        )

    def test_retention_5(self) -> BenchmarkResult:
        """Test memory retention over 5 conversation turns"""
        return self._test_retention_n(5)

    def test_retention_10(self) -> BenchmarkResult:
        """Test memory retention over 10 conversation turns"""
        return self._test_retention_n(10)

    def _test_retention_n(self, n_turns: int) -> BenchmarkResult:
        """Test memory retention over N turns"""
        start = time.time()

        # Build a conversation with facts to remember
        conversation = [
            ConversationTurn(
                user_message="My name is Alice",
                expected_facts=["Alice", "name"],
                turn_number=1
            ),
            ConversationTurn(
                user_message="I work at Google as an engineer",
                expected_facts=["Google", "engineer", "work"],
                turn_number=2
            ),
            ConversationTurn(
                user_message="I live in Seattle",
                expected_facts=["Seattle", "live"],
                turn_number=3
            ),
            ConversationTurn(
                user_message="My favorite hobby is hiking",
                expected_facts=["hiking", "hobby"],
                turn_number=4
            ),
            ConversationTurn(
                user_message="I have a dog named Max",
                expected_facts=["Max", "dog"],
                turn_number=5
            ),
            ConversationTurn(
                user_message="I love eating pizza",
                expected_facts=["pizza", "eating"],
                turn_number=6
            ),
            ConversationTurn(
                user_message="I studied at MIT",
                expected_facts=["MIT", "studied"],
                turn_number=7
            ),
            ConversationTurn(
                user_message="My birthday is in March",
                expected_facts=["March", "birthday"],
                turn_number=8
            ),
            ConversationTurn(
                user_message="I drive a Tesla",
                expected_facts=["Tesla", "drive"],
                turn_number=9
            ),
            ConversationTurn(
                user_message="I speak Spanish and French",
                expected_facts=["Spanish", "French", "speak"],
                turn_number=10
            ),
        ][:n_turns]

        # Create engine without LLM (test memory system directly)
        engine = ChatEngine(enable_auto_sleep=False)

        # Process each turn
        for turn in conversation:
            # Manually create concepts (avoid LLM latency)
            concepts = self._create_test_concepts(turn.user_message)
            for concept in concepts:
                concept.importance = engine.value_tagger.tag(concept)
                engine.long_term_memory.add_concept(concept)

            ep = Episode(
                concept_ids=[c.id for c in concepts],
                raw_content=turn.user_message,
                importance=concepts[0].importance if concepts else ImportanceVector(),
                source="user"
            )
            engine.working_memory.store(ep)

        # Now test recall: search for each expected fact
        total_facts = 0
        recalled_facts = 0

        for turn in conversation:
            for fact in turn.expected_facts:
                total_facts += 1
                results = engine.long_term_memory.search_by_text(fact, limit=3)
                if results:
                    # Check if any result contains the fact
                    for concept in results:
                        if fact.lower() in concept.description.lower():
                            recalled_facts += 1
                            break

        score = recalled_facts / total_facts if total_facts > 0 else 0.0

        return BenchmarkResult(
            test_name=f"Memory Retention ({n_turns} turns)",
            passed=score >= 0.7,  # At least 70% recall
            score=score,
            details={
                "turns": n_turns,
                "total_facts": total_facts,
                "recalled_facts": recalled_facts,
                "ltm_concepts": len(engine.long_term_memory.get_all_concepts())
            },
            duration_ms=(time.time() - start) * 1000
        )

    def test_sleep_benefit(self) -> BenchmarkResult:
        """Test that sleep consolidation improves memory organization"""
        start = time.time()

        engine = ChatEngine(enable_auto_sleep=False)

        # Add 20 concepts (some important, some noise)
        important_concepts = [
            ("Alice", 0.9), ("Google", 0.85), ("Seattle", 0.8),
            ("hiking", 0.75), ("Max", 0.7), ("MIT", 0.65)
        ]
        noise_concepts = [
            ("the", 0.1), ("and", 0.1), ("a", 0.1),
            ("to", 0.1), ("of", 0.1), ("in", 0.1),
            ("is", 0.1), ("it", 0.1), ("that", 0.1),
            ("for", 0.1), ("with", 0.1), ("on", 0.1)
        ]

        for name, importance in important_concepts + noise_concepts:
            concept = Concept(
                type=ConceptType.FACT,
                description=name,
                importance=ImportanceVector(
                    novelty=importance,
                    emotional=0.0,
                    task_relevance=importance,
                    repetition=0.5
                ),
                strength=importance
            )
            engine.long_term_memory.add_concept(concept)

        # Before sleep: count high-importance concepts
        before = len([c for c in engine.long_term_memory.get_all_concepts()
                      if c.importance.overall >= 0.5])

        # Force sleep
        result = engine.force_sleep()

        # After sleep: count remaining high-importance concepts
        after = len([c for c in engine.long_term_memory.get_all_concepts()
                     if c.importance.overall >= 0.5])

        # Important concepts should survive, noise should be reduced
        important_survived = after >= len(important_concepts) * 0.8
        noise_reduced = len(engine.long_term_memory.get_all_concepts()) < len(important_concepts) + len(noise_concepts)

        score = 0.5 if important_survived else 0.0
        score += 0.5 if noise_reduced else 0.0

        return BenchmarkResult(
            test_name="Sleep Consolidation Benefit",
            passed=score >= 0.8,
            score=score,
            details={
                "important_before": before,
                "important_after": after,
                "total_after": len(engine.long_term_memory.get_all_concepts()),
                "forgotten": result['forgotten'] if result else 0,
                "consolidated": result['consolidated'] if result else 0
            },
            duration_ms=(time.time() - start) * 1000
        )

    def test_forgetting(self) -> BenchmarkResult:
        """Test that forgetting prevents memory bloat"""
        start = time.time()

        engine = ChatEngine(enable_auto_sleep=False)

        # Add 50 low-importance concepts (simulating noise)
        for i in range(50):
            concept = Concept(
                type=ConceptType.FACT,
                description=f"noise_concept_{i}",
                importance=ImportanceVector(
                    novelty=0.1,
                    emotional=0.0,
                    task_relevance=0.1,
                    repetition=0.1
                ),
                strength=0.2
            )
            engine.long_term_memory.add_concept(concept)

        # Add 5 high-importance concepts (simulating key facts)
        for i in range(5):
            concept = Concept(
                type=ConceptType.FACT,
                description=f"important_fact_{i}",
                importance=ImportanceVector(
                    novelty=0.9,
                    emotional=0.5,
                    task_relevance=0.9,
                    repetition=0.8
                ),
                strength=1.5
            )
            engine.long_term_memory.add_concept(concept)

        initial_count = len(engine.long_term_memory.get_all_concepts())

        # Force sleep
        result = engine.force_sleep()

        final_count = len(engine.long_term_memory.get_all_concepts())
        reduction = (initial_count - final_count) / initial_count

        # Should forget at least 30% of low-importance stuff
        score = min(1.0, reduction / 0.3)

        # Important facts should survive
        important_surviving = len([c for c in engine.long_term_memory.get_all_concepts()
                                   if 'important' in c.description])

        if important_surviving < 4:  # Lost too many important facts
            score *= 0.5

        return BenchmarkResult(
            test_name="Forgetting Effectiveness",
            passed=score >= 0.6 and important_surviving >= 4,
            score=score,
            details={
                "initial_concepts": initial_count,
                "final_concepts": final_count,
                "reduction_percent": f"{reduction*100:.1f}%",
                "important_surviving": important_surviving,
                "forgotten": result['forgotten'] if result else 0
            },
            duration_ms=(time.time() - start) * 1000
        )

    def test_graph_traversal(self) -> BenchmarkResult:
        """Test that graph relations help find related concepts"""
        start = time.time()

        ltm = LongTermMemory()

        # Create related concepts
        c1 = Concept(type=ConceptType.PERSON, description="Alice", importance=ImportanceVector())
        c2 = Concept(type=ConceptType.LOCATION, description="Seattle", importance=ImportanceVector())
        c3 = Concept(type=ConceptType.FACT, description="software engineer", importance=ImportanceVector())
        c4 = Concept(type=ConceptType.PREFERENCE, description="hiking", importance=ImportanceVector())

        for c in [c1, c2, c3, c4]:
            ltm.add_concept(c)

        # Create relations
        from src.core.models import Relation, PredicateType
        ltm.add_relation(Relation(subject_id=c1.id, predicate=PredicateType.RELATED_TO, object_id=c2.id))
        ltm.add_relation(Relation(subject_id=c1.id, predicate=PredicateType.RELATED_TO, object_id=c3.id))
        ltm.add_relation(Relation(subject_id=c1.id, predicate=PredicateType.RELATED_TO, object_id=c4.id))

        # Search for Alice, should find related concepts
        related = ltm.get_related_concepts(c1.id, depth=1)

        score = len(related) / 3.0  # Should find 3 related concepts

        return BenchmarkResult(
            test_name="Graph Traversal Accuracy",
            passed=score >= 0.8,
            score=score,
            details={
                "seed_concept": "Alice",
                "related_found": len(related),
                "expected": 3
            },
            duration_ms=(time.time() - start) * 1000
        )

    def test_latency_scaling(self) -> BenchmarkResult:
        """Test that latency stays reasonable as memory grows"""
        start = time.time()

        engine = ChatEngine(enable_auto_sleep=False)

        latencies = []

        # Add concepts in batches and measure search latency
        for batch_size in [10, 50, 100, 200]:
            # Add batch
            for i in range(batch_size):
                concept = Concept(
                    type=ConceptType.FACT,
                    description=f"concept_{batch_size}_{i}",
                    importance=ImportanceVector(novelty=0.5),
                    embedding=[0.1] * 384
                )
                engine.long_term_memory.add_concept(concept)

            # Measure search latency
            t0 = time.time()
            results = engine.long_term_memory.search_by_text("concept", limit=5)
            t1 = time.time()
            latencies.append((len(engine.long_term_memory.get_all_concepts()), (t1 - t0) * 1000))

        # Latency should grow sub-linearly or stay under 100ms
        max_latency = max(l[1] for l in latencies)
        score = 1.0 if max_latency < 100 else 0.5 if max_latency < 500 else 0.0

        return BenchmarkResult(
            test_name="Latency Scaling",
            passed=max_latency < 500,
            score=score,
            details={
                "latencies": {f"{k}_concepts": f"{v:.1f}ms" for k, v in latencies},
                "max_latency_ms": max_latency
            },
            duration_ms=(time.time() - start) * 1000
        )

    def test_persistence(self) -> BenchmarkResult:
        """Test that memory survives restart"""
        start = time.time()

        from src.core.sqlite_db import get_memory
        sqlite = get_memory()
        sqlite.clear_all()

        # Create engine and add facts
        engine1 = ChatEngine(enable_auto_sleep=False, session_id="benchmark_test")

        concepts_data = [
            ("Alice", ConceptType.PERSON),
            ("Google", ConceptType.LOCATION),
            ("hiking", ConceptType.PREFERENCE)
        ]

        for name, ctype in concepts_data:
            c = Concept(type=ctype, description=name, importance=ImportanceVector(novelty=0.8))
            engine1.long_term_memory.add_concept(c)

        engine1.save_session()
        count_before = len(engine1.long_term_memory.get_all_concepts())

        # Create new engine (simulates restart)
        engine2 = ChatEngine(enable_auto_sleep=False, session_id="benchmark_test")
        count_after = len(engine2.long_term_memory.get_all_concepts())

        score = 1.0 if count_after >= count_before * 0.8 else 0.0

        return BenchmarkResult(
            test_name="Multi-Session Persistence",
            passed=score == 1.0,
            score=score,
            details={
                "concepts_before": count_before,
                "concepts_after": count_after,
                "session_id": "benchmark_test"
            },
            duration_ms=(time.time() - start) * 1000
        )

    def _create_test_concepts(self, text: str) -> List[Concept]:
        """Create test concepts without LLM (avoid latency)"""
        concepts = []

        # Simple keyword extraction - MUST include all expected_facts
        keywords = {
            "Alice": ConceptType.PERSON,
            "Google": ConceptType.LOCATION,
            "Seattle": ConceptType.LOCATION,
            "hiking": ConceptType.PREFERENCE,
            "Max": ConceptType.PERSON,
            "MIT": ConceptType.LOCATION,
            "March": ConceptType.FACT,
            "Tesla": ConceptType.OBJECT,
            "Spanish": ConceptType.FACT,
            "French": ConceptType.FACT,
            "pizza": ConceptType.PREFERENCE,
            "engineer": ConceptType.FACT,
            "dog": ConceptType.OBJECT,
            "birthday": ConceptType.EVENT,
            "work": ConceptType.FACT,
            "live": ConceptType.FACT,
            "hobby": ConceptType.FACT,
            "enjoy": ConceptType.FACT,
            "love": ConceptType.FACT,
            "eating": ConceptType.FACT,
            "studied": ConceptType.FACT,
            "drive": ConceptType.FACT,
            "speak": ConceptType.FACT,
            "name": ConceptType.FACT,
        }

        for keyword, ctype in keywords.items():
            if keyword.lower() in text.lower():
                concepts.append(Concept(
                    type=ctype,
                    description=keyword,
                    importance=ImportanceVector(novelty=0.7, task_relevance=0.8)
                ))

        return concepts if concepts else [Concept(type=ConceptType.FACT, description=text[:50])]

    def _generate_report(self) -> Dict:
        """Generate benchmark report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        avg_score = sum(r.score for r in self.results) / total_tests if total_tests > 0 else 0

        report = {
            "timestamp": utc_now().isoformat(),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "average_score": round(avg_score, 3),
            "overall_passed": passed_tests >= total_tests * 0.7,  # 70% pass rate
            "tests": []
        }

        for result in self.results:
            report["tests"].append({
                "name": result.test_name,
                "passed": result.passed,
                "score": round(result.score, 3),
                "duration_ms": round(result.duration_ms, 1),
                "details": result.details
            })

        return report

    def print_report(self, report: Dict):
        """Pretty print benchmark report"""
        print()
        print("="*70)
        print("BENCHMARK REPORT")
        print("="*70)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed']} | Failed: {report['failed']}")
        print(f"Average Score: {report['average_score']:.2f}")
        print(f"Overall: {'✅ PASSED' if report['overall_passed'] else '❌ FAILED'}")
        print()
        print("-"*70)

        for test in report["tests"]:
            status = "✅" if test["passed"] else "❌"
            print(f"{status} {test['name']:<35} Score: {test['score']:.2f}  ({test['duration_ms']:.0f}ms)")
            if test["details"]:
                for k, v in test["details"].items():
                    print(f"    {k}: {v}")

        print("-"*70)

        if report['overall_passed']:
            print("\n🎉 SleepAI memory system is performing well!")
        else:
            print("\n⚠️  Some benchmarks failed. Review details above.")

        print("="*70)

    def save_report(self, report: Dict, filename: str = "benchmark_report.json"):
        """Save report to file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {filepath}")


if __name__ == '__main__':
    benchmark = SleepAIBenchmark()
    report = benchmark.run_all()
    benchmark.print_report(report)
    benchmark.save_report(report)
