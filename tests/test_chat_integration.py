"""
Phase 3 Integration Tests
Tests the full conversational memory pipeline
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import re
from src.core.models import Concept, ConceptType, ImportanceVector
from src.chat.engine import ChatEngine


class _FailLLM:
    def _chat(self, prompt: str, num_predict: int = 512) -> str:
        raise RuntimeError("force fallback response")


class _PreferenceEncoder:
    def extract(self, text: str):
        match = re.search(r"(?:i prefer)\s+([^.!?]+)", text, flags=re.IGNORECASE)
        if not match:
            return []

        pref = re.sub(
            r"\s+(?:right now|for now|currently|at the moment|now)\s*$",
            "",
            match.group(1).strip(),
            flags=re.IGNORECASE,
        ).strip(" .,!?:;")

        return [
            Concept(
                type=ConceptType.PREFERENCE,
                description=f"I prefer {pref}",
                embedding=[0.1] * 384,
                importance=ImportanceVector(novelty=0.8, task_relevance=0.8, repetition=0.3),
                salience_score=0.8,
                grasp_score=0.8,
            )
        ]

    def _get_embedding(self, text: str):
        return [0.1] * 384


class _EmptyEncoder:
    def extract(self, text: str):
        return []

    def _get_embedding(self, text: str):
        return [0.1] * 384


class TestChatEngineIntegration(unittest.TestCase):
    """Integration tests for ChatEngine"""

    def setUp(self):
        self.engine = ChatEngine(enable_auto_sleep=False)

    def test_conversation_remembers_name(self):
        """Test that SleepAI remembers user's name across messages"""
        # User introduces themselves
        response1, meta1 = self.engine.chat("Hello, my name is John")

        # Working memory stores user + assistant response as episodes
        self.assertEqual(self.engine.working_memory.size(), 2)

        # LTM should have concept
        concepts = self.engine.long_term_memory.get_all_concepts(include_suppressed=False)
        self.assertGreater(len(concepts), 0)

        # User asks for their name
        response2, meta2 = self.engine.chat("What is my name?")

        # Should retrieve memories
        self.assertGreater(meta2['memories_retrieved'], 0)

        # Response should contain the name
        self.assertIn("John", response2)

    def test_conversation_remembers_preferences(self):
        """Test preference memory across conversation"""
        self.engine.chat("I love eating pizza")
        self.engine.chat("My favorite color is blue")

        # Ask about preferences
        response, meta = self.engine.chat("What do I like?")

        # Should recall preferences
        self.assertGreater(meta['memories_retrieved'], 0)

        # Response should mention pizza or blue
        has_preference = "pizza" in response.lower() or "blue" in response.lower()
        self.assertTrue(has_preference, f"Response should mention preference: {response}")

    def test_working_memory_capacity(self):
        """Test that working memory respects capacity"""
        # Send more messages than capacity
        for i in range(10):
            self.engine.chat(f"Message number {i}")

        # Should not exceed capacity
        self.assertLessEqual(
            self.engine.working_memory.size(),
            self.engine.working_memory.capacity
        )

    def test_conversation_metadata(self):
        """Test that metadata is populated correctly"""
        response, meta = self.engine.chat("Tell me about yourself")

        self.assertIn('user_concepts', meta)
        self.assertIn('response_concepts', meta)
        self.assertIn('memories_retrieved', meta)
        self.assertIn('latency_ms', meta)
        self.assertIn('sleep_triggered', meta)

        # Latency should be reasonable
        self.assertGreater(meta['latency_ms'], 0)

    def test_force_sleep(self):
        """Test manual sleep trigger"""
        # Add some content
        for i in range(5):
            self.engine.chat(f"Fact {i}: Something important to remember")

        initial_wm_size = self.engine.working_memory.size()

        # Force sleep
        result = self.engine.force_sleep()

        # Should return stats or None
        if result:
            self.assertIn('consolidated', result)
            self.assertIn('forgotten', result)
            # WM should be cleared after sleep
            self.assertEqual(self.engine.working_memory.size(), 0)

    def test_memory_report(self):
        """Test memory report generation"""
        self.engine.chat("Test message")

        report = self.engine.get_memory_report()

        self.assertIn('conversation_duration_minutes', report)
        self.assertIn('messages_exchanged', report)
        self.assertIn('working_memory', report)
        self.assertIn('long_term_memory', report)
        self.assertIn('sleep_readiness', report)

    def test_multi_turn_conversation(self):
        """Test a realistic multi-turn conversation"""
        conversation = [
            ("Hello, I'm Sarah", "Should acknowledge Sarah"),
            ("I work as a software engineer", "Should remember profession"),
            ("I live in San Francisco", "Should remember location"),
            ("What is my name?", "Should say Sarah"),
            ("Where do I live?", "Should mention San Francisco"),
            ("What do I do?", "Should mention software engineer"),
        ]

        correct_recalls = 0
        for user_msg, expected in conversation:
            response, meta = self.engine.chat(user_msg)

            # Check recall questions
            if "name" in user_msg.lower() and "Sarah" in response:
                correct_recalls += 1
            if "live" in user_msg.lower() and "Francisco" in response:
                correct_recalls += 1
            if "do" in user_msg.lower() and "engineer" in response:
                correct_recalls += 1

        # Should recall at least 2 out of 3
        self.assertGreaterEqual(correct_recalls, 2,
            f"Expected at least 2 correct recalls, got {correct_recalls}")

    def test_empty_message(self):
        """Test empty message handling"""
        response, meta = self.engine.chat("")
        # Should not crash
        self.assertIsInstance(response, str)

    def test_special_characters(self):
        """Test special characters in conversation"""
        response, meta = self.engine.chat("Hello! My name is José and I like 🍕")
        # Should not crash
        self.assertIsInstance(response, str)

    def test_memory_persistence_after_multiple_messages(self):
        """Test that LTM grows with conversation"""
        initial_concepts = len(self.engine.long_term_memory.get_all_concepts())

        for i in range(3):
            self.engine.chat(f"Important fact {i}: The sky is blue")

        final_concepts = len(self.engine.long_term_memory.get_all_concepts())
        self.assertGreater(final_concepts, initial_concepts)

    def test_preference_update_reads_like_a_person(self):
        """Test that updated preferences are phrased naturally."""
        engine = ChatEngine(
            llm=_FailLLM(),
            encoder=_PreferenceEncoder(),
            enable_auto_sleep=False,
            session_id="human_style_fallback",
        )

        engine.chat("I prefer morning meetings.")
        engine.chat("Actually no, I prefer evening meetings now.")
        response, meta = engine.chat("What do I prefer?")

        self.assertIn("evening", response.lower())
        self.assertTrue(
            "updated" in response.lower() or "first mentioned" in response.lower(),
            f"Expected a human-style update phrase, got: {response}",
        )

    def test_missing_profile_reply_sounds_human(self):
        """Test that uncertainty replies still sound conversational."""
        engine = ChatEngine(
            llm=_FailLLM(),
            encoder=_EmptyEncoder(),
            enable_auto_sleep=False,
            session_id="human_style_unknown",
        )

        response, meta = engine.chat("Where do I live?")

        self.assertIn("tell me once", response.lower())
        self.assertIn("location", response.lower())


class TestChatEngineStress(unittest.TestCase):
    """Stress tests for ChatEngine"""

    def test_rapid_fire_messages(self):
        """Test handling many rapid messages"""
        engine = ChatEngine(enable_auto_sleep=False)

        for i in range(20):
            response, meta = engine.chat(f"Quick message {i}")
            self.assertIsInstance(response, str)

        # Should have some concepts stored
        self.assertGreater(len(engine.long_term_memory.get_all_concepts()), 0)

    def test_long_message(self):
        """Test very long user message"""
        engine = ChatEngine(enable_auto_sleep=False)

        long_message = "I enjoy " + ", ".join([f"activity {i}" for i in range(50)])
        response, meta = engine.chat(long_message)

        self.assertIsInstance(response, str)
        self.assertGreater(meta['user_concepts'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
