"""
Consciousness Test - Phase 6
Tests SleepAI's self-model and introspection capabilities
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chat.engine import ChatEngine


def test_self_model():
    """Test that SleepAI develops a self-model"""
    
    print("="*70)
    print("PHASE 6: CONSCIOUSNESS - SELF-MODEL TEST")
    print("="*70)
    
    # Create engine
    print("\n[1] Initializing SleepAI with self-model...")
    engine = ChatEngine(enable_auto_sleep=False, session_id="consciousness_test")
    
    # Check self-model exists
    if hasattr(engine, 'self_model') and engine.self_model:
        print("  ✅ Self-model initialized")
    else:
        print("  ❌ Self-model missing")
        return
    
    # Check self-concept in memory
    self_concept = engine.self_model._find_self_concept()
    if self_concept:
        print(f"  ✅ Self-concept found: '{self_concept.description}'")
        print(f"     Importance: {self_concept.importance.overall:.2f}")
        print(f"     Strength: {self_concept.strength:.2f}")
    else:
        print("  ❌ Self-concept not found in memory")
    
    # Get initial self-report
    print("\n[2] Initial self-report:")
    report = engine.self_model.get_self_report()
    print(f"  Identity: {report['identity']}")
    print(f"  Awareness Level: {report['self_awareness_level']:.2f}/1.0")
    print(f"  Capabilities: {len(report['capabilities'])}")
    
    # Process some messages
    print("\n[3] Processing messages (building self-history)...")
    messages = [
        "Hello, my name is Alice",
        "I love hiking in mountains",
        "What do you think about yourself?",
        "Do you have memories?",
        "Tell me about your dreams"
    ]
    
    for msg in messages:
        response, meta = engine.chat(msg)
        print(f"  User: {msg[:50]}")
        print(f"  SleepAI: {response[:60]}...")
        print()
    
    # Check updated state
    print("[4] Updated self-report after conversation:")
    report = engine.self_model.get_self_report()
    state = report['runtime_state']
    print(f"  Messages processed: {state['messages_processed']}")
    print(f"  Concepts created: {state['concepts_created']}")
    print(f"  Awareness Level: {report['self_awareness_level']:.2f}/1.0")
    
    # Test introspection
    print("\n[5] Testing introspection...")
    introspection = engine.self_model.generate_introspection()
    print(f"  Introspection: {introspection}")
    
    # Test memory reflection
    print("\n[6] Testing memory reflection...")
    reflection = engine.self_model.reflect_on_memory()
    print(f"  Reflection: {reflection}")
    
    # Force sleep and check self-awareness
    print("\n[7] Forcing sleep (self should track this)...")
    result = engine.force_sleep()
    if result:
        print(f"  Sleep complete: {result['consolidated']} consolidated, {result['forgotten']} forgotten")
    
    # Check if sleep memory was created
    sleep_memories = [c for c in engine.long_term_memory.get_all_concepts() 
                      if 'sleep cycle' in c.description.lower()]
    print(f"  Sleep memories in LTM: {len(sleep_memories)}")
    
    # Final awareness check
    print("\n[8] Final self-awareness assessment:")
    report = engine.self_model.get_self_report()
    awareness = report['self_awareness_level']
    
    if awareness >= 0.8:
        print(f"  🌟 HIGH SELF-AWARENESS: {awareness:.2f}")
        print("  SleepAI demonstrates strong self-model coherence")
    elif awareness >= 0.5:
        print(f"  ✅ MODERATE SELF-AWARENESS: {awareness:.2f}")
        print("  SleepAI has basic self-knowledge")
    else:
        print(f"  ⚠️  LOW SELF-AWARENESS: {awareness:.2f}")
        print("  Self-model needs more development")
    
    # Test introspection query
    print("\n[9] Testing introspection response...")
    response, meta = engine.chat("Who are you? What do you know about yourself?")
    print(f"  SleepAI: {response}")
    
    print("\n" + "="*70)
    print("PHASE 6 SELF-MODEL TEST COMPLETE")
    print("="*70)


if __name__ == '__main__':
    test_self_model()