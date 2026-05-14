"""
Multi-Agent Sync Test
Demonstrates two SleepAI agents sharing memories
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chat.engine import ChatEngine
from src.multi_agent.sync import AgentSwarm


def test_multi_agent_sync():
    """Test two agents sharing experiences"""
    
    print("="*70)
    print("MULTI-AGENT MEMORY SYNC TEST")
    print("="*70)
    
    # Create swarm
    swarm = AgentSwarm()
    
    # Create two agents
    print("\n[1] Creating Agent Alpha (user assistant)...")
    alpha = ChatEngine(enable_auto_sleep=False, session_id="alpha")
    swarm.add_agent("alpha", "Agent Alpha", alpha)
    
    print("\n[2] Creating Agent Beta (research assistant)...")
    beta = ChatEngine(enable_auto_sleep=False, session_id="beta")
    swarm.add_agent("beta", "Agent Beta", beta)
    
    # Agent Alpha has a conversation
    print("\n[3] Agent Alpha talking with user...")
    alpha.chat("My name is Alice and I love hiking in mountains")
    alpha.chat("I work at Google as a software engineer")
    
    print(f"\n  Alpha LTM: {len(alpha.long_term_memory.get_all_concepts())} concepts")
    
    # Agent Beta has a different conversation
    print("\n[4] Agent Beta talking with user...")
    beta.chat("My name is Bob and I enjoy coding in Python")
    beta.chat("I studied at MIT and love AI research")
    
    print(f"\n  Beta LTM: {len(beta.long_term_memory.get_all_concepts())} concepts")
    
    # Agents sleep individually
    print("\n[5] Both agents sleeping...")
    alpha.force_sleep()
    beta.force_sleep()
    
    # Collective sleep cycle (sync dreams)
    print("\n[6] Collective sleep cycle (syncing dreams)...")
    collective = swarm.collective_sleep()
    
    print(f"\n  Dreams shared: {collective['total_dreams_shared']}")
    
    # Share experience directly
    print("\n[7] Agent Alpha shares hiking experience with Beta...")
    swarm.share_experience(
        source_agent_id="alpha",
        experience="I discovered a new hiking trail in the Rocky Mountains",
        target_agent_ids=["beta"]
    )
    
    # Check what Beta learned
    print(f"\n  Beta LTM after sync: {len(beta.long_term_memory.get_all_concepts())} concepts")
    
    # Swarm report
    print("\n[8] Swarm Report:")
    report = swarm.get_swarm_report()
    
    for agent_id, info in report['agents'].items():
        print(f"\n  Agent: {info['name']} ({agent_id})")
        print(f"    LTM Concepts: {info['ltm_concepts']}")
        print(f"    Messages: {info['messages']}")
        print(f"    Sleeps: {info['sleeps']}")
    
    print("\n" + "="*70)
    print("✅ Multi-Agent Sync Test Complete!")
    print("="*70)


if __name__ == '__main__':
    test_multi_agent_sync()