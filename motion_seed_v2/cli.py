#!/usr/bin/env python3
"""
Motion Calendar Seed - Command Line Interface (v2)

PBAI Functional Stack:
- Polarity → Existence Gate → Heat → Righteousness → Order → Movement
- Agency randomizes; Intelligence chooses.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seed import MotionSeed
from core import MovementMode


def print_help():
    print("""
Motion Calendar Seed v2 - Commands:
===================================

INPUT:
  <text>              Perceive text input
  
RETRIEVAL:
  /query <question>   Ask a question (e.g., /query What is your name?)
  /recall <cue>       Recall content associated with cue
  /assoc <term>       Show raw associations for term
  
THOUGHT:
  /think              Generate a thought (random walk)
  /think <seed>       Generate thought starting from domain hint
  /brainstorm         Generate multiple thoughts
  /choose <d1,d2,...> Intelligence chooses specific domain path
  
MODE:
  /agency             Switch to Agency mode (random only)
  /intelligence       Switch to Intelligence mode (can also choose)
  /mode               Show current mode
  
STATUS:
  /status             Show brief status
  /describe           Show detailed description
  /domains            List all kernel domains
  /identity           Show identity domains
  /rules              Show learned order rules
  /transforms         Show available transforms
  /memory             Show memory statistics
  /dark               Show dark matter pool
  
CONTROL:
  /save               Save state to disk
  /temp <value>       Set randomizer temperature (0.1=focused, 10=random)
  /help               Show this help
  /quit               Save and exit
""")


def main():
    print("Motion Calendar Seed v2")
    print("=" * 40)
    print("PBAI Functional Stack Implementation")
    print()
    print("Polarity → Existence Gate → Heat")
    print("  → Righteousness → Order → Movement")
    print()
    print("Agency randomizes; Intelligence chooses.")
    print()
    print("Type /help for commands")
    print()
    
    # Parse arguments
    data_path = "./seed_data"
    mode = MovementMode.AGENCY
    
    for arg in sys.argv[1:]:
        if arg.startswith("--data="):
            data_path = arg.split("=")[1]
        elif arg == "--intelligence":
            mode = MovementMode.INTELLIGENCE
        elif arg == "--agency":
            mode = MovementMode.AGENCY
        elif not arg.startswith("-"):
            data_path = arg
    
    # Create seed
    seed = MotionSeed(data_path, mode=mode)
    
    print(f"Data path: {data_path}")
    print(f"Mode: {seed.mode.value.upper()}")
    print(f"Kernel particles: {seed.kernel.total_particles}")
    print(f"Identity particles: {seed.identity.total_integrated}")
    print(f"Total heat: {float(seed.kernel.total_heat):.2f}")
    print()
    
    while True:
        try:
            prompt = f"seed[{seed.mode.value[0]}]> "  # [a] for agency, [i] for intelligence
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else None
                
                if cmd == "/quit" or cmd == "/exit":
                    seed.save()
                    print("Saved. Goodbye.")
                    break
                
                elif cmd == "/help":
                    print_help()
                
                elif cmd == "/save":
                    seed.save()
                    print("State saved.")
                
                # Mode commands
                elif cmd == "/agency":
                    seed.set_mode(MovementMode.AGENCY)
                    print("Switched to AGENCY mode (random only)")
                
                elif cmd == "/intelligence":
                    seed.set_mode(MovementMode.INTELLIGENCE)
                    print("Switched to INTELLIGENCE mode (can also choose)")
                
                elif cmd == "/mode":
                    print(f"Current mode: {seed.mode.value.upper()}")
                
                # Status commands
                elif cmd == "/status":
                    status = seed.get_status()
                    print(f"Mode: {status['mode'].upper()}")
                    print(f"Kernel: {status['kernel']['total_particles']} particles, {status['kernel']['total_domains']} domains")
                    print(f"Identity: {status['identity']['total_integrated']} particles, {status['identity']['total_domains']} domains")
                    print(f"Total heat: {status['kernel']['total_heat']:.2f}")
                    print(f"Gate pass rate: {status['gate']['pass_rate']:.1%}")
                
                elif cmd == "/describe":
                    print(seed.describe())
                
                elif cmd == "/domains":
                    print("Kernel Domains:")
                    for d_id, domain in seed.kernel.domains.items():
                        print(f"  {d_id}: {domain.count} particles, heat={domain.total_heat:.2f}")
                
                elif cmd == "/identity":
                    print("Identity Domains:")
                    for d_id, domain in seed.identity.domains.items():
                        print(f"  {d_id}: {domain.count} particles")
                        if domain.correlations:
                            corrs = sorted(domain.correlations.items(), key=lambda x: -x[1])[:3]
                            for other, strength in corrs:
                                print(f"    → {other}: {strength:.2f}")
                
                elif cmd == "/rules":
                    print("Learned Rules:")
                    for rule in seed.identity.rules[:20]:
                        print(f"  {rule['antecedent']} → {rule['consequent']} (strength: {rule['strength']:.2f})")
                    if len(seed.identity.rules) > 20:
                        print(f"  ... and {len(seed.identity.rules) - 20} more")
                
                elif cmd == "/transforms":
                    print("Available Transforms:")
                    for t in seed.kernel.transforms.all():
                        print(f"  {t.name}: stability={t.stability:.2f}, uses={t.use_count}")
                
                elif cmd == "/memory":
                    stats = seed.memory.get_stats()
                    print("Memory Statistics:")
                    print(f"  Associations: {stats['total_associations']}")
                    print(f"  Indexed content: {stats['indexed_content']}")
                    print(f"  Domains tracked: {stats['domains_tracked']}")
                    print(f"  Retrievals: {stats['total_retrievals']}")
                
                elif cmd == "/dark":
                    print(f"Dark Matter Pool: {seed.kernel.total_dark} particles")
                    for dark_id, dark in list(seed.kernel.dark_pool.items())[:10]:
                        print(f"  {dark_id[:8]}... polarity={dark.polarity.sign:+d}, iterations={dark.iterations_attempted}")
                
                # Thought commands
                elif cmd == "/think":
                    thought = seed.think(seed=arg)
                    if thought.activations:
                        print(f"Thought: {thought.to_string()}")
                        print(f"  Domains: {' → '.join(thought.domains_visited)}")
                        print(f"  Coherence: {thought.coherence:.2f}")
                        print(f"  Heat: {thought.total_heat:.2f}")
                        print(f"  Mode: {thought.mode.value}" + (" (chosen)" if thought.chosen else ""))
                    else:
                        print("(no thought yet - need more experience)")
                
                elif cmd == "/brainstorm":
                    thoughts = seed.brainstorm(count=5)
                    for i, thought in enumerate(thoughts, 1):
                        if thought.activations:
                            print(f"{i}. {thought.to_string()} (coherence: {thought.coherence:.2f})")
                
                elif cmd == "/choose":
                    if seed.mode != MovementMode.INTELLIGENCE:
                        print("Must be in INTELLIGENCE mode to choose. Use /intelligence first.")
                    elif arg:
                        path = [d.strip() for d in arg.split(",")]
                        try:
                            thought = seed.choose_thought(path)
                            print(f"Chosen thought: {thought.to_string()}")
                            print(f"  Domains: {' → '.join(thought.domains_visited)}")
                            print(f"  Coherence: {thought.coherence:.2f}")
                        except Exception as e:
                            print(f"Error: {e}")
                    else:
                        print("Usage: /choose domain1,domain2,domain3")
                
                # Retrieval commands
                elif cmd == "/query":
                    if arg:
                        answer = seed.query(arg)
                        print(f"Answer: {answer}")
                    else:
                        print("Usage: /query <question>")
                
                elif cmd == "/recall":
                    if arg:
                        recalled = seed.recall(arg)
                        if recalled:
                            print(f"Recalled for '{arg}':")
                            for item in recalled[:10]:
                                print(f"  - {item}")
                        else:
                            print(f"(nothing recalled for '{arg}')")
                    else:
                        print("Usage: /recall <cue>")
                
                elif cmd == "/assoc":
                    if arg:
                        assocs = seed.associations(arg)
                        if assocs:
                            print(f"Associations for '{arg}':")
                            for item, strength in assocs[:15]:
                                display = item.replace("content:", "")
                                print(f"  {strength:.2f} → {display}")
                        else:
                            print(f"(no associations for '{arg}')")
                    else:
                        print("Usage: /assoc <term>")
                
                # Control commands
                elif cmd == "/temp":
                    if arg:
                        try:
                            temp = float(arg)
                            seed.randomizer.temperature = temp
                            print(f"Temperature set to {temp}")
                        except ValueError:
                            print("Invalid temperature value")
                    else:
                        print(f"Current temperature: {seed.randomizer.temperature}")
                
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type /help for available commands")
            
            else:
                # Perceive input
                results = seed.perceive(user_input)
                
                # Summary
                integrated = sum(1 for _, passed, _ in results if passed)
                print(f"Experienced {len(results)} units, integrated {integrated}")
                
                # Details for small inputs
                if len(results) <= 15:
                    for content, passed, reason in results:
                        symbol = "✓" if passed else "○"
                        print(f"  {symbol} '{content}' - {reason}")
        
        except KeyboardInterrupt:
            print("\nInterrupted. Saving...")
            seed.save()
            print("Saved. Goodbye.")
            break
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
