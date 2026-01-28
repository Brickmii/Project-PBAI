#!/usr/bin/env python3
"""
PBAI Maternal Chat - Watch PBAI learn to speak

This is a terminal interface to the MaternalDriver.
Watch PBAI grow from infant to adult.

Usage:
    python maternal_chat.py
    python maternal_chat.py --no-mother  # No Qwen, pure thermal
    python maternal_chat.py --fresh      # Start with empty manifold
"""

import sys
import os
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_chat(use_mother: bool = True, fresh: bool = False, growth_path: str = None,
             model_size: str = "0.5B"):
    """Run the maternal chat interface."""
    from core import get_pbai_manifold, reset_pbai_manifold
    from core.node_constants import get_growth_path
    from maternal_driver import MaternalDriver
    
    # Resolve growth path
    if not growth_path:
        growth_path = get_growth_path("growth_map.json")
    
    # Resolve model
    model_map = {
        "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
        "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
        "3B": "Qwen/Qwen2.5-3B-Instruct",
    }
    qwen_model = model_map.get(model_size, model_map["0.5B"])
    
    # Handle fresh start
    if fresh and os.path.exists(growth_path):
        os.remove(growth_path)
        reset_pbai_manifold()
        print("🔄 Starting fresh (deleted existing growth map)")
    
    # Load manifold
    print("Loading PBAI mind...")
    manifold = get_pbai_manifold(growth_path)
    print(f"✓ Loaded {len(manifold.nodes)} nodes")
    
    # Create driver
    driver = MaternalDriver(manifold, qwen_model=qwen_model, use_qwen=use_mother)
    
    # Print header
    print()
    print("═" * 60)
    print("  PBAI MATERNAL TRAINING")
    print("  Watch PBAI learn to speak")
    print("═" * 60)
    print()
    
    stats = driver.get_stats()
    print(f"Stage: {stats['development_stage'].upper()}")
    print(f"Words: {stats['vocabulary_size']}")
    print(f"Avg Heat: {stats['avg_word_heat']:.2f}")
    print(f"Mother (Qwen): {'ENABLED' if use_mother else 'DISABLED'}")
    if use_mother:
        print(f"Model: {model_size}")
    print()
    print("Commands:")
    print("  /stats     - Show PBAI stats")
    print("  /words     - Show hottest words")
    print("  /history   - Show conversation history")
    print("  /mother    - Toggle mother (Qwen)")
    print("  /save      - Save growth map")
    print("  /quit      - Exit")
    print()
    print("─" * 60)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input[1:].lower().split()[0]
            
            if cmd == "quit" or cmd == "exit":
                # Save before exit
                manifold.save_growth_map(growth_path)
                print(f"Saved to {growth_path}")
                print("Goodbye!")
                break
            
            elif cmd == "stats":
                print()
                print(driver.introspect())
                print()
            
            elif cmd == "words":
                print()
                hot = driver.get_hottest_words(15)
                print("Hottest words:")
                for word, heat in hot:
                    bar = "█" * int(heat / 0.5)
                    print(f"  {word:15} {heat:6.2f} {bar}")
                print()
            
            elif cmd == "history":
                print()
                print("Conversation history:")
                for turn in driver.history[-10:]:
                    print(f"  [User] {turn.user_input[:50]}...")
                    print(f"  [PBAI] {turn.pbai_response[:50]}...")
                    print()
                print()
            
            elif cmd == "mother":
                driver.use_qwen = not driver.use_qwen
                if driver.use_qwen:
                    print("Mother: ENABLED")
                else:
                    print("Mother: DISABLED (pure thermal mode)")
            
            elif cmd == "save":
                manifold.save_growth_map(growth_path)
                print(f"Saved to {growth_path}")
            
            else:
                print(f"Unknown command: {cmd}")
        
        else:
            # Process input
            stats_before = driver.get_stats()
            
            response = driver.process_input(user_input)
            
            stats_after = driver.get_stats()
            
            # Show response with stage indicator
            stage = stats_after['development_stage']
            stage_emoji = {
                'infant': '👶',
                'child': '🧒',
                'adolescent': '🧑',
                'adult': '🧠'
            }.get(stage, '?')
            
            # Show if mother was consulted
            consulted = stats_after['qwen_consultations'] > stats_before['qwen_consultations']
            mother_indicator = " [asked mom]" if consulted else ""
            
            print()
            print(f"{stage_emoji} PBAI: {response}{mother_indicator}")
            print()
            
            # Show learning if new words
            new_words = stats_after['vocabulary_size'] - stats_before['vocabulary_size']
            if new_words > 0:
                print(f"   (learned {new_words} new words)")
                print()
            
            # Auto-save periodically
            if driver.total_turns % 10 == 0:
                manifold.save_growth_map(growth_path)


def main():
    parser = argparse.ArgumentParser(description="PBAI Maternal Training Chat")
    parser.add_argument("--no-mother", action="store_true", help="Disable mother (Qwen)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore existing)")
    parser.add_argument("--growth-path", type=str, help="Path to growth map")
    parser.add_argument("--model", type=str, default="0.5B", 
                       choices=["0.5B", "1.5B", "3B"],
                       help="Qwen model size (default: 0.5B for Pi)")
    args = parser.parse_args()
    
    run_chat(
        use_mother=not args.no_mother,
        fresh=args.fresh,
        growth_path=args.growth_path,
        model_size=args.model
    )


if __name__ == "__main__":
    main()
