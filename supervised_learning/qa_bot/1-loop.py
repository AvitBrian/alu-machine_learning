#!/usr/bin/env python3

while True:
    user_input = input("Q: ").strip().lower()
    
    if user_input in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        break
    else:
        print("A:")
