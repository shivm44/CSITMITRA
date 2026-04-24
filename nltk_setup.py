"""
Run this ONCE after deployment to download NLTK data.
On Render / Railway: add this as a Build Command, or run it in the shell.

Usage:
    python nltk_setup.py
"""
import nltk
import sys

packages = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]
for pkg in packages:
    print(f"Downloading {pkg}...", end=" ")
    try:
        nltk.download(pkg, quiet=False)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")

print("\nDone. NLTK data ready.")
print("If downloads failed (network blocked), the chatbot will use the built-in fallback tokeniser.")
