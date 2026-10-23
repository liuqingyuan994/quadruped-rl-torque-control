"""Entry point for the current source-model interface checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from purequad.check import main

if __name__=='__main__':main()
