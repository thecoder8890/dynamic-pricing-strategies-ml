#!/usr/bin/env python3
"""
Quick Start Script for Dynamic Pricing System
Run this script to set up and test the entire system quickly.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and provide feedback"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_file_exists(filename):
    """Check if a file exists"""
    return Path(filename).exists()

def main():
    """Main setup and test function"""
    print("🧝‍♂️ Welcome to the Dynamic Pricing System Quick Start!")
    print("=" * 60)
    
    print("\n📋 System Status Check:")
    
    # Check required files
    required_files = [
        "requirements.txt",
        "generate_dataset.py", 
        "pricing_system.py",
        "dynamic_pricing_elf_guide.ipynb"
    ]
    
    missing_files = []
    for file in required_files:
        if check_file_exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        print("Please ensure all required files are present.")
        return
    
    # Check if dataset exists
    dataset_exists = check_file_exists("elves_marketplace_data.csv")
    
    if not dataset_exists:
        print(f"\n📊 Generating sample dataset...")
        if not run_command("python generate_dataset.py", "Dataset generation"):
            return
    else:
        print(f"✅ Dataset already exists")
    
    # Test the pricing system
    print(f"\n🧪 Testing pricing system components...")
    
    tests = [
        ("python pricing_system.py", "Core pricing system"),
        ("python test_examples.py", "README examples validation"), 
        ("python test_integration.py", "System integration"),
        ("python test_comprehensive.py", "Comprehensive test suite"),
    ]
    
    all_tests_passed = True
    for command, description in tests:
        if not run_command(command, description):
            all_tests_passed = False
    
    if not all_tests_passed:
        print(f"\n⚠️ Some tests failed. Please check the output above.")
        return
    
    # Run demonstrations
    print(f"\n🎭 Running system demonstrations...")
    
    demos = [
        ("python demo_pricing_system.py", "Pricing strategy demo"),
        ("python pricing_api.py", "API interface demo")
    ]
    
    for command, description in demos:
        run_command(command, description)
    
    # Generate documentation diagrams
    print(f"\n📊 Generating flow diagrams...")
    run_command("python generate_flow_diagrams.py", "Flow diagram generation")
    
    # Final status
    print(f"\n🎉 Quick Start Completed!")
    print(f"=" * 60)
    
    print(f"\n📚 What's Available:")
    print(f"  🧙‍♂️ Interactive Tutorial: jupyter notebook dynamic_pricing_elf_guide.ipynb")
    print(f"  🧪 Pricing Functions: from pricing_system import rule_based_pricing, predict_optimal_price")
    print(f"  🚀 API Interface: python pricing_api.py")
    print(f"  🎭 Demo Script: python demo_pricing_system.py")
    
    print(f"\n📊 Generated Files:")
    generated_files = [
        "elves_marketplace_data.csv",
        "pricing_demo_results.png", 
        "docs/system_architecture_diagram.png",
        "docs/rule_based_flow_diagram.png",
        "docs/ml_training_flow_diagram.png",
        "docs/price_optimization_flow.png"
    ]
    
    for file in generated_files:
        if check_file_exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️ {file} (may not have been generated)")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Explore the interactive notebook: jupyter notebook dynamic_pricing_elf_guide.ipynb")
    print(f"  2. Try the pricing functions in Python: python -c \"from pricing_system import *\"")
    print(f"  3. Read the documentation files (README.md, TECHNICAL_DOCUMENTATION.md)")
    print(f"  4. Customize the algorithms for your specific use case")
    
    print(f"\n🧝‍♂️ Happy pricing! May your algorithms be optimal and your revenues abundant! ✨")

if __name__ == "__main__":
    main()