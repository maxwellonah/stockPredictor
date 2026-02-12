# Simple test script to verify Python environment and basic operations

def main():
    print("=== Python Environment Test ===\n")
    
    # 1. Basic Python operations
    print("1. Testing basic Python operations...")
    try:
        x = 2 + 2
        print(f"  2 + 2 = {x}")
        print("  ✅ Basic Python operations work")
    except Exception as e:
        print(f"  ❌ Error in basic operations: {e}")
    
    # 2. File operations
    print("\n2. Testing file operations...")
    test_file = "test_file.txt"
    try:
        with open(test_file, "w") as f:
            f.write("test")
        print(f"  ✅ Successfully wrote to {test_file}")
        os.remove(test_file)
        print(f"  ✅ Successfully deleted {test_file}")
    except Exception as e:
        print(f"  ❌ File operation failed: {e}")
    
    # 3. Check current directory
    print("\n3. Current working directory:")
    print(f"  {os.getcwd()}")
    
    # 4. List files in current directory
    print("\n4. Files in current directory:")
    try:
        files = os.listdir()
        for f in files[:10]:  # Show first 10 files
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
    except Exception as e:
        print(f"  ❌ Error listing directory: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    import os
    main()
