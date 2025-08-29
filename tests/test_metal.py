#!/usr/bin/env python3
"""
Test script to verify Metal Performance Shaders (MPS) acceleration for embeddings.
Run this to test if Metal acceleration is working properly.
"""

import torch
import time
from sentence_transformers import SentenceTransformer

def test_device_performance():
    """Test embedding performance on different devices."""
    
    # Test texts
    test_texts = [
        "This is a sample document for testing embedding performance.",
        "Machine learning models can benefit from GPU acceleration.",
        "Metal Performance Shaders provide GPU acceleration on macOS.",
        "ChromaDB is a vector database for storing embeddings.",
        "PDF processing can be computationally intensive."
    ] * 20  # Multiply to get more data for timing
    
    print("Testing Embedding Performance")
    print("=" * 40)
    
    # Test available devices
    devices_to_test = []
    
    # Check MPS (Metal)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices_to_test.append(("mps", "Metal Performance Shaders"))
    
    # Check CUDA
    if torch.cuda.is_available():
        devices_to_test.append(("cuda", f"CUDA GPU: {torch.cuda.get_device_name()}"))
    
    # Always test CPU
    devices_to_test.append(("cpu", "CPU"))
    
    results = {}
    
    for device, description in devices_to_test:
        print(f"\nTesting {description} ({device})...")
        
        try:
            # Initialize model on device
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            
            # Warm up
            _ = model.encode(test_texts[:5])
            
            # Time the encoding
            start_time = time.time()
            embeddings = model.encode(test_texts)
            end_time = time.time()
            
            duration = end_time - start_time
            results[device] = {
                'duration': duration,
                'description': description,
                'texts_processed': len(test_texts),
                'embeddings_shape': embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'
            }
            
            print(f"  ✅ Successfully processed {len(test_texts)} texts")
            print(f"  ⏱️  Time taken: {duration:.3f} seconds")
            print(f"  📊 Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'}")
            
        except Exception as e:
            print(f"  ❌ Error testing {device}: {e}")
            results[device] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 40)
    print("PERFORMANCE SUMMARY")
    print("=" * 40)
    
    successful_tests = {k: v for k, v in results.items() if 'duration' in v}
    
    if successful_tests:
        fastest_device = min(successful_tests.keys(), key=lambda x: successful_tests[x]['duration'])
        
        for device, result in successful_tests.items():
            speed_indicator = "🚀 FASTEST" if device == fastest_device else ""
            print(f"{result['description']}: {result['duration']:.3f}s {speed_indicator}")
        
        # Performance comparison
        if len(successful_tests) > 1:
            cpu_time = successful_tests.get('cpu', {}).get('duration', 0)
            if 'mps' in successful_tests and cpu_time > 0:
                mps_time = successful_tests['mps']['duration']
                speedup = cpu_time / mps_time
                print(f"\n🔥 Metal speedup vs CPU: {speedup:.2f}x faster")
            
    else:
        print("No successful tests completed.")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 20)
    
    if 'mps' in successful_tests:
        print("✅ Metal acceleration is working!")
        print("   Your ChromaDB Navigator will use GPU acceleration for embeddings.")
    elif torch.backends.mps.is_available():
        print("⚠️  Metal is available but failed to run.")
        print("   The app will fall back to CPU processing.")
    else:
        print("ℹ️  Metal is not available on this system.")
        print("   The app will use CPU processing.")

if __name__ == "__main__":
    test_device_performance()