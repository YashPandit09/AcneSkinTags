"""
Step 21: Local Testing Script
Automated tests for the Streamlit app and all components
"""
import os
import sys
import subprocess

def test_imports():
    """Test if all required libraries can be imported."""
    print("\n" + "="*80)
    print("TESTING IMPORTS")
    print("="*80 + "\n")
    
    required_modules = [
        'tensorflow',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'cv2',
        'streamlit',
        'sklearn',
        'PIL',
        'tqdm'
    ]
    
    failed = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module:20s} - OK")
        except ImportError as e:
            print(f"✗ {module:20s} - FAILED: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ Missing modules: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓✓✓ All imports successful!")
        return True

def test_dataset():
    """Verify dataset is present and accessible."""
    print("\n" + "="*80)
    print("TESTING DATASET")
    print("="*80 + "\n")
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    
    # Check directories
    if not os.path.exists(config.DATASET_DIR):
        print(f"❌ Dataset directory not found: {config.DATASET_DIR}")
        return False
    
    if not os.path.exists(config.METADATA_CSV):
        print(f"❌ Metadata CSV not found: {config.METADATA_CSV}")
        return False
    
    # Count images
    from glob import glob
    image_count = 0
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        part_dir = os.path.join(config.DATASET_DIR, part)
        if os.path.exists(part_dir):
            images = glob(os.path.join(part_dir, '*.jpg'))
            image_count += len(images)
            print(f"✓ {part}: {len(images)} images")
    
    print(f"\n✓ Total images: {image_count}")
    
    if image_count < 10000:
        print("⚠ Warning: Expected ~10,000 images")
    
    return True

def test_preprocessing():
    """Test preprocessing modules."""
    print("\n" + "="*80)
    print("TESTING PREPROCESSING")
    print("="*80 + "\n")
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from preprocessing.hair_removal import remove_hair
        from preprocessing.normalization import normalize_imagenet
        import numpy as np
        
        # Test with dummy image
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test hair removal
        result = remove_hair(dummy_img)
        assert result.shape == dummy_img.shape
        print("✓ Hair removal: OK")
        
        # Test normalization
        normalized = normalize_imagenet(dummy_img)
        assert normalized.dtype == np.float32
        print("✓ Normalization: OK")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_model_factory():
    """Test model creation."""
    print("\n" + "="*80)
    print("TESTING MODEL FACTORY")
    print("="*80 + "\n")
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from models.model_factory import ModelFactory
        
        # Test building a small model
        print("Creating MobileNetV2 (this may take a moment)...")
        model = ModelFactory.create('mobilenetv2', num_classes=7)
        
        # Verify output shape
        import numpy as np
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        
        assert output.shape == (1, 7), f"Expected (1, 7), got {output.shape}"
        print(f"✓ Model output shape: {output.shape}")
        print("✓ Model factory: OK")
        
        return True
    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradcam():
    """Test Grad-CAM module."""
    print("\n" + "="*80)
    print("TESTING GRAD-CAM")
    print("="*80 + "\n")
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from explainability.gradcam import make_gradcam_heatmap, get_last_conv_layer_name
        from models.model_factory import ModelFactory
        import numpy as np
        
        # Create model
        model = ModelFactory.create('mobilenetv2', num_classes=7)
        
        # Get last conv layer
        last_layer = get_last_conv_layer_name(model)
        print(f"✓ Last conv layer detected: {last_layer}")
        
        # Test Grad-CAM generation
        dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        heatmap = make_gradcam_heatmap(dummy_img, model, last_layer)
        
        assert heatmap.shape == (7, 7) or len(heatmap.shape) == 2
        print(f"✓ Heatmap shape: {heatmap.shape}")
        print("✓ Grad-CAM: OK")
        
        return True
    except Exception as e:
        print(f"❌ Grad-CAM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_syntax():
    """Check if Streamlit app has syntax errors."""
    print("\n" + "="*80)
    print("TESTING STREAMLIT APP SYNTAX")
    print("="*80 + "\n")
    
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    
    if not os.path.exists(app_path):
        print(f"❌ app.py not found at: {app_path}")
        return False
    
    try:
        with open(app_path, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, app_path, 'exec')
        print("✓ app.py syntax: OK")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("DERM-X PROJECT - AUTOMATED TESTING")
    print("Step 21: Local Testing")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Dataset", test_dataset),
        ("Preprocessing", test_preprocessing),
        ("Model Factory", test_model_factory),
        ("Grad-CAM", test_gradcam),
        ("Streamlit Syntax", test_streamlit_syntax)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED!")
        print("\nYou can now run the Streamlit app:")
        print("  streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before running the app.")
    print("="*80 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
