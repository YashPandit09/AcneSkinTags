"""
Project Cleanup Script - Keep Only Deployment Files
Removes unnecessary files and organizes for deployment
"""
import os
import shutil

# Files to KEEP (essential for deployment)
KEEP_FILES = {
    # Core application
    'app_pytorch.py',
    'config.py',
    'requirements.txt',
    'README.md',
    
    # Trained model
    'best_model_8class_pytorch.pth',
    
    # Results & visualizations
    'training_history_8class_pytorch.png',
    'confusion_matrix_8class.png',
    'classification_report_8class.txt',
    'per_class_metrics.csv',
    
    # Documentation
    'FINAL_RESULTS_8CLASS.md',
    'PAPER_METHODOLOGY.md',
    'PYTORCH_GPU_GUIDE.md',
    'RESEARCH_PAPER_KIT.md',
    
    # Git
    '.gitignore',
}

# Directories to KEEP
KEEP_DIRS = {
    'Dataset',
    'gradcam_8class',
    'preprocessing',
    'explainability',
    'utils',
    '.git',
}

# Files to DELETE (old versions, duplicates, temporary)
DELETE_FILES = {
    # Old TensorFlow files
    'app.py',  # Old TensorFlow version
    'train_model.py',
    'train_model_optimized.py',
    'train_comprehensive.py',
    'gpu_utils.py',
    'verify_gpu_setup.py',
    'benchmark_training.py',
    
    # Old PyTorch training versions
    'train_pytorch.py',  # Keep only 8-class version
    'train_pytorch_8class_ph2.py',  # PH2 version not completed
    'verify_pytorch_gpu.py',
    
    # Exploration/analysis scripts
    'explore_ph2.py',
    'analyze_ph2_new.py',
    'check_ph2_labels.py',
    'extract_ph2_melanoma.py',
    
    # Old evaluation scripts
    'evaluate_model.py',
    'evaluate_pytorch.py',  # Keep only 8-class version
    'final_metrics_pytorch.py',  # Old 7-class version
    'evaluate_confusion_8class.py',  # Already generated outputs
    
    # Other utilities
    'create_demo_model.py',
    'test_local.py',
    'prepare_submission.py',
    'generate_paper_figures.py',
    
    # Old/duplicate documentation
    'GPU_OPTIMIZATION_GUIDE.md',
    'GPU_OPTIMIZATION_SUMMARY.md',
    'GRADCAM_ANALYSIS_RESULTS.md',
    'README_GPU_OPTIMIZATION.md',
    'QUICKSTART.md',
    'USAGE_GUIDE.md',
    'Derm-X_Final_Descriptive_Report_Template.md',
    
    # Old model files
    'best_model_pytorch.pth',  # 7-class version
    
    # Old results
    'training_history_pytorch.png',  # 7-class version
    'final_confusion_matrix.png',  # 7-class version
    'final_classification_report.txt',  # 7-class version
}

# Directories to DELETE
DELETE_DIRS = {
    '__pycache__',
    'saved_models',
    'models',
    'evaluation',
    'results',
    'visualizations',
    'paper_figures',
    'gradcam_visualizations',  # Old 7-class version, keep gradcam_8class
}

def cleanup_project():
    print("=" * 80)
    print("PROJECT CLEANUP FOR DEPLOYMENT")
    print("=" * 80)
    
    deleted_files = []
    deleted_dirs = []
    
    # Delete unnecessary files
    print("\n[1/3] Removing unnecessary files...")
    for filename in DELETE_FILES:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                deleted_files.append(filename)
                print(f"  ✓ Deleted: {filename}")
            except Exception as e:
                print(f"  ✗ Failed to delete {filename}: {e}")
    
    # Delete unnecessary directories
    print("\n[2/3] Removing unnecessary directories...")
    for dirname in DELETE_DIRS:
        if os.path.exists(dirname):
            try:
                shutil.rmtree(dirname)
                deleted_dirs.append(dirname)
                print(f"  ✓ Deleted: {dirname}/")
            except Exception as e:
                print(f"  ✗ Failed to delete {dirname}: {e}")
    
    # Create deployment structure
    print("\n[3/3] Creating deployment-ready structure...")
    
    # Create docs folder for documentation
    if not os.path.exists('docs'):
        os.makedirs('docs')
        print("  ✓ Created: docs/")
    
    # Move documentation files to docs/
    doc_files = ['FINAL_RESULTS_8CLASS.md', 'PAPER_METHODOLOGY.md', 
                 'PYTORCH_GPU_GUIDE.md', 'RESEARCH_PAPER_KIT.md']
    for doc in doc_files:
        if os.path.exists(doc):
            try:
                shutil.move(doc, os.path.join('docs', doc))
                print(f"  ✓ Moved {doc} to docs/")
            except Exception as e:
                print(f"  ✗ Failed to move {doc}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CLEANUP COMPLETE")
    print("=" * 80)
    print(f"✓ Deleted {len(deleted_files)} files")
    print(f"✓ Deleted {len(deleted_dirs)} directories")
    print(f"✓ Organized documentation into docs/")
    print("\nDeployment-ready structure created!")
    print("=" * 80)
    
    # Show final structure
    print("\n📁 Final Project Structure:")
    print("├── app_pytorch.py          # Main Streamlit app")
    print("├── config.py               # Configuration")
    print("├── requirements.txt        # Dependencies")
    print("├── best_model_8class_pytorch.pth  # Trained model")
    print("├── README.md               # Project readme")
    print("├── Dataset/                # Training data")
    print("├── gradcam_8class/         # Grad-CAM visualizations")
    print("├── docs/                   # Documentation")
    print("│   ├── FINAL_RESULTS_8CLASS.md")
    print("│   ├── PAPER_METHODOLOGY.md")
    print("│   ├── PYTORCH_GPU_GUIDE.md")
    print("│   └── RESEARCH_PAPER_KIT.md")
    print("├── preprocessing/          # Preprocessing utilities")
    print("├── explainability/         # Grad-CAM utilities")
    print("└── utils/                  # Helper functions")
    print("\n✅ Ready for deployment!")

if __name__ == "__main__":
    response = input("\n⚠️  This will delete unnecessary files. Continue? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        cleanup_project()
    else:
        print("Cleanup cancelled.")
