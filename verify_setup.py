"""
Verify ModelArena Setup
"""
import os
from pathlib import Path

def verify_setup():
    """Verify that the project structure is correct before uploading to Google Drive"""
    
    print("="*70)
    print("🔍 MODELARENA SETUP VERIFICATION")
    print("="*70)
    
    issues = []
    warnings = []
    
    # Check main files
    print("\n📄 Checking main files...")
    required_files = [
        'TRAINING_PIPELINE.ipynb',
        'INFERENCE_PIPELINE.ipynb',
        'README.md',
        'README_SETUP.md',
        'QUICK_REFERENCE.md',
        'requirements.txt',
        '.gitignore'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            issues.append(f"Missing file: {file}")
    
    # Check archive structure
    print("\n📁 Checking archive structure...")
    archive_path = Path('archive')
    
    if archive_path.exists():
        print(f"  ✅ archive/ folder exists")
        
        # Check CSV files
        if (archive_path / 'train_labels.csv').exists():
            print(f"  ✅ archive/train_labels.csv")
        else:
            print(f"  ❌ archive/train_labels.csv - MISSING")
            issues.append("Missing train_labels.csv")
        
        if (archive_path / 'test_public.csv').exists():
            print(f"  ✅ archive/test_public.csv")
        else:
            print(f"  ❌ archive/test_public.csv - MISSING")
            issues.append("Missing test_public.csv")
        
        # Check train folders
        fake_path = archive_path / 'train' / 'fake'
        real_path = archive_path / 'train' / 'real'
        test_path = archive_path / 'test'
        
        if fake_path.exists():
            fake_count = len(list(fake_path.glob('*.mp4')))
            if fake_count == 300:
                print(f"  ✅ archive/train/fake/ ({fake_count} videos)")
            elif fake_count > 0:
                print(f"  ⚠️ archive/train/fake/ ({fake_count} videos - expected 300)")
                warnings.append(f"Expected 300 fake videos, found {fake_count}")
            else:
                print(f"  ⚠️ archive/train/fake/ (empty)")
                warnings.append("Fake folder is empty")
        else:
            print(f"  ❌ archive/train/fake/ - MISSING")
            issues.append("Missing train/fake folder")
        
        if real_path.exists():
            real_count = len(list(real_path.glob('*.mp4')))
            if real_count == 300:
                print(f"  ✅ archive/train/real/ ({real_count} videos)")
            elif real_count > 0:
                print(f"  ⚠️ archive/train/real/ ({real_count} videos - expected 300)")
                warnings.append(f"Expected 300 real videos, found {real_count}")
            else:
                print(f"  ⚠️ archive/train/real/ (empty)")
                warnings.append("Real folder is empty")
        else:
            print(f"  ❌ archive/train/real/ - MISSING")
            issues.append("Missing train/real folder")
        
        if test_path.exists():
            test_count = len(list(test_path.glob('*.mp4')))
            if test_count == 200:
                print(f"  ✅ archive/test/ ({test_count} videos)")
            elif test_count > 0:
                print(f"  ⚠️ archive/test/ ({test_count} videos - expected 200)")
                warnings.append(f"Expected 200 test videos, found {test_count}")
            else:
                print(f"  ⚠️ archive/test/ (empty)")
                warnings.append("Test folder is empty")
        else:
            print(f"  ❌ archive/test/ - MISSING")
            issues.append("Missing test folder")
    else:
        print(f"  ❌ archive/ folder - MISSING")
        issues.append("Missing archive folder")
    
    # Check submission directory
    print("\n📦 Checking submission directory...")
    submission_path = Path('SUBMISSION_DIRECTORY')
    
    if submission_path.exists():
        print(f"  ✅ SUBMISSION_DIRECTORY/ folder exists")
        if (submission_path / 'README.md').exists():
            print(f"  ✅ SUBMISSION_DIRECTORY/README.md")
        else:
            print(f"  ⚠️ SUBMISSION_DIRECTORY/README.md - missing (will be created)")
    else:
        print(f"  ❌ SUBMISSION_DIRECTORY/ folder - MISSING")
        issues.append("Missing SUBMISSION_DIRECTORY folder")
    
    # Calculate total size
    print("\n💾 Calculating dataset size...")
    try:
        total_size = 0
        if archive_path.exists():
            for file in archive_path.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
            
            size_mb = total_size / (1024 * 1024)
            size_gb = size_mb / 1024
            
            if size_gb > 1:
                print(f"  📊 Total archive size: {size_gb:.2f} GB")
            else:
                print(f"  📊 Total archive size: {size_mb:.2f} MB")
            
            if size_mb < 100:
                warnings.append(f"Archive size ({size_mb:.0f}MB) seems too small - videos may be missing")
    except Exception as e:
        print(f"  ⚠️ Could not calculate size: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    if not issues and not warnings:
        print("\n✅ ALL CHECKS PASSED!")
        print("\n🚀 Next steps:")
        print("  1. Upload the 'archive' folder to Google Drive at:")
        print("     My Drive/ModelArena/archive/")
        print("  2. Open TRAINING_PIPELINE.ipynb in Google Colab")
        print("  3. Enable GPU Runtime")
        print("  4. Start training!")
    else:
        if issues:
            print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"  • {issue}")
        
        if warnings:
            print(f"\n⚠️ WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"  • {warning}")
        
        print("\n📖 See README_SETUP.md for detailed setup instructions")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    verify_setup()
