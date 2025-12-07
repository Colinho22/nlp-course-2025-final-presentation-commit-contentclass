# Moodle Site Screenshots - Before/After Documentation

**Date**: 2025-12-07
**Purpose**: Visual documentation of Moodle site improvements

## Overview

This directory contains visual evidence of the transformation from FHGR-branded Moodle site to GitHub-based course site.

## File Inventory

### Before Screenshots (5 files)
Baseline screenshots before improvements:
- `before_index.png` (196K) - Original landing page
- `before_schedule.png` (181K) - Original weekly schedule
- `before_pdfs.png` (234K) - Original PDF listing
- `before_notebooks.png` (143K) - Original notebook listing
- `before_assignments.png` (129K) - Original assignments page

### After Screenshots (5 files)
Screenshots after all improvements applied:
- `after_index.png` (214K) - Enhanced landing page (+18K)
- `after_schedule.png` (221K) - Richer weekly schedule (+40K)
- `after_pdfs.png` (161K) - Streamlined PDF listing (-73K)
- `after_notebooks.png` (150K) - Improved notebook listing (+7K)
- `after_assignments.png` (180K) - Enhanced assignments page (+51K)

### Comparison Images (5 files)
Side-by-side before/after comparisons:
- `comparison_index.png` (612K) - 3816x1080px
- `comparison_schedule.png` (602K) - 3598x1448px
- `comparison_pdfs.png` (554K) - 3469x1246px
- `comparison_notebooks.png` (435K) - 3804x1080px
- `comparison_assignments.png` (503K) - 3625x1080px

## Scripts

### Screenshot Capture
- `capture_before.py` - Captures baseline screenshots
- `capture_after.py` - Captures improved screenshots
- Uses Playwright with 1920x1080 viewport, full-page capture

### Comparison Generation
- `create_comparison_grid.py` - Creates side-by-side comparisons
- Resizes to same height, adds 20px gap between images

## Key Changes Documented

### Visual Improvements
1. **Removed FHGR Branding**: All university-specific references eliminated
2. **Added GitHub Links**: All resources link to public GitHub repository
3. **Enhanced Content**: Richer descriptions, better organization
4. **Improved Navigation**: Consistent header/footer across all pages
5. **Better Visual Hierarchy**: Clearer sections, improved readability

### Size Changes (Screenshot File Sizes)
- **Expanded pages**: index (+18K), schedule (+40K), assignments (+51K)
- **Simplified page**: pdfs (-73K) - removed redundancy
- **Enhanced page**: notebooks (+7K) - improved descriptions

## Verification Results

Automated verification performed on regenerated HTML:
- FHGR mentions: 0 (PASSED)
- GitHub links: 11 occurrences across 5 files (PASSED)

## Usage

To regenerate screenshots:
```bash
cd D:\Joerg\Research\slides\2025_NLP_16\docs\moodle\screenshots

# Capture before (if needed)
python capture_before.py

# Regenerate site with improvements
cd ../../../docs_backup_static/scripts
python generate_moodle_site.py

# Capture after
cd ../../docs/moodle/screenshots
python capture_after.py

# Create comparisons
python create_comparison_grid.py
```

## Report

See `COMPARISON_REPORT.md` for detailed before/after analysis.
