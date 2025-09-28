import os
import sys
import time
import subprocess
from pathlib import Path
import json

def test_notebook(notebook_path):
    """
    Test a single Jupyter notebook by executing it.
    Returns: (success, execution_time, error_message)
    """
    print(f"\nTesting: {notebook_path}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [
                'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--ExecutePreprocessor.timeout=600',
                '--output', '/dev/null',
                notebook_path
            ],
            capture_output=True,
            text=True,
            timeout=700
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(f"  PASS ({execution_time:.1f}s)")
            return True, execution_time, None
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            print(f"  FAIL ({execution_time:.1f}s)")
            print(f"  Error: {error_msg[:200]}")
            return False, execution_time, error_msg

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print(f"  TIMEOUT (>{execution_time:.1f}s)")
        return False, execution_time, "Execution timeout (>10 minutes)"

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"  ERROR ({execution_time:.1f}s)")
        print(f"  Error: {str(e)}")
        return False, execution_time, str(e)

def find_all_notebooks():
    """Find all lab notebooks in the course."""
    notebook_paths = []

    base_path = Path('NLP_slides')

    for week_dir in sorted(base_path.glob('week*')):
        lab_dir = week_dir / 'lab'
        if lab_dir.exists():
            for notebook in sorted(lab_dir.glob('*.ipynb')):
                if '.ipynb_checkpoints' not in str(notebook):
                    notebook_paths.append(notebook)

    return notebook_paths

def main():
    """Test all lab notebooks and generate report."""
    print("=" * 70)
    print("NLP Course 2025 - Lab Notebook Testing")
    print("=" * 70)

    notebooks = find_all_notebooks()
    print(f"\nFound {len(notebooks)} notebooks to test\n")

    results = []

    for notebook in notebooks:
        week = notebook.parts[1]
        success, exec_time, error = test_notebook(str(notebook))

        results.append({
            'week': week,
            'notebook': notebook.name,
            'path': str(notebook),
            'success': success,
            'execution_time': exec_time,
            'error': error
        })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed

    print(f"\nTotal: {len(results)} notebooks")
    print(f"Passed: {passed} ({100*passed/len(results):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(results):.1f}%)")

    total_time = sum(r['execution_time'] for r in results)
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")

    if failed > 0:
        print("\nFailed notebooks:")
        for r in results:
            if not r['success']:
                print(f"  - {r['week']}: {r['notebook']}")
                if r['error']:
                    print(f"    Error: {r['error'][:100]}")

    report_path = 'notebook_test_results.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {report_path}")

    markdown_report = generate_markdown_report(results)
    report_md_path = 'NOTEBOOK_TEST_REPORT.md'
    with open(report_md_path, 'w') as f:
        f.write(markdown_report)

    print(f"Markdown report saved to: {report_md_path}")

    return 0 if failed == 0 else 1

def generate_markdown_report(results):
    """Generate a markdown report of test results."""
    report = ["# NLP Course 2025 - Lab Notebook Test Report\n"]
    report.append(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Total Notebooks:** {len(results)}\n")

    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed

    report.append(f"**Passed:** {passed} ({100*passed/len(results):.1f}%)\n")
    report.append(f"**Failed:** {failed} ({100*failed/len(results):.1f}%)\n")

    total_time = sum(r['execution_time'] for r in results)
    report.append(f"**Total Execution Time:** {total_time/60:.1f} minutes\n")

    report.append("\n## Results by Week\n")
    report.append("\n| Week | Notebook | Status | Time (s) | Notes |\n")
    report.append("|------|----------|--------|----------|-------|\n")

    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        time_str = f"{r['execution_time']:.1f}"
        notes = "" if r['success'] else (r['error'][:50] + "..." if r['error'] else "")
        report.append(f"| {r['week']} | {r['notebook']} | {status} | {time_str} | {notes} |\n")

    if failed > 0:
        report.append("\n## Failed Notebooks Details\n")
        for r in results:
            if not r['success']:
                report.append(f"\n### {r['week']} - {r['notebook']}\n")
                report.append(f"**Path:** `{r['path']}`\n")
                report.append(f"**Execution Time:** {r['execution_time']:.1f}s\n")
                if r['error']:
                    report.append(f"**Error:**\n```\n{r['error'][:500]}\n```\n")

    report.append("\n## Recommendations\n")

    if failed == 0:
        report.append("All notebooks executed successfully! The course is ready for delivery.\n")
    else:
        report.append("The following actions are recommended:\n")
        report.append("1. Review and fix failed notebooks\n")
        report.append("2. Check for missing dependencies\n")
        report.append("3. Verify data file paths\n")
        report.append("4. Test in clean environment\n")

    return "".join(report)

if __name__ == "__main__":
    sys.exit(main())