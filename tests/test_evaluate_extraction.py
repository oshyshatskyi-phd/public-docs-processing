import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from evaluate_extraction import calculate_metrics, compare_parcels, compare_documentation
from collections import defaultdict

def test_calculate_metrics_empty():
    """Test calculate_metrics with empty stats"""
    stats = {}
    results = calculate_metrics(stats)
    assert len(results) == 0

def test_calculate_metrics_perfect_score():
    """Test calculate_metrics with perfect scores"""
    stats = {
        "number": {"tp": 10, "fp": 0, "fn": 0}
    }
    results = calculate_metrics(stats)
    assert len(results) == 1
    row = results.iloc[0]
    assert row["field"] == "number"
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0
    assert row["true_positives"] == 10
    assert row["false_positives"] == 0
    assert row["false_negatives"] == 0

def test_calculate_metrics_zero_score():
    """Test calculate_metrics with zero scores"""
    stats = {
        "number": {"tp": 0, "fp": 5, "fn": 5}
    }
    results = calculate_metrics(stats)
    assert len(results) == 1
    row = results.iloc[0]
    assert row["field"] == "number"
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert row["f1"] == 0.0

def test_compare_parcels_simple():
    """Test compare_parcels with simple test case"""
    human_data = [
        {"number": "123", "area": 100, "purpose_code": "A"},
        {"number": "456", "area": 200, "purpose_code": "B"}
    ]
    ai_data = [
        {"number": "123", "area": 100, "purpose_code": "C"},  # purpose_code mismatch
        {"number": "789", "area": 300, "purpose_code": "D"}   # unmatched parcel
    ]
    
    metrics = compare_parcels(human_data, ai_data)
    
    # Check area field metrics
    assert metrics["area"]["tp"] == 1  # One matching area
    assert metrics["area"]["fp"] == 1  # One false area (from unmatched parcel)
    assert metrics["area"]["fn"] == 1  # One missing area (from parcel 456)
    
    # Check purpose_code field metrics
    assert metrics["purpose_code"]["tp"] == 0  # No matching purpose codes
    assert metrics["purpose_code"]["fp"] == 2  # Two wrong purpose codes
    assert metrics["purpose_code"]["fn"] == 2  # Two missing correct purpose codes

def test_compare_documentation_simple():
    """Test compare_documentation with simple test case"""
    human_data = [
        {"type": "parcel", "id": 1, "number": "123"},
        {"type": "parcel", "id": 2, "number": None},
        {'documentation_type': 'LAND_PLOT_ALLOCATION_PROJECT', 'involved_parcels': [1], 'id': 0, 'type': 'documentation'},
        {'documentation_type': 'LAND_PLOT_ALLOCATION_PROJECT', 'involved_parcels': [2], 'id': 1, 'type': 'documentation'}
    ]
    ai_data = [
        {"type": "parcel", "id": 1, "number": "123"},
        {'documentation_type': 'LAND_PLOT_ALLOCATION_PROJECT', 'involved_parcels': [1], 'id': 0, 'type': 'documentation'},
        {'documentation_type': 'LAND_PLOT_ALLOCATION_PROJECT', 'involved_parcels': [2], 'id': 1, 'type': 'documentation'},
        {"type": "documentation", "documentation_type": "license", "involved_parcels": [0]}  # Extra doc
    ]
    
    metrics = compare_documentation(human_data, ai_data)
    
    # Check documentation references
    assert metrics["documentation_type"]["tp"] == 2  # Two matching doc type
    assert metrics["documentation_type"]["fp"] == 1  # One extra doc type
    assert metrics["documentation_type"]["fn"] == 0  # No missing doc types
    
    # Check parcel references
    assert metrics["parcel_references"]["tp"] == 2  # Two correct parcel reference
    assert metrics["parcel_references"]["fp"] == 1  # One extra parcel reference
    assert metrics["parcel_references"]["fn"] == 0  # No missing parcel references

def test_compare_parcels_missing_values():
    """Test compare_parcels handling of missing values"""
    human_data = [
        {"number": "123", "area": 100},  # Missing purpose_code
        {"number": "456", "area": None, "purpose_code": "B"}  # None value
    ]
    ai_data = [
        {"number": "123", "purpose_code": "A"},  # Missing area
        {"number": "456", "area": 200}  # Missing purpose_code
    ]
    
    metrics = compare_parcels(human_data, ai_data)
    
    # Check area field metrics
    assert metrics["area"]["tp"] == 0
    assert metrics["area"]["fp"] == 1  # One wrong area value
    assert metrics["area"]["fn"] == 1  # One missing area value
    
    # Check purpose_code field metrics
    assert metrics["purpose_code"]["tp"] == 0
    assert metrics["purpose_code"]["fp"] == 1  # One wrong purpose code
    assert metrics["purpose_code"]["fn"] == 1  # One missing purpose code
