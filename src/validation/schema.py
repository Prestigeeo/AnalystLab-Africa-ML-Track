"""
Data Validation Schemas and Result Containers
==============================================
MENTOR NOTE FOR INTERNS:
In data engineering and ML engineering, "Garbage In, Garbage Out" is the golden rule.
A model cannot reliably score transactions if:
1. Expected columns are missing.
2. Data types are silently converted or corrupted (e.g., amount as text).
3. Critical business fields have missing values.
4. Categorical variables contain values never seen during training or unauthorized by business rules.

We use custom dataclasses to encapsulate validation outputs cleanly.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd


@dataclass
class ValidationErrorItem:
    """Represents an individual validation issue detected."""
    field: str
    error_type: str  # e.g., 'MISSING_COLUMN', 'TYPE_MISMATCH', 'INVALID_CATEGORY', 'NULL_VALUE', 'RANGE_ERROR'
    message: str
    severity: str    # 'ERROR' (blocks pipeline) or 'WARNING' (logged/handled)
    invalid_rows_count: int = 0
    sample_invalid_values: List[Any] = field(default_factory=list)


@dataclass
class ValidationReport:
    """
    Comprehensive validation report summarizing data quality.
    
    MENTOR NOTE FOR INTERNS:
    Returning a structured object rather than simply throwing an unhandled exception
    gives the calling application flexibility:
    - In an automated batch pipeline, you can quarantine invalid rows and process valid ones.
    - In an online REST API, you can return a detailed 422 Unprocessable Entity payload.
    """
    is_valid: bool
    total_records: int
    errors: List[ValidationErrorItem] = field(default_factory=list)
    warnings: List[ValidationErrorItem] = field(default_factory=list)
    action_taken: str = "PENDING"  # 'PROCEED', 'QUARANTINE_RECORDS', 'REJECT_ALL'

    def summary(self) -> Dict[str, Any]:
        """Returns a human-readable dictionary summary of the validation report."""
        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "action_taken": self.action_taken,
            "errors": [e.__dict__ for e in self.errors],
            "warnings": [w.__dict__ for w in self.warnings],
        }

    def print_report(self) -> None:
        """Pretty print the validation report to console."""
        status = "PASSED" if self.is_valid else "FAILED"
        print(f"\n{'='*20} DATA VALIDATION REPORT: {status} {'='*20}")
        print(f"Total Records Inspected: {self.total_records}")
        print(f"Errors: {len(self.errors)} | Warnings: {len(self.warnings)}")
        print(f"Pipeline Action: {self.action_taken}")
        
        if self.errors:
            print("\nCritical Errors (Blocking):")
            for err in self.errors:
                print(f"  [x] Field: {err.field} | Type: {err.error_type}")
                print(f"      Message: {err.message}")
                if err.sample_invalid_values:
                    print(f"      Samples: {err.sample_invalid_values[:5]}")
                    
        if self.warnings:
            print("\nNon-Critical Warnings (Monitored/Imputed):")
            for warn in self.warnings:
                print(f"  [!] Field: {warn.field} | Type: {warn.error_type}")
                print(f"      Message: {warn.message}")
        print(f"{'='*60}\n")
