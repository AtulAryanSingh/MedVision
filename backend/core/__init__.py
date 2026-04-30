"""
core/__init__.py

What this package does:
  Provides the low-level primitives that every other module builds on:
  image loading (DICOM, NIfTI, PNG/JPG), NumPy conversion, metadata
  extraction, slice helpers, and display utilities.

Why it exists:
  Centralising all format-specific I/O here keeps every higher-level
  module agnostic about the storage format.
"""

PACKAGE_NAME = "MedVision Core"
