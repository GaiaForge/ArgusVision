"""
Lightweight HTTP client for efficientad_server.py.

Implementation is shared with PatchCore in detector_client.py - both servers
expose an identical API and differ only in port. No torch/anomalib
dependency, so app.py stays light.

Score only, no anomaly map - same limitation as PatchCore's client.
"""

from detector_client import AnomalyDetectorClient


class EfficientAdClient(AnomalyDetectorClient):
    DEFAULT_PORT = 8002
