"""
Lightweight HTTP client for patchcore_server.py.

The actual implementation lives in detector_client.py, shared with
EfficientAD - the two servers expose an identical API and differ only in
port. No torch/anomalib dependency, so app.py stays light.

Note: unlike LiZADClient.run(), this returns a score only - no anomaly map.
PatchCore's anomaly-map output isn't wired through yet (see
patchcore_server.py), so the UI can show a score/verdict but not a heatmap
or bounding boxes for this detector.
"""

from detector_client import AnomalyDetectorClient


class PatchCoreClient(AnomalyDetectorClient):
    DEFAULT_PORT = 8001
