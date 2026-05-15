from __future__ import annotations


class OvertureError(RuntimeError):
    """Base class for all Overture-loader failures."""


class OvertureUnreachable(OvertureError):
    """Backend cannot reach the data source (network failure, S3 timeout, etc.)."""


class OvertureSchemaMismatch(OvertureError):
    """A theme's columns do not match the curated schema in cfm.data.overture.schema.

    Usually means Overture added or removed columns between releases. To resolve:
    bump the pin in configs/data/overture_release.yaml, re-snapshot fixtures via
    scripts/snapshot_overture_fixtures.py, and update schema.py if needed.
    """


class RegionNotFound(OvertureError):
    """No configs/data/regions/<name>.yaml found for the requested region."""


class ReleaseNotConfigured(OvertureError):
    """configs/data/overture_release.yaml is missing or malformed."""


class OversizedFetch(OvertureError):
    """Estimated download exceeds the 2 GB threshold and confirm=False was passed.

    Pass confirm=True to load_region if you genuinely want a fetch this large.
    """


class CacheCorrupt(OvertureError):
    """A cached parquet's sha256 does not match what manifest.yaml recorded.

    The cache has been partially written or tampered with. Run
    scripts/cfm_data_invalidate.py to remove and refetch.
    """
