# Clone verification — 2026-09-17

- GitHub repository cloned into a separate directory.
- New Python 3.12 virtual environment installed from the pinned requirements; pip check passed.
- Setup PowerShell entry point completed successfully in that clone.
- Both desktop and manual inspection windows constructed/shown/closed offscreen without camera or robot access.
- CUDA checked on RTX 3090.
- Overview, detail-object and detail-defect models loaded and inferred on a blank synthetic image. This is an execution check, not an accuracy result.
- VLM 2B base and both adapters loaded successfully with unchanged file-integrity guards.
- 17 targeted tests passed (fresh camera configuration, FOV, detail deduplication, VLM region contracts, runtime restore integrity).
- All 15 release-part sizes and GitHub server SHA-256 digests matched the manifest. Actual authenticated release downloads were exercised on initial files; remaining large files were restored using the identical local release parts to avoid a redundant full transfer.
- Source-controlled runtime metadata survived Git checkout with identical hashes via .gitattributes.

The fresh-camera startup fix defers lens-profile binding until a camera is registered. It does not remove the registered-camera identity guard or approve robot coordinates. Existing tests were not edited.

Not checked here: actual camera capture on another PC, robot motion, laptop latency, independent defect accuracy, or a full VLM explanation run in the fresh environment.
