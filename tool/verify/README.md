Python checks that mirror the Dart crop/preprocessing code line-for-line and run
the real TFLite models. `pip install ai-edge-litert scikit-image numpy`, then:

    python test_geometry.py   # NV21 sampler: all 4 sensor rotations, padded strides
    python test_align.py      # 5-point alignment: embedding stability under jitter/tilt/scale
    python test_spoof.py      # anti-spoof model I/O (scores on sample photos)
