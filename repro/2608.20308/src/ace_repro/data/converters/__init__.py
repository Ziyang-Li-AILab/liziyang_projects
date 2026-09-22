"""Dataset -> shared-format converters (gaps_filled G "Shared MANO format").

Each converter writes ``Clip`` objects (``ace_repro.data.store.save_clip``) under
``<data_root>/<dataset>/{train,test}/``. All of them go through
``common.build_clip`` so the MANO convention, visibility gate, K rescaling and
VAE encoding are identical across sources.
"""
