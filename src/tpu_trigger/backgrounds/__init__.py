"""Background-noise generation for a KM3NeT 31-PMT DOM.

All generators emit the common hit-stream format: a tuple of numpy arrays
(t_ns int64, pmt int8, tot_ns int16) for a single DOM, sorted by time.
"""
