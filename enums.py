import enum

# query strategy in active learning
STRATEGY = enum.Enum('STRATEGY', ('RAND', 'LC','MC', 'MNLP', 'TTE', 'TE', 'PRECISION', 'LAZY', 'NORMAL', 'SELF'))
