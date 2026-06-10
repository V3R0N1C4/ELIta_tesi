# Palette colori estesa (include le 8 base + amore e neutrale)
EMOTION_COLORS = {
    'gioia':        '#FDD835',  # Giallo
    'aspettativa':  '#FB8C00',  # Arancione
    'rabbia':       '#E53935',  # Rosso
    'disgusto':     '#8E24AA',  # Viola
    'tristezza':    '#0000FF',  # Blu
    'sorpresa':     '#039BE5',  # Azzurro
    'paura':        '#228B22',  # Verde scuro
    'fiducia':      '#81C784',  # Verde chiaro
    'amore':        '#E91E63',  # Rosa/Magenta
    'neutrale':     '#9E9E9E'   # Grigio
}

# Liste di definizioni colonne
BASIC_EMOTIONS = [ 'gioia', 'aspettativa', 'rabbia', 'disgusto', 'tristezza', 'sorpresa', 'paura',  'fiducia']

ALL_EMOTIONS = BASIC_EMOTIONS + ['amore', 'neutrale']

VAD_FEATURES = ['valenza', 'attivazione', 'dominanza']