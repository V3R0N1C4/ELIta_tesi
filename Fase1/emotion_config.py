# Palette colori estesa (include le 8 base + amore e neutrale)
EMOTION_COLORS = {
    'gioia': '#FDD835',       # Giallo
    'tristezza': '#1E88E5',   # Blu
    'rabbia': '#E53935',      # Rosso
    'paura': '#43A047',       # Verde scuro
    'disgusto': '#8E24AA',    # Viola
    'fiducia': '#81C784',     # Verde chiaro
    'sorpresa': '#039BE5',    # Azzurro
    'aspettativa': '#FB8C00', # Arancione
    'amore': '#E91E63',       # Rosa/Magenta
    'neutrale': '#9E9E9E'     # Grigio
}

# Liste di definizioni colonne
BASIC_EMOTIONS = [
    'gioia', 'tristezza', 'rabbia', 'paura',
    'disgusto', 'fiducia', 'sorpresa', 'aspettativa'
]

ALL_EMOTIONS = BASIC_EMOTIONS + ['amore', 'neutrale']

VAD_FEATURES = ['valenza', 'attivazione', 'dominanza']