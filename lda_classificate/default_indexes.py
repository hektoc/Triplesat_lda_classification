default_indexes= {
    'MCARI2': {
        'in_values': ['g', 'r', 'ir'],
        'return': '1.5 * (2.5 * (ir - r) - 1.3 * (ir - g)) / math.sqrt((2 * ir + 1) ** 2 - (6 * ir - 5 * math.sqrt(r)) - 0.5)'},

    'MTVI': {
        'in_values': ['g', 'r', 'ir'],
        'return': '1.2 * (1.2 * (ir - g) - 2.5 * (r - g))'},

    'MTVI2': {
        'in_values': ['g', 'r', 'ir'],
        'return': '(1.5 * (1.2 * (ir - g) - 2.5 * (r - g))) / math.sqrt((2 * ir + 1) ** 2 - (6 * ir - 5 * math.sqrt(g)) - 0.5)'},

    'SIPI': {
        'in_values': ['b', 'r', 'ir'],
        'return': '(ir - b) / (ir - r)'},

    'LV': {
        'in_values': ['b', 'g', 'r'],
        'return': 'g * r / b ** 2'},

    'BR': {
        'in_values': ['b', 'r'],
        'return': 'b / r'},

    'GNDVI2': {
        'in_values': ['g', 'ir'],
        'return': '(ir - g) / (ir + g)'},

    'DI1': {
        'in_values': ['g', 'ir'],
        'return': 'ir - g'},

    'SIPI2': {
        'in_values': ['b', 'r', 'ir'],
        'return': '(ir - b) / (ir + r)'},

    'NPCI': {
        'in_values': ['b', 'r'],
        'return': '(r - b) / (r + b)'},

    'BR625': {
        'in_values': ['r', 'ir'],
        'return': 'r / ir'},

    'PSNDchla': {
        'in_values': ['r', 'ir'],
        'return': '(ir - r) / (ir + r)'},

    'PSSRa': {
        'in_values': ['r', 'ir'],
        'return': 'ir / r'},

    'PSSRc': {
        'in_values': ['b', 'ir'],
        'return': 'ir / b'},

    'New2': {
        'in_values': ['b', 'r'],
        'return': '(b - r) / (b + r)'},

    'New13': {
        'in_values': ['g', 'ir'],
        'return': 'ir / g'},

}
