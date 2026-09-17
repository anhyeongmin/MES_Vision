"""Screen reading order for a frozen overview, independent of detector score."""
from statistics import median


def reading_order(objects):
    """Cluster nearby centre Y values into rows, then order each row by X.

    A row spans at most half the median object height. Anchoring at the
    topmost centre prevents chained neighbours from merging several rows.
    Return the original objects without modifying their evidence IDs or boxes.
    """
    objects = list(objects)
    if not objects:
        return []

    def centre(obj):
        b = obj['effective_box']
        return ((b['x1'] + b['x2']) / 2, (b['y1'] + b['y2']) / 2)

    tolerance = median(o['effective_box']['y2'] - o['effective_box']['y1']
                       for o in objects) * .5
    pending = sorted(objects, key=lambda o: (centre(o)[1], centre(o)[0], o.get('object_id', '')))
    rows = []
    for obj in pending:
        y = centre(obj)[1]
        if not rows or y - centre(rows[-1][0])[1] > tolerance:
            rows.append([])
        rows[-1].append(obj)
    return [obj for row in rows for obj in sorted(
        row, key=lambda o: (centre(o)[0], centre(o)[1], o.get('object_id', '')))]
