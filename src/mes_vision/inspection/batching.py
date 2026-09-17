"""Small batches contain already available crops from one capture only."""


def validate_crops(crops):
    crops=tuple(crops)
    if len(crops)>4: raise ValueError('At most four current crops per batch')
    if any(not isinstance(value,str) or not value for crop in crops for value in (crop.object_id,crop.frame_id)):
        raise ValueError('Missing batch identity')
    if len({c.object_id for c in crops})!=len(crops): raise ValueError('Duplicate batch object identity')
    if len({c.frame_id for c in crops})>1: raise ValueError('Mixed capture batch')
    return crops


def validate_results(crops,checks,inspector):
    checks=tuple(checks)
    if len(checks)!=len(crops): raise ValueError('Batch result count mismatch')
    for crop,check in zip(crops,checks,strict=True):
        if (check.frame_id,check.object_id,check.check_id,check.model)!=(crop.frame_id,crop.object_id,inspector.check_id,inspector.model):
            raise ValueError('Batch result has mismatched identity/order/provenance')
    return checks
