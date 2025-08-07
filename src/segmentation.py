def manual_treshold_segmentation(ndvi, 
                                 threshold1, 
                                 threshold2, 
                                 threshold3,
                                 treshold4):
    thresholds = [threshold1, threshold2, threshold3, treshold4]
    for i, threshold in enumerate(thresholds):
        mask = ndvi > threshold
    return mask