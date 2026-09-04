import ee

def create_annual_medoid(collection: ee.ImageCollection, year: int) -> ee.Image:
    """
    Creates an annual Medoid composite for a given year.
    The Medoid is the observation that minimizes the sum of squared differences 
    to the median across all bands, ensuring the resulting pixel is an actual observation.
    
    Args:
        collection (ee.ImageCollection): The pre-processed and harmonized collection.
        year (int): The target year.
        
    Returns:
        ee.Image: The Medoid composite for the year.
    """
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    
    yearly_col = collection.filterDate(start, end)
    
    # Calculate the median of all bands
    median = yearly_col.median()
    
    def calc_distance(img: ee.Image) -> ee.Image:
        diff = img.subtract(median)
        # Calculate Euclidean distance squared
        dist = diff.pow(2).reduce(ee.Reducer.sum()).sqrt()
        return img.addBands(dist.rename('distance'))
    
    # Sort by distance (ascending) so the smallest distance (medoid) is on top
    medoid = (yearly_col.map(calc_distance)
              .sort('distance', True)
              .mosaic()
              .select(['SR_B.*']))  # Select spectral bands, drop 'distance'
              
    return medoid.set('system:time_start', ee.Date(start).millis())

def create_annual_timeseries(collection: ee.ImageCollection, start_year: int, end_year: int) -> ee.ImageCollection:
    """
    Creates an annual Medoid time series from a given start year to an end year.
    
    Args:
        collection (ee.ImageCollection): The harmonized collection.
        start_year (int): The starting year.
        end_year (int): The ending year.
        
    Returns:
        ee.ImageCollection: A collection of annual medoid composites.
    """
    years = ee.List.sequence(start_year, end_year)
    
    def get_yearly(year: ee.Number) -> ee.Image:
        return create_annual_medoid(collection, year)
        
    medoid_list = years.map(get_yearly)
    return ee.ImageCollection.fromImages(medoid_list)
