# This script will read corrected landcover maps from my GEE assets, samples them at specified number of points
# per class, and extract Landsat time series for the selected points. Each asset time series is output in a separate
# CSV file.

import ee
ee.Initialize()
Landsat5SR = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR")
Landsat7SR = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")
Landsat8SR = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
start = '2005-01-01'
end = '2018-01-01'
NumPoints = 100
# Will combine scenes from Landsat 5, 7, and 8. But the Landsat 8 bands are different. So I defined below lists
# to match corresponding bands in ETM/OLI and make unified names for bands in output feature list
ETM_selected_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa', 'radsat_qa']
OLI_selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa', 'radsat_qa']
processed_bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa', 'radsat_qa']
output_bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

# This function will mask cloud/cloud shadow and saturated pixels
def maskLandsatSR(image):
    # Bits 3, 5, 7, and 9  in Landsat7/8 show non-clear conditions
    cloudShadowBitMask = 2 ** 3
    cloudsBitMask = 2 ** 5
    cloudsConfBitMask = 2 ** 7
    cirrusConfBitMask = 2 ** 9
    # Get the pixel QA bands
    qa = image.select('pixel_qa')
    radsat = image.select('radsat_qa')
    # All flags should be set to zero, indicating clear conditions.
    mask = (qa.bitwiseAnd(cloudShadowBitMask).eq(0)) and (qa.bitwiseAnd(cloudsBitMask).eq(0)) and \
           (qa.bitwiseAnd(cloudsConfBitMask).eq(0)) and (qa.bitwiseAnd(cirrusConfBitMask).eq(0)) and (radsat.eq(0))
    # Return the masked image, scaled to [0, 1].
    return image.updateMask(mask)

# This function set additional fields for extracted features for each point
def writeFeatures(feature):
    str = ee.String(feature.get('system:index'))
    n = str.index('L')
    str = str.slice(n)
    sensor_id = str.slice(0, 4)
    d = str.slice(12, 20)
    return feature.set('Sensor', sensor_id, 'Date', d)

assets = ee.data.getList({'id':'users/shshheydari/CorrectedLCmaps'})
for asset in assets:
    # Read the input map
    labels = ee.Image(asset['id'])
    # Extract the map sample ID
    id_str = asset['id']
    index = id_str.find('samp')+4
    samp_id = id_str[index:index+7]
    # Mask invalid landcovers (less than 1 or greater than 27)
    mask = labels.lt(28).multiply(labels.gt(0))
    new_labels = labels.updateMask(mask)
    # Set the map's band name to 'LandCover'
    new_labels = new_labels.select(['b1'],['LandCover'])
    # Do sampling on the input map
    points = new_labels.stratifiedSample(numPoints= NumPoints, classBand= 'LandCover', geometries= True)

    #Build merged Landsat collection for the specified time period and region, and unify the band names
    LandsatCol1 = Landsat5SR.filterDate(start, end).filterBounds(points).select(ETM_selected_bands, processed_bands)
    LandsatCol2 = Landsat7SR.filterDate(start, end).filterBounds(points).select(ETM_selected_bands, processed_bands)
    LandsatCol3 = Landsat8SR.filterDate(start, end).filterBounds(points).select(OLI_selected_bands, processed_bands)
    LandsatCol = LandsatCol1.merge(LandsatCol2).merge(LandsatCol3).map(maskLandsatSR)
    LandsatScale = LandsatCol.first().projection().nominalScale().getInfo()

    # Extract series each points
    LandsatSeries = LandsatCol.map(lambda image:
        image.sampleRegions(collection= points, properties= ['LandCover'], scale= LandsatScale)) \
        .flatten() \
        .map(writeFeatures)     # write auxiliary fields

    # Export time series for sampled points
    task = ee.batch.Export.table.toDrive(
      collection= LandsatSeries,
      description= 'LandsatSeries_'+samp_id+'_'+start+'_'+end+'_'+str(NumPoints)+'pts',
      selectors= ['Date', 'LandCover', 'Sensor']+output_bands
    )
    task.start()
