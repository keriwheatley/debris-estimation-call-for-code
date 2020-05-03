# debris-estimation-call-for-code

## Problem Statement

After a hurricane causes destruction in a region in the United States, FEMA (Federal Emergency Management Agency) is deployed to provide financial aid to the victims. FEMA tries to concentrate aid in the areas of most need. FEMA needs to be able to identify the amount of destruction for each affected area, the density of the population of the area, and the number of people affected in that population. FEMA would like to categorize people in different areas as: Not affected, Lightly Affected, Moderately Affected, Severely Affected.

## Objective

Our project can take on multiple objectives based on the amount of time we have before the project deadline of June 30. For anything we don't build out, we can speak to as our "next steps".

Phase 1 - Create a computer vision model that identifies damaged structures (based on light, moderate, severe damage) in post-storm aerial imagery. Stitch the imagery together using geocoordinates to produce a "damage report" based on location details (ex. zip code, street).

Phase 2 - Create some way for a FEMA employee to interact with this model. This is up for discussion. One example - we display it as an ArcGIS map with a download button for the "damage report".

Phase 3 - Merge the "damage report" to information about general population density in these areas.

Phase 4 - Merge the "damage report" to information about property values in these areas to predict total cost of damage.


Feel free to add to this README as you find interesting resources.

## Resources

### 1. Research Paper about fallen tree analysis
	https://www.fs.fed.us/rm/pubs_journals/2012/rmrs_2012_szantoi_z001.pdf

### 2. Orginal Model from FEMA
	https://communities.geoplatform.gov/disasters/hurricane-michael-debris-detection/

### 3. Model Download link
	https://disasters.geoplatform.gov/publicdata/NationalDisasters/2018/HurricaneMichael/Data/

### 4. ArcGIS Hosted
	https://kwheatley9.maps.arcgis.com/home/item.html?id=2246a9a841434005b1b06ceaba8d8505

### 5. OpenCV CVAT for image annotation
	https://github.com/opencv/cvat#share-path
	
## Data

### 1. Hurricane Dorian

	Original source: https://storms.ngs.noaa.gov/storms/dorian/index.html

### 2. Hurricane Irma

	Original source: https://storms.ngs.noaa.gov/storms/irma/index.html

### 3. Hurricane Matthew
	
	Original source: https://geodesy.noaa.gov/storm_archive/storms/matthew/index.html

### 4. Hurricane Maria

	Original source: https://storms.ngs.noaa.gov/storms/maria/index.html

### 5. Hurricane Michael
	
	Original source: https://storms.ngs.noaa.gov/storms/michael/index.html
	Google Drive access: https://drive.google.com/drive/folders/1hTamQn1sb1AtWa6yNrkiKDkyFUnI3uvJ?usp=sharing

### 6. Hurricane Nate
	
	Original source: https://storms.ngs.noaa.gov/storms/nate/index.html

### 7. Hurricane Sandy
	Original source: https://geodesy.noaa.gov/storm_archive/storms/sandy/index.html
	Google Drive access: https://drive.google.com/drive/folders/16-yCf_cgl7ifxqGugaoAwCqsHbf3jbge?usp=sharing
