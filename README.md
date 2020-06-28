# debris-estimation-call-for-code

## Problem Statement

After a hurricane causes destruction in a region in the United States, federal government organizations, such as the Federal Emergency Management Agency (FEMA), are deployed to provide need-based financial aid to the victims. In order to identify need in a timely matter, these organiations must be able to identify the amount of destruction for each affected area, the density of the population of the area, and the number of people affected in that population. Once identified, victims of hurricane disasters can be further categorized by extent of financial damage using an ordinal measurement system such as: Not Affected, Lightly Affected, Moderately Affected, and Severely Affected.

## Objective

The objective of our project was to use aerial images of areas affected by hurricanes to determine the locations and extent of damage. This project was broken down into the phases below.

- **Phase 1** - Create a computer vision model that identifies damaged structures (based on light, moderate, severe damage) in post-storm aerial imagery. Stitch the imagery together using geocoordinates to produce a "damage report" based on location details (ex. zip code, street). [*COMPLETED*]

- **Phase 2** - Create some way for a user to interact with this model. This is up for discussion. One example - we display it as an ArcGIS map with a download button for the "damage report". [*COMPLETED*]

- **Phase 3** - Merge the "damage report" to information about general population density in these areas. [*NEXT STEPS*]

- **Phase 4** - Merge the "damage report" to information about property values in these areas to predict total cost of damage. [*NEXT STEPS*]

## Code

## [1. Data Extraction](https://github.com/keriwheatley/debris-estimation-call-for-code/blob/master/1-data-extraction/)

Data used to train this model comes from the National Oceanic and Atmospheric Administration. We used images taken from disaster areas following two hurricanes: Hurricane Dorian in August 2019 and Hurricane Michael in October 2018. Below are links to the original datasets:

	Hurricane Dorian: https://storms.ngs.noaa.gov/storms/dorian/index.html
	Hurricane Michael: https://storms.ngs.noaa.gov/storms/michael/index.html

### 1.1 Data Extraction Steps

The original data was stored in IBM Cloud Object Storage to be general accessible to our team members. Here are the commands written for Ubuntu Linux command line to interact with IBM Cloud Object Storage.



1. Download original zipped folder 

		curl –O https://stormscdn.ngs.noaa.gov/downloads/20181011a_jpgs.tar

2. Create new directory to store unzipped files 

		mkdir 20181011a_jpgs

3. Unzip tar file

		tar -C 20181011a_jpgs -xvf 20181011a_jpgs.tar

4. Iterate through each file and upload to IBM Cloud Object Storage (contains 2 commands)

		for FILE in 20181011a_jpgs/*
		    do ibmcloud cos put-object 
			    --bucket <cfc-image-storage-hurricane-michael> 
			    --key $FILE 
			    --body $FILE
	    done;
	    
    	for FILE in 20181011a_jpgs/jpgs/*
   	    	do ibmcloud cos put-object 
   	    		--bucket cfc-image-storage-hurricane-michael 
   	    		--key $FILE 
   	    		--body $FILE
	    done;

### 1.2 Accessing Data in IBM Cloud Object Storage

Here is a step-by-step tutorial for accessing IBM Cloud object storage buckets via API key using pseudo-keys.

	crn: <945f0b4t-3837-4y92-6jb7-4f5h83sd9826>
	region: <us-east>
	access_key_id: <08fbcce4727fd6400913dcc42016afc3>
	secret_access_key: <e82ecd203fcb303653d3af8d073bd4864a385ec4cd786d22>

1. Type `ibmcloud cos config auth` and choose `2. HMAC`
2. Type `ibmcloud cos config hmac` and add the keys above
3. Type `ibmcloud cos config region` and add the region above
4. Type `ibmcloud cos config crn` and add the CRN above
5. Type `ibmcloud cos config list` and verify all inputs are correct
6. Type `ibmcloud cos list-buckets` and see if you have access to the buckets

## [2. Data Processing](https://github.com/keriwheatley/debris-estimation-call-for-code/blob/master/2-data-processing/)

### 2.1 Creating Object Detection Labels

In order to generate our model, it was necessary to annotate a representative sample of images. We used the open-sourced technology [CVAT](https://github.com/opencv/cvat) to annotate full sized images. The following categories were used to label objects in the images:

	1. no-damage-small-structure
	2. lightly-damaged-small-structure
	3. moderately-damaged-small-structure
	4. heavily-damaged-small-structure
	5. no-damage-medium-building
	6. lightly-damaged-medium-building
	7. moderately-damaged-medium-building
	8. heavily-damaged-medium-building
	9. no-damage-large-building
	10. lightly-damaged-large-building
	11. moderately-damaged-large-building
	12. heavily-damaged-large-building
	13. residential-building

#### *Object Label Details*

	1. small-structures
		- A structure no one would live in
		- Sheds
		- Stand-alone garages
		- Pool houses
		
	2. medium-buildings
		- House of any size
		- Larger houses with attached garages
		- Gas stations
		- Small churches
		- Small duplexes
		
	3. large-buildings
		- Attached townhouses
		- Large churches
		- Office buildings
		- Strip malls
		- Apartment complexes
		- Walmart
		- Grocery stores
		
	4. no-damage
		- Little/no debris around the house
		- No roof damage
		
	5. lightly-damaged
		- Tree fell on it but almost no roof damage
		- Little/no debris around the house
		- Broken fence debris against house
		
	6. moderately-damaged
		- Some roof damage, but not a lot
		- Some missing shingles on roof, but major sections look intact
		- Debris around the house
		- Lots of trees fallen on house, even though no visible signs of roof damage
		
	7. heavily-damaged
		- Parts of roof is missing
		- You can see wood studs in some parts
		- Lots of debris around the house

### 2.2 Using CVAT

We used [CVAT](https://github.com/opencv/cvat) as an annotation tool to manually annotate images. CVAT was set up on an IBM Cloud Virtual Machine using IBM Cloud Object Storage. 

Image annotations were saved as xml files. See for a sample of annotated images: [Sample Images](https://github.com/keriwheatley/debris-estimation-call-for-code/tree/master/2-data-processing/code-crop-annotated-images/sample_images_dir) and [Sample Annotations](https://github.com/keriwheatley/debris-estimation-call-for-code/tree/master/2-data-processing/code-crop-annotated-images/sample_annotations_dir)

2.2.1 CVAT allows user to create annotation tasks, assign to other users, and track progress.
![CVAT Image 1](https://github.com/keriwheatley/debris-estimation-call-for-code/blob/master/2-data-processing/cvat-image-1.png)

2.2.2 User specifies class labels and attributes.
![CVAT Image 2](https://github.com/keriwheatley/debris-estimation-call-for-code/blob/master/2-data-processing/cvat-image-2.png)

2.2.3 Here is an example of the labeling for an image.
![CVAT Image 3](https://github.com/keriwheatley/debris-estimation-call-for-code/blob/master/2-data-processing/cvat-image-3.png)

### 2.3 Cropping Annotated Images

We created a function to crop annotated images into smaller tiles to be used for model training. See this file for information on running the function: [crop_images.py](https://github.com/keriwheatley/debris-estimation-call-for-code/blob/master/2-data-processing/code-crop-annotated-images/crop_images.py)

## [3. Train Model](https://github.com/keriwheatley/debris-estimation-call-for-code/blob/master/3-train-model/)

## [4. Display Results](https://github.com/keriwheatley/debris-estimation-call-for-code/blob/master/4-display-results/)

## Misc Exploration

1. Watson Studio

2. Roboflow.ai

## Team
1. [Lamogha Chiazor](https://www.linkedin.com/in/lamogha/), IBM, Research Software Engineer

2. [Tetiana Korchak](https://www.linkedin.com/in/tetianakorchak/), IBM,  Test Automation Engineer

3. [Krish Rekapalli](https://www.linkedin.com/in/krrish1729/), IBM,  Data Scientist

4. [Jon Wesneski](https://www.linkedin.com/in/jon-wesneski-01110789/), IBM, Test Automation Engineer

5. [Keri Wheatley](https://www.linkedin.com/in/keri-wheatley/), IBM, Data Scientist