	
## 1-data-extraction

Data used to train this model comes from the National Oceanic and Atmospheric Administration. We used images taken from disaster areas following two hurricanes: Hurricane Dorian in August 2019 and Hurricane Michael in October 2018. Below are links to the original datasets:

### 1. Hurricane Dorian

	Original source: https://storms.ngs.noaa.gov/storms/dorian/index.html

### 2. Hurricane Michael
	
	Original source: https://storms.ngs.noaa.gov/storms/michael/index.html

# Data Extraction Steps

The original data was stored in IBM Cloud Object Storage to be general accessible to our team members. Here are the commands written for Ubuntu Linux command line to interact with IBM Cloud Object Storage.



1. Download original zipped folder 

		curl –O https://stormscdn.ngs.noaa.gov/downloads/20181011a_jpgs.tar

2. Create new directory to store unzipped files 

		mkdir 20181011a_jpgs

3. Unzip tar file

		tar -C 20181011a_jpgs -xvf 20181011a_jpgs.tar

4. Iterate through each file and upload to IBM Cloud Object Storage (contains 2 commands)

		for FILE in 20181011a_jpgs/*
		    do ibmcloud cos put-object --bucket <cfc-image-storage-hurricane-michael> --key $FILE --body $FILE
	    done;
	    
    	for FILE in 20181011a_jpgs/jpgs/*
   	    	do ibmcloud cos put-object --bucket cfc-image-storage-hurricane-michael --key $FILE --body $FILE
	    done;

# Accessing Data in IBM Cloud Object Storage

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