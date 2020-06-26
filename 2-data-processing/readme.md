	
## 2-data-processing-readme


# Creating Object Detection Labels

In order to generate our model, it was necessary to annotate a representative sample of images. We used the open-sourced technology CVAT to annotate full sized images. 

## Object Labels

The following categories were used to label objects in the images:

![Test Image 3](/3DTest.png)


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

## Object Label Details

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

## Using CVAT

We used CVAT as an annotation tool to manually annotate images. Image annotations were saved as xml files. See for a sample of the annotated images. 


![Test Image 3](/3DTest.png)

![Test Image 3](/3DTest.png)