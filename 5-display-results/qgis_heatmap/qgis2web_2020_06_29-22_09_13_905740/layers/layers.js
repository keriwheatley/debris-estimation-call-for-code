var wms_layers = [];


        var lyr_GoogleMaps_0 = new ol.layer.Tile({
            'title': 'Google Maps',
            'type': 'base',
            'opacity': 1.000000,
            
            
            source: new ol.source.XYZ({
    attributions: ' ',
                url: 'http://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
            })
        });
var format_cfcresults_1 = new ol.format.GeoJSON();
var features_cfcresults_1 = format_cfcresults_1.readFeatures(json_cfcresults_1, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_cfcresults_1 = new ol.source.Vector({
    attributions: ' ',
});
jsonSource_cfcresults_1.addFeatures(features_cfcresults_1);
var lyr_cfcresults_1 = new ol.layer.Heatmap({
                declutter: true,
                source:jsonSource_cfcresults_1, 
                radius: 10 * 2,
                gradient: ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],
                blur: 15,
                shadow: 250,
    weight: function(feature){
        var weightField = 'damage_count';
        var featureWeight = feature.get(weightField);
        var maxWeight = 4;
        var calibratedWeight = featureWeight/maxWeight;
        return calibratedWeight;
    },
                title: 'cfc-results'
            });

lyr_GoogleMaps_0.setVisible(true);lyr_cfcresults_1.setVisible(true);
var layersList = [lyr_GoogleMaps_0,lyr_cfcresults_1];
