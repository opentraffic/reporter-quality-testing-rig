from __future__ import division
import requests
import time as t
from shapely.geometry import LineString, MultiPoint, MultiLineString
import numpy as np
import json
import pandas as pd
from random import shuffle
from geojson import Feature, FeatureCollection
import itertools
from pyproj import Proj, transform
from scipy.stats import norm
from ipywidgets import Layout
from ipyleaflet import (
    Map,
    TileLayer,
    Circle,
    GeoJSON
)


def convert_coords_to_meters(coords, localEpsg, inputOrder='lonlat'):
    if inputOrder == 'latlon':
        indices = [1, 0]
    elif inputOrder == 'lonlat':
        indices = [0, 1]
    else:
        print('"inputOrder" param cannot be processed')
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:{0}'.format(localEpsg))
    projCoords = [
        transform(inProj, outProj, coord[indices[0]], coord[indices[1]])
        for coord in coords]
    return projCoords


def convert_coords_to_lat_lon(coords, localEpsg, inputOrder='xy'):
    if inputOrder == 'yx':
        indices = [1, 0]
    elif inputOrder == 'xy':
        indices = [0, 1]
    else:
        print('"inputOrder" param cannot be processed')
    inProj = Proj(init='epsg:{0}'.format(localEpsg))
    outProj = Proj(init='epsg:4326')
    projCoords = [
        transform(inProj, outProj, coord[indices[0]], coord[indices[1]])
        for coord in coords]
    return projCoords


def decode(encoded):
    inv = 1.0 / 1e6
    decoded = []
    previous = [0, 0]
    i = 0
    while i < len(encoded):
        ll = [0, 0]
        for j in [0, 1]:
            shift = 0
            byte = 0x20
            while byte >= 0x20:
                byte = ord(encoded[i]) - 63
                i += 1
                ll[j] |= (byte & 0x1f) << shift
                shift += 5
            ll[j] = previous[j] + \
                (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
            previous[j] = ll[j]
        decoded.append(
            [float('%.6f' % (ll[1] * inv)), float('%.6f' % (ll[0] * inv))])
    return decoded


def get_coords_per_second(shapeCoords, edges, localEpsg):
    mProj = Proj(init='epsg:{0}'.format(localEpsg))
    llProj = Proj(init='epsg:4326')
    coords = shapeCoords
    projCoords = convert_coords_to_meters(coords, localEpsg=localEpsg)
    for i, edge in enumerate(edges):
        subSegmentCoords = []
        if i == 0:
            subSegmentCoords.append(coords[edge['begin_shape_index']])
        dist = edge['length']
        distMeters = dist * 1e3
        speed = edge['speed']
        mPerSec = speed * 1e3 / 3600.0
        beginShapeIndex = edge['begin_shape_index']
        endShapeIndex = edge['end_shape_index']
        if (beginShapeIndex >= len(coords) - 1) | \
           (endShapeIndex >= len(coords)):
            continue
        line = LineString(projCoords[beginShapeIndex:endShapeIndex + 1])
        seconds = 0
        while mPerSec * seconds < distMeters:
            seconds += 1
            newPoint = line.interpolate(mPerSec * seconds)
            newLon, newLat = transform(mProj, llProj, newPoint.x, newPoint.y)
            subSegmentCoords.append([newLon, newLat])
        if i == len(edges) - 1:
            subSegmentCoords.append(coords[edge['end_shape_index']])
        edge['oneSecCoords'] = subSegmentCoords
        edge['numOneSecCoords'] = len(subSegmentCoords)
    return edges


def synthesize_gps(dfEdges, shapeCoords, localEpsg, mode="auto",
                   distribution="normal", noise=0, sampleRate=1,
                   uuid="999999"):

    accuracy = round(min(100, norm.ppf(0.95, loc=0, scale=max(1, noise))), 2)
    mProj = Proj(init='epsg:{0}'.format(localEpsg))
    llProj = Proj(init='epsg:4326')
    jsonDict = {"uuid": uuid, "trace": [], "match_options": {
        "mode": mode,
        "turn_penalty_factor": 500}}
    trueRouteCoords = []
    resampledCoords = []
    gpsRouteCoords = []
    displacementLines = []
    lonAdjs = []
    latAdjs = []
    noiseLookback = int(np.ceil(10 / sampleRate))
    sttm = int(t.time()) - 86400   # yesterday
    seconds = 0
    shapeIndexCounter = 0
    for i, edge in dfEdges.iterrows():
        if i == 0:
            trueCoords = shapeCoords[edge['begin_shape_index']]
            trueRouteCoords.append(trueCoords)
        trueCoords = shapeCoords[edge['end_shape_index']]
        trueRouteCoords.append(trueCoords)
        edgeShapeIndices = []
        for j, coordPair in enumerate(edge['oneSecCoords']):
            if (not seconds % sampleRate) | (
                (i + 1 == len(dfEdges)) &
                (j + 1 == len(edge['oneSecCoords']))
            ):
                lon, lat = coordPair
                resampledCoords.append([lon, lat])
                if noise > 0:
                    projLon, projLat = transform(llProj, mProj, lon, lat)
                    while True:
                        lonAdj = np.random.normal(scale=noise)
                        latAdj = np.random.normal(scale=noise)
                        if shapeIndexCounter == 0:
                            noiseQuad = [np.sign(lonAdj), np.sign(latAdj)]
                            break
                        elif [np.sign(lonAdj), np.sign(latAdj)] == noiseQuad:
                            break
                    lonAdjs.append(lonAdj)
                    latAdjs.append(latAdj)
                    newProjLon = projLon + np.mean(lonAdjs[-noiseLookback:])
                    newProjLat = projLat + np.mean(latAdjs[-noiseLookback:])
                    projLon, projLat = newProjLon, newProjLat
                    lon, lat = transform(mProj, llProj, projLon, projLat)
                time = sttm + seconds
                lat = round(lat, 6)
                lon = round(lon, 6)
                jsonDict["trace"].append({
                    "lat": lat, "lon": lon, "time": time,
                    "accuracy": accuracy})
                gpsRouteCoords.append([lon, lat])
                displacementLines.append([coordPair, [lon, lat]])
                edgeShapeIndices.append(shapeIndexCounter)
                shapeIndexCounter += 1
            seconds += 1
        if len(edgeShapeIndices) > 0:
            dfEdges.loc[
                i, 'begin_resampled_shape_index'] = min(edgeShapeIndices)
            dfEdges.loc[
                i, 'end_resampled_shape_index'] = max(edgeShapeIndices)

    gpsShape = [{"lat": d["lat"], "lon": d["lon"]} for d in jsonDict['trace']]
    _, matches, _ = get_trace_attrs(
        gpsShape, encoded=False, gpsAccuracy=accuracy,
        output='matches')
    gpsMatchCoords = matches

    geojson = FeatureCollection([
        Feature(geometry=LineString(
            trueRouteCoords), properties={"style": {
                "color": "#ff0000",
                "weight": "3px"},
                "name": "true_route_coords"}),
        Feature(geometry=MultiPoint(
            resampledCoords), properties={"style": {
                "color": "#ff0000",
                "weight": "3px"},
                "name": "resampled_coords"}),
        Feature(geometry=MultiPoint(
            gpsRouteCoords), properties={"style": {
                "color": "#0000ff",
                "weight": "3px"},
                "name": "gps_coords"}),
        Feature(geometry=MultiLineString(
            displacementLines), properties={"style": {
                "color": "#000000",
                "weight": "1px",
                "name": "displacement_lines"}}),
        Feature(geometry=LineString(
            gpsMatchCoords), properties={"style": {
                "fillcolor": "#0000ff",
                "weight": "3px",
                "name": "matched_gps_route"}})])

    return dfEdges, jsonDict, geojson


def get_route_shape(stLat, stLon, endLat, endLon):

    jsonDict = {"locations": [{
        "lat": stLat, "lon": stLon, "type": "break"},
        {
        "lat": endLat, "lon": endLon, "type": "break"}],
        "costing": "auto",
        "id": "my_work_route"}
    payload = {"json": json.dumps(jsonDict, separators=(',', ':'))}
    baseUrl = 'http://valhalla:8002/route'
    route = requests.get(baseUrl, params=payload)
    shape = route.json()['trip']['legs'][0]['shape']

    if route.status_code == 200:
        return shape, route.url
    else:
        return None, 'No shape returned.'


def get_trace_attrs(shape, encoded=True, shapeMatch='map_snap',
                    gpsAccuracy=5, output='edges'):
    if encoded:
        shape_param = 'encoded_polyline'
    else:
        shape_param = 'shape'

    jsonDict = {
        shape_param: shape,
        "costing": "auto",
        "directions_options": {
            "units": "kilometers"
        },
        "shape_match": shapeMatch,
        "trace_options": {
            "turn_penalty_factor": 500,
            "gps_accuracy": gpsAccuracy
        }
    }
    payload = {"json": json.dumps(jsonDict, separators=(',', ':'))}
    baseUrl = 'http://valhalla:8002/trace_attributes?'
    matched = requests.get(baseUrl, params=payload)
    edges = matched.json()['edges']
    matchedPts = decode(matched.json()['shape'])
    return edges, matchedPts, matched.url


def format_edge_df(edges):

    dfEdges = pd.DataFrame(edges)
    dfEdges = dfEdges[[
        'begin_shape_index', 'end_shape_index', 'length',
        'speed', 'density', 'traffic_segments', 'oneSecCoords']]
    dfEdges['segment_id'] = dfEdges['traffic_segments'].apply(
        lambda x: str(x[0]['segment_id']) if type(x) is list else None)
    dfEdges['num_segments'] = dfEdges['traffic_segments'].apply(
        lambda x: len(x) if type(x) is list else 0)
    dfEdges['starts_segment'] = dfEdges['traffic_segments'].apply(
        lambda x: x[0]['starts_segment'] if type(x) is list else None)
    dfEdges['ends_segment'] = dfEdges['traffic_segments'].apply(
        lambda x: x[0]['starts_segment'] if type(x) is list else None)
    dfEdges['begin_percent'] = dfEdges['traffic_segments'].apply(
        lambda x: x[0]['begin_percent'] if type(x) is list else None)
    dfEdges['end_percent'] = dfEdges['traffic_segments'].apply(
        lambda x: x[0]['end_percent'] if type(x) is list else None)
    dfEdges.drop('traffic_segments', axis=1, inplace=True)
    dfEdges['begin_resampled_shape_index'] = None
    dfEdges['end_resampled_shape_index'] = None
    return dfEdges


def get_reporter_segments(gpsTrace):

    baseUrl = 'http://reporter:8003/report'
    payload = {"json": json.dumps(gpsTrace, separators=(',', ':'))}
    report = requests.get(baseUrl, params=payload)
    # report = requests.post(baseUrl, json=gpsTrace)
    if report.status_code == 200:
        segments = report.json()['segment_matcher']['segments']
    else:
        return None, report.reason
    if len(segments) > 0:
        return segments, report.url
    else:
        return 0, report.url


def get_matches(segments, dfEdges):

    segDf = pd.DataFrame(segments, columns=[
        'begin_shape_index', 'end_shape_index', 'end_time', 'internal',
        'segment_id', 'length', 'start_time'])
    segDf = segDf[~pd.isnull(segDf['segment_id'])]
    segDf.loc[:, 'segment_id'] = segDf['segment_id'].astype(int).astype(str)
    matches = pd.merge(
        dfEdges, segDf, on='segment_id', how='outer', suffixes=(
            '_tr_attr', '_rprtr'))
    segMatches = segDf['segment_id'].isin(dfEdges['segment_id'])
    edgeMatches = dfEdges['segment_id'].isin(segDf['segment_id'])
    score = (np.sum(segMatches) + np.sum(edgeMatches)) / \
        (len(segMatches) + len(edgeMatches))
    return matches, score


def get_POI_routes(locString, numResults, apiKey):

    url = 'https://maps.googleapis.com/maps/api/place' + \
        '/textsearch/json?query={0}&radius={1}&key={2}'
    url = url.format("{0} point of interest".format(locString), 25000, apiKey)
    r = requests.get(url)
    POIs = [{x['name']: {
        "lat": x['geometry']['location']['lat'],
        "lon": x['geometry']['location']['lng']}} for x in r.json()['results']]
    routeList = list(itertools.combinations(POIs, 2))
    shuffle(routeList)
    numResults = min(len(routeList), numResults)
    routeList = routeList[:numResults]
    return routeList


def get_routes_by_length(cityStr, minRouteLength, maxRouteLength,
                         numResults, apiKey):

    mapzenKey = apiKey

    baseUrl = 'https://search.mapzen.com/v1/search?'
    cityQuery = 'sources={0}&text={1}&api_key={2}&layer={3}&size=1'.format(
        'whosonfirst', cityStr, mapzenKey, 'locality')
    city = requests.get(baseUrl + cityQuery)
    cityID = city.json()['features'][0]['properties']['source_id']
    bbox = city.json()['bbox']

    goodRoutes = []

    baseUrlCity = 'https://whosonfirst-api.mapzen.com?' + \
        'api_key={0}&'.format(mapzenKey)

    venueQuery = 'method={0}&id={1}&placetype={2}'.format(
        'whosonfirst.places.getDescendants', cityID, 'venue') + \
        '&page=1&per_page=2000'
    venues = requests.get(baseUrlCity + venueQuery)
    venueIDs = [x['wof:id'] for x in venues.json()['places']]
    shuffle(venueIDs)
    venueListBreakPoints = range(0, len(venueIDs), 20)
    venueListIter = 0
    sttm = t.time()

    while (len(goodRoutes) < numResults) & (t.time() - sttm < 300):
        venueChunkIdx = venueListBreakPoints[venueListIter]
        POIs = []
        baseUrlVenues = 'https://whosonfirst-api.mapzen.com?' + \
            'api_key={0}&page=1&per_page=1&'.format(mapzenKey) + \
            'extras=geom:latitude,geom:longitude'
        for venueID in venueIDs[venueChunkIdx:venueChunkIdx + 20]:
            geoQuery = '&method={0}&id={1}&placetype={2}'.format(
                'whosonfirst.places.getInfo', venueID, 'venue')
            info = requests.get(baseUrlVenues + geoQuery).json()['place']
            POIs.append({info['wof:name']: {
                "lat": info['geom:latitude'],
                "lon": info['geom:longitude']}})
        routeList = list(itertools.combinations(POIs, 2))
        for route in routeList:
            stLat = route[0].values()[0]["lat"]
            stLon = route[0].values()[0]["lon"]
            endLat = route[1].values()[0]["lat"]
            endLon = route[1].values()[0]["lon"]
            jsonDict = {"locations": [{
                "lat": stLat, "lon": stLon, "type": "break"},
                {
                "lat": endLat, "lon": endLon, "type": "break"}],
                "costing": "auto",
                "id": "my_work_route"}
            payload = {"json": json.dumps(jsonDict, separators=(',', ':'))}
            baseUrlValhalla = 'http://valhalla:8002/route'
            routeCheck = requests.get(baseUrlValhalla, params=payload)
            length = routeCheck.json()['trip']['summary']['length']
            if minRouteLength < length < maxRouteLength:
                goodRoutes.append(route)
        venueListIter += 1

    shuffle(goodRoutes)
    goodRoutes = goodRoutes[:numResults]
    return goodRoutes


def generate_route_map(pathToGeojson, zoomLevel=11):

    with open(pathToGeojson, "r") as f:
        data = json.load(f)
    ctrLon, ctrLat = np.mean(
        np.array(data['features'][0]['geometry']['coordinates']), axis=0)
    url = "http://stamen-tiles-{s}.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png"
    provider = TileLayer(url=url, opacity=1)
    center = [ctrLat, ctrLon]
    m = Map(default_tiles=provider, center=center, zoom=zoomLevel)
    m.layout = Layout(width='100%', height='800px')
    trueRouteCoords, resampledCoords, gpsRouteCoords, \
        displacementLines, gpsMatchCoords = data['features']
    g = GeoJSON(data=FeatureCollection(
        [trueRouteCoords, gpsMatchCoords]))
    m.add_layer(g)
    for coords in resampledCoords['geometry']['coordinates']:
        cm = Circle(
            location=coords[::-1], radius=10, weight=1, color='#ff0000',
            opacity=1.0, fill_opacity=0.4, fill_color='#ff0000')
        m.add_layer(cm)
    for coords in gpsRouteCoords['geometry']['coordinates']:
        cm = Circle(
            location=coords[::-1], radius=10, weight=1, color='#0000ff',
            opacity=1.0, fill_opacity=0.4, fill_color='#0000ff')
        m.add_layer(cm)
    g = GeoJSON(data=displacementLines)
    m.add_layer(g)
    return m


def getLineFromPoints(point1, point2):

    x1, y1 = point1
    x2, y2 = point2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    return m, b


def getPerpLineThruEndpt(slope, endpoint):

    m = -1 / slope
    x, y = endpoint
    b = y - (m * x)
    return m, b


def getBoundaryLineCoords(slope, intercept, midpoint, noise, localEpsg='2768'):

    midx, midy = midpoint
    tmpLeftX = midx - (noise * 2)
    tmpLeftY = slope * tmpLeftX + intercept
    leftBisect = LineString([[midx, midy], [tmpLeftX, tmpLeftY]])
    leftEndPt = leftBisect.interpolate(noise * 2)
    tmpRightX = midx + (noise * 2)
    tmpRightY = slope * tmpRightX + intercept
    rightBisect = LineString([[midx, midy], [tmpRightX, tmpRightY]])
    rightEndPt = rightBisect.interpolate(noise * 2)
    boundaryLine = LineString([leftEndPt, midpoint, rightEndPt])
    boundaryLineCoords = [
        [endpt.xy[0][0], endpt.xy[1][0]] for endpt in boundaryLine.boundary]
    mProj = Proj(init='epsg:{0}'.format(localEpsg))
    llProj = Proj(init='epsg:4326')
    boundaryLineCoords = [
        transform(mProj, llProj, x[0], x[1]) for x in boundaryLineCoords]
    return boundaryLineCoords


def checkForBackTrack(lastSegCoords, newPoint, noise):
    lastSegSlope, lastSegIntercept = getLineFromPoints(
        lastSegCoords[0], lastSegCoords[1])
    perpLineSlope, perpLineIntercept = getPerpLineThruEndpt(
        lastSegSlope, lastSegCoords[1])
    bl = getBoundaryLineCoords(
        perpLineSlope, perpLineIntercept, lastSegCoords[1], noise)
    firstPtPos = np.sign(
        perpLineSlope * lastSegCoords[0][0] +
        perpLineIntercept - lastSegCoords[0][1])
    newPtPos = np.sign(
        perpLineSlope * newPoint[0] + perpLineIntercept - newPoint[1])
    if firstPtPos == newPtPos:
        return True, bl
    else:
        return False, bl
