from __future__ import division
import requests
import time as t
from shapely.geometry import LineString
import numpy as np
import json
import pandas as pd
from random import shuffle
from geojson import LineString, Feature, Point, FeatureCollection, dumps
import itertools


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


def synthesize_gps(edges, shape, distribution="normal",
                   stddev=0, uuid='999999'):

    jsonDict = {"uuid": uuid, "trace": []}
    trueRouteCoords = []
    gpsRouteCoords = []
    coords = decode(shape)
    maxCoordIndex = max([edge['end_shape_index'] for edge in edges])
    if maxCoordIndex >= len(coords):
        return None, None
    sttm = t.time() - 86400   # yesterday

    for i, edge in enumerate(edges):

        dist = edge['length']
        speed = edge['speed']

        beginShapeIndex = edge['begin_shape_index']
        endShapeIndex = edge['end_shape_index']
        lon, lat = coords[endShapeIndex]

        if i == 0:
            st_lon, st_lat = coords[beginShapeIndex]
            trueRouteCoords.append([st_lon, st_lat])
        trueRouteCoords.append([lon, lat])

        if stddev > 0:
            avgLat = np.mean(np.array(coords)[:, 1])
            # approx. 111.111 km per deg lon unless very close to the poles
            stddevLon = stddev / 111.111
            # approx 111.111 km * cos(lat) per deg lat
            stddevLat = stddev / (111.111 * np.cos(avgLat))
            lon += np.random.normal(scale=stddevLon)
            lat += np.random.normal(scale=stddevLat)
        dur = dist / speed * 3600.0
        time = sttm + dur
        time = int(round(time))
        if i == 0:
            st_lon, st_lat = coords[beginShapeIndex]
            jsonDict["trace"].append(
                {"lat": st_lat, "lon": st_lon, "time": sttm, "accuracy": min(
                    5, stddev * 1e3)})
            gpsRouteCoords.append([st_lon, st_lat])
        jsonDict["trace"].append(
            {"lat": lat, "lon": lon, "time": time, "accuracy": min(
                5, stddev * 1e3)})
        gpsRouteCoords.append([lon, lat])
        sttm = time

    geojson = FeatureCollection([
        Feature(geometry=LineString(
            trueRouteCoords), properties={
                "stroke": "#ff0000",
                "stroke-width": 2,
                "stroke-opacity": 1}),
        Feature(geometry=LineString(
            gpsRouteCoords), properties={
                "stroke": "#0000ff",
                "stroke-width": 2,
                "stroke-opacity": 1,
                "strokeColor": "#fff"})])
    return jsonDict, geojson


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
        print 'No shape returned'


def get_trace_attrs(shape):

    jsonDict = {
        "encoded_polyline": shape,
        "costing": "auto",
        "directions_options": {
            "units": "kilometers"
        },
        "shape_match": "edge_walk",
        "trace_options": {
            "turn_penalty_factor": 500
        }
    }
    payload = {"json": json.dumps(jsonDict, separators=(',', ':'))}
    baseUrl = 'http://valhalla:8002/trace_attributes?'
    matched = requests.get(baseUrl, params=payload)
    edges = matched.json()['edges']

    return edges, matched.url


def format_edge_df(edges):

    dfEdges = pd.DataFrame(edges)
    dfEdges = dfEdges[[
        'begin_shape_index', 'end_shape_index', 'length',
        'speed', 'traffic_segments']]
    dfEdges['segment_id'] = dfEdges['traffic_segments'].apply(
        lambda x: str(x[0]['segment_id']) if type(x) is list else None)
    dfEdges['starts_segment'] = dfEdges['traffic_segments'].apply(
        lambda x: x[0]['starts_segment'] if type(x) is list else None)
    dfEdges['ends_segment'] = dfEdges['traffic_segments'].apply(
        lambda x: x[0]['starts_segment'] if type(x) is list else None)
    dfEdges['begin_percent'] = dfEdges['traffic_segments'].apply(
        lambda x: x[0]['begin_percent'] if type(x) is list else None)
    dfEdges['end_percent'] = dfEdges['traffic_segments'].apply(
        lambda x: x[0]['end_percent'] if type(x) is list else None)
    dfEdges.drop('traffic_segments', axis=1, inplace=True)

    return dfEdges


def get_reporter_segments(gpsTrace):

    baseUrl = 'http://reporter:8003/report'
    payload = {"json": json.dumps(gpsTrace, separators=(',', ':'))}
    report = requests.get(baseUrl, params=payload)
    if report.status_code == 200:
        segments = report.json()['segments']
    else:
        print(report.reason)
        return None, report.url
    if len(segments) > 0:
        return segments, report.url
    else:
        return 0, report.url


def get_matches(segments, dfEdges):

    matches = dfEdges.copy()
    matches.loc[:, 'matched_segment_id'] = None
    matches.loc[:, 'matched_segment_sttm'] = None
    matches.loc[:, 'matched_segment_endtm'] = None
    for segment in segments:
        matches.loc[
            segment['begin_shape_index']:segment['end_shape_index'],
            'matched_segment_id'] = str(segment['segment_id'])
        matches.loc[
            segment['begin_shape_index'],
            'matched_segment_sttm'] = str(segment['start_time'])
        matches.loc[
            segment['end_shape_index'],
            'matched_segment_endtm'] = str(segment['end_time'])
    matches.loc[:, 'match'] = matches['segment_id'] == \
        matches['matched_segment_id']
    score = np.sum(matches['match']) / \
        len(matches[~pd.isnull(matches['segment_id'])])
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
