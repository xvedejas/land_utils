#!/usr/bin/python3
"""
Note: all coordinates are in the form of (lon, lat) in units of degrees * 10^-5.
Google calls these 'E5' coordinates.
"""
import urllib.request as req
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def array(x):
    return np.array(x, dtype=np.int_)

def zeros(x):
    return np.zeros(x, dtype=np.int_)

def linspace(x, y, z):
    return np.linspace(x, y, z, dtype=np.int_)

def polyline(x):
    if x == 0.0:
        return 'A'
    if x < 0:
        e = (2**32 - 1) - ((2**31 + x) << 1)
    else:
        e = x << 1
    e = bin(e)[2:] # get binary form and remove the leading '0b'
    
    # Make e have length divisible by 5
    rem = len(e) % 5
    if e.startswith(rem * '0'):
        e = e[rem:]
    else:
        e = '0' * (5 - rem) + e
    chunks = [e[(i-1)*5:i*5] for i in range(len(e) // 5, 0, -1)]
    
    for i, chunk in enumerate(chunks[:-1]):
        chunks[i] = chr((int(chunk, base=2) | 0x20) + 63)
    chunks[-1] = chr(int(chunks[-1], base=2) + 63)
    return ''.join(chunks)

def polyline_points(points):
    """
    See:
    https://developers.google.com/maps/documentation/utilities/polylinealgorithm
    Accepts an Nx2 numpy array.
    
    Although this function appears to be 'correct', sometimes the data received
    from google seems to drift with a large number of points. It's unclear what
    is going on.
    """
    assert points.shape[1] == 2
    string = ''
    previous = None
    for point in points:
        if previous is not None:
            string += (polyline(point[0] - previous[0]) +
                       polyline(point[1] - previous[1]))
        else:
            string += polyline(point[0]) + polyline(point[1])
        previous = point
    return string

def test_polyline_points():
    """
    Make sure our algorithm to generate polyline strings is working correctly.
    """
    points = array(((3850000, -12020000),))
    encoded = polyline_points(points)
    assert encoded == '_p~iF~ps|U'
    
    points = array(((255200, -550300),))
    encoded = polyline_points(points)
    assert encoded == '_mqNvxq`@'
    
    points = array(((3850000, -12020000),
                       (4070000, -12095000),
                       (4325200, -12645300)))
    encoded = polyline_points(points)
    assert encoded == '_p~iF~ps|U_ulLnnqC_mqNvxq`@'

# We can maximum request 512 data points at a time from google
points_per_request = 512

def request_data(start, end, steps, request):
    """
    Request data from Google over an area.
    """
    lat0, lon0 = start
    lat1, lon1 = end
    latsteps, lonsteps = steps
    
    # Find all (lat,lon) coordinates as a 2xN numpy array
    lats = linspace(lat0, lat1, latsteps)
    lons = linspace(lon0, lon1, lonsteps)
    lats, lons = np.meshgrid(lats, lons)
    coords = array((lats.flatten(), lons.flatten())).T
    ncoords = len(coords)
    data = zeros(ncoords)

    for i in range(ncoords // points_per_request):
        data[i*points_per_request:(i+1)*points_per_request] = \
            request(coords[i*points_per_request:(i+1)*points_per_request])
    leftover = ncoords % points_per_request
    if leftover > 0:
        data[ncoords-leftover:ncoords] = request(coords[ncoords-leftover:ncoords])
    return data.reshape(steps), coords.reshape((latsteps, lonsteps, 2))

def request_elevation_data(coords):
    """
    Use this as the last argument to request_data to get elevation data
    """
    assert coords.shape[1] == 2
    n_coords = coords.shape[0]
    assert n_coords <= points_per_request
    data = zeros(n_coords)
    # Connect to the google maps API
    api = 'http://maps.googleapis.com/maps/api/elevation/xml?locations=enc:%s'
    url = api % polyline_points(coords)
    response = req.urlopen(url)
    xml = response.read()
    root = ET.fromstring(xml)
    data = zeros(n_coords)
    for i, result in enumerate(root.findall('result')):
        #~ loc = result.find('location')
        #~ lat = float(loc.find('lat').text)
        #~ lon = float(loc.find('lng').text)
        data[i] = float(result.find('elevation').text)
    # Sometimes there's weird negative spikes in this data, so we'll remove
    # everything below sea level for now.
    data[data < 0] = 0
    return data

def plot_data(data, start, end, steps):
    """
    Plot 1d spatial data.
    """
    lat0, lon0 = start
    lat1, lon1 = end
    latsteps, lonsteps = steps
    # By virtue of how we store it as a numpy array, our data set is rotated
    # 90deg clockwise from how imshow wants it, so use np.rot90:
    # Also, use a decent colormap; see:
    # https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
    plt.imshow(np.rot90(data), 'cubehelix', interpolation='none')
    plt.colorbar()
    
    plt.xticks((0, lonsteps - 1), (lon0, lon1), fontsize=15)
    plt.yticks((0, latsteps - 1), (lat1, lat0), fontsize=15)
    
    plt.xlabel('Longitude', fontsize=15)
    plt.ylabel('Latitude', fontsize=15)
    
    plt.show()
    plt.savefig('example.svg')

if __name__ == '__main__':
    #~ test_polyline_points()
    
    # For demonstration purposes, we use the easily recognizable area around
    # Gualala, CA.
    start = (3874000, -12355000)
    end = (3880000, -12349000)
    # Make the steps number equal to the points_per_request for best results.
    # steps = (512, 512) will generate the sample picture but take a while.
    steps = (32, 32)

    elevations, coords = request_data(start, end, steps, request_elevation_data)
    plot_data(elevations, start, end, steps)
