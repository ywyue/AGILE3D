#
#
#      0===============================0
#      |    PLY files reader/writer    |
#      0===============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      function to read/write .ply files
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 10/02/2017
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Basic libs
import numpy as np
import sys


# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None


    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])
    
    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.
    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 
    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.
    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])
    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)
    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False    

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False    

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description
    Parameters
    ----------
    name: str
    df: pandas DataFrame
    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element


def slice_points(points, XY_MIN, XY_MAX, num_sample):
    """
    Slice sampled points using bounding box.

    :param mesh:      trimesh, path to the GeoTiff raster file
    :param x:    x upper left corner of patch
    :param y:    y upper left corner of patch
    :param gsdX:    positive float, GSD (ground sampling distance) in X-direction
    :param gsdY:    positive float, GSD (ground sampling distance) in Y-direction
    :param tile_size:    int, tile size in pixels
    :return:        trimesh, cropped mesh
    """

    # dsm_minX = dsm_extent['minX']
    # # dsm_minY = dsm_extent['minY']
    # # dsm_maxX = dsm_extent['maxX']
    # dsm_maxY = dsm_extent['maxY']

    # dsm_gsdX = dsm_extent['gsdX']
    # dsm_gsdY = dsm_extent['gsdY']

    # box_minX = dsm_minX + x*dsm_gsdX
    # box_maxX = box_minX + tile_size*dsm_gsdX

    # box_maxY = dsm_maxY - y*dsm_gsdY
    # box_minY = box_maxY - tile_size*dsm_gsdY

    # B_MIN = [box_minX, box_minY]
    # B_MAX = [box_maxX, box_maxY]

    # crop points within the x-y bounding box
    sliced_points = points[np.all(points[:,:2] >= XY_MIN, axis =1) & np.all(points[:,:2] <= XY_MAX, axis =1)]
    del points

    # shift x and y to [-32, 32]
    sliced_points[:,:2] = sliced_points[:,:2] - XY_MIN - 32

    # add padding in z axis
    # z_padding = 5 

    # inside_z_max = sliced_points[sliced_points[:,3] == 1][:,2].max()
    # outside_z_min = sliced_points[sliced_points[:,3] == 0][:,2].min()

    # # crop points within z extent
    # sliced_points = sliced_points[(sliced_points[:, 2] >= (outside_z_min - z_padding)) & (sliced_points[:,2] <= (inside_z_max + z_padding))]
 
    inside_points = sliced_points[sliced_points[:,3] == 1]
    outside_points = sliced_points[sliced_points[:,3] == 0]

    # print('inside: ' + str(inside_points.shape[0]) + ', outside: '+str(outside_points.shape[0]) + ', all: ' + str(inside_points.shape[0]+ outside_points.shape[0]))

    # nout = outside_points.shape[0]
    # outside_points = outside_points[:num_sample // 2] if nout > num_sample // 2 else outside_points
    # inside_points = inside_points[:num_sample // 2] if nout > num_sample // 2 else inside_points[:(num_sample - nout)]

    nin = inside_points.shape[0]
    inside_points = inside_points[:num_sample // 2] if nin > num_sample // 2 else inside_points
    outside_points = outside_points[:num_sample // 2] if nin > num_sample // 2 else outside_points[:(num_sample - nin)]

    samples = np.concatenate([inside_points, outside_points], 0)

    N = samples.shape[0]
    if N < num_sample:
        sample = np.random.choice(N, num_sample-N)
        dup_data = samples[sample, ...]
        samples = np.concatenate([samples, dup_data], 0)

    # samples[:,[0,1]] = samples[:,[1,0]]
    # labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)
    np.random.shuffle(samples)
    labels = np.expand_dims(samples[:,3], axis=0)
    loss_mask = np.expand_dims(samples[:,4], axis=0)

    if samples.shape[1] == 6:
        weight_mask = np.expand_dims(samples[:,5], axis=0)
    else:
        weight_mask = []
    # labels = np.concatenate([np.ones((inside_points.shape[0],1)), np.zeros((outside_points.shape[0],1))], 0)

    return samples[:,:3].T, labels, loss_mask, weight_mask



def export_points(points, x, y, dsm_extent, tile_size, num_sample, out_ori, out_final):
    """
    Slice sampled points using bounding box.

    :param mesh:      trimesh, path to the GeoTiff raster file
    :param x:    x upper left corner of patch
    :param y:    y upper left corner of patch
    :param gsdX:    positive float, GSD (ground sampling distance) in X-direction
    :param gsdY:    positive float, GSD (ground sampling distance) in Y-direction
    :param tile_size:    int, tile size in pixels
    :return:        trimesh, cropped mesh
    """

    dsm_minX = dsm_extent['minX']
    # dsm_minY = dsm_extent['minY']
    # dsm_maxX = dsm_extent['maxX']
    dsm_maxY = dsm_extent['maxY']

    dsm_gsdX = dsm_extent['gsdX']
    dsm_gsdY = dsm_extent['gsdY']

    box_minX = dsm_minX + x*dsm_gsdX
    box_maxX = box_minX + tile_size*dsm_gsdX

    box_maxY = dsm_maxY - y*dsm_gsdY
    box_minY = box_maxY - tile_size*dsm_gsdY

    B_MIN = [box_minX, box_minY]
    B_MAX = [box_maxX, box_maxY]

    # crop points within the x-y bounding box
    sliced_points = points[np.all(points[:,:2] >= B_MIN, axis =1) & np.all(points[:,:2] <= B_MAX, axis =1)]
    del points

    # shift x and y to [-32, 32]
    # sliced_points[:,:2] = sliced_points[:,:2] - B_MIN - 32

    # add padding in z axis
    z_padding = 5 

    inside_z_max = sliced_points[sliced_points[:,3] == 1][:,2].max()
    outside_z_min = sliced_points[sliced_points[:,3] == 0][:,2].min()

    # crop points within z extent
    sliced_points = sliced_points[(sliced_points[:, 2] >= (outside_z_min - z_padding)) & (sliced_points[:,2] <= (inside_z_max + z_padding))]
 
    write_ply(out_ori, sliced_points, ['x','y','z','label'])

    inside_points = sliced_points[sliced_points[:,3] == 1]
    outside_points = sliced_points[sliced_points[:,3] == 0]

    nout = outside_points.shape[0]
    outside_points = outside_points[:num_sample // 2] if nout > num_sample // 2 else outside_points
    inside_points = inside_points[:num_sample // 2] if nout > num_sample // 2 else inside_points[:(num_sample - nout)]

    samples = np.concatenate([inside_points, outside_points], 0)

    N = samples.shape[0]
    if N < num_sample:
        sample = np.random.choice(N, num_sample-N)
        dup_data = samples[sample, ...]
        samples = np.concatenate([samples, dup_data], 0)

    write_ply(out_final, samples, ['x','y','z','label'])

    # samples[:,[0,1]] = samples[:,[1,0]]
    # labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)
    # labels = np.expand_dims(samples[:,3], axis=0)
    # labels = np.concatenate([np.ones((inside_points.shape[0],1)), np.zeros((outside_points.shape[0],1))], 0)

    # return samples[:,:3].T, labels